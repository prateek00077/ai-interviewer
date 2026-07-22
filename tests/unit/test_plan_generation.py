"""Validation of model-generated plans, and the JSON extraction around it.

No API calls: these pin the rules that decide whether model output is
trustworthy. Every one of them exists because the failure it prevents is
silent -- a rubric that sums wrong produces plausible scores that are simply
incorrect, and nothing downstream notices.
"""

from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from app.integrations.nim_client import extract_json
from app.modules import prompts
from app.modules.question_plan.generator import GeneratedPlan


def _criterion(name: str, weight: str, **overrides) -> dict:
    return {
        "name": name,
        "description": f"measures {name}",
        "weight": weight,
        "descriptors": {"1": "weak answer", "3": "adequate answer", "5": "strong answer"},
        **overrides,
    }


def _plan(criteria: list[dict], questions: list[dict] | None = None) -> dict:
    return {
        "criteria": criteria,
        "questions": questions
        or [{"body": "Walk me through the Kafka migration.", "competency": criteria[0]["name"]}],
    }


THREE_EVEN = [_criterion("depth", "0.4"), _criterion("breadth", "0.3"), _criterion("comms", "0.3")]


# --- Weights ----------------------------------------------------------------


def test_exact_weights_pass_through_unchanged():
    plan = GeneratedPlan.model_validate(_plan(THREE_EVEN))
    assert [c.weight for c in plan.criteria] == [Decimal("0.4"), Decimal("0.3"), Decimal("0.3")]


def test_weights_that_overshoot_are_rescaled_not_rejected():
    """MEASURED: Nemotron-3-Nano returns 1.05 and repeats it when shown the error.

    The model's contribution is the relative importance of the criteria;
    the normalisation constant is arithmetic we can do ourselves.
    """
    plan = GeneratedPlan.model_validate(
        _plan([_criterion("aa", "0.5"), _criterion("bb", "0.3"), _criterion("cc", "0.25")])
    )
    assert sum(c.weight for c in plan.criteria) == Decimal("1")


def test_rescaling_preserves_relative_importance():
    plan = GeneratedPlan.model_validate(
        _plan([_criterion("big", "0.6"), _criterion("mid", "0.3"), _criterion("small", "0.3")])
    )
    weights = {c.name: c.weight for c in plan.criteria}
    # 0.6 : 0.3 : 0.3 -> 0.5 : 0.25 : 0.25
    assert weights["big"] == Decimal("0.5000")
    assert weights["mid"] == weights["small"]


def test_rounding_drift_lands_on_the_largest_weight():
    """Three-way splits do not divide evenly at four decimal places."""
    plan = GeneratedPlan.model_validate(
        _plan([_criterion("aa", "0.33"), _criterion("bb", "0.33"), _criterion("cc", "0.33")])
    )
    assert sum(c.weight for c in plan.criteria) == Decimal("1")


def test_weights_nowhere_near_one_are_rejected():
    """Far from 1.0 means the model misunderstood the scale, not fumbled rounding.
    Silently rescaling that would hide a real failure."""
    with pytest.raises(ValidationError, match="expected roughly 1.0"):
        GeneratedPlan.model_validate(
            _plan([_criterion("aa", "1.0"), _criterion("bb", "1.0"), _criterion("cc", "1.0")])
        )


def test_a_zero_weight_criterion_is_rejected():
    with pytest.raises(ValidationError):
        GeneratedPlan.model_validate(
            _plan([_criterion("aa", "0.5"), _criterion("bb", "0.5"), _criterion("cc", "0")])
        )


# --- Structural rules -------------------------------------------------------


def test_duplicate_criterion_names_are_rejected():
    """Names are the join key between a question and its criterion."""
    with pytest.raises(ValidationError, match="unique"):
        GeneratedPlan.model_validate(
            _plan([_criterion("depth", "0.4"), _criterion("depth", "0.3"), _criterion("xx", "0.3")])
        )


def test_a_question_naming_a_nonexistent_criterion_is_rejected():
    """It would produce evidence nobody scores."""
    with pytest.raises(ValidationError, match="do not exist"):
        GeneratedPlan.model_validate(
            _plan(THREE_EVEN, [{"body": "A question about things.", "competency": "invented"}])
        )


def test_missing_descriptor_bands_are_rejected():
    """Without them the scorer interpolates, which is the unfalsifiable scoring
    the rubric exists to prevent."""
    bad = _criterion("depth", "0.4")
    bad["descriptors"] = {"1": "weak", "5": "strong"}  # no "3"
    with pytest.raises(ValidationError, match="missing bands"):
        GeneratedPlan.model_validate(_plan([bad, _criterion("bb", "0.3"), _criterion("cc", "0.3")]))


def test_blank_descriptors_do_not_count_as_present():
    bad = _criterion("depth", "0.4")
    bad["descriptors"] = {"1": "weak", "3": "   ", "5": "strong"}
    with pytest.raises(ValidationError, match="missing bands"):
        GeneratedPlan.model_validate(_plan([bad, _criterion("bb", "0.3"), _criterion("cc", "0.3")]))


def test_too_few_criteria_are_rejected():
    with pytest.raises(ValidationError):
        GeneratedPlan.model_validate(_plan([_criterion("only", "1.0")]))


def test_uncovered_criteria_are_reported_but_do_not_fail_the_plan():
    """A conversation may still surface the evidence; rejecting would trade a
    usable plan for no plan."""
    plan = GeneratedPlan.model_validate(_plan(THREE_EVEN))
    assert set(plan.uncovered_criteria) == {"breadth", "comms"}


def test_a_question_with_no_competency_is_allowed():
    plan = GeneratedPlan.model_validate(
        _plan(THREE_EVEN, [{"body": "A general opening question here."}])
    )
    assert plan.questions[0].competency is None


# --- JSON extraction --------------------------------------------------------


class Tiny(BaseModel):
    value: int


@pytest.mark.parametrize(
    "raw",
    [
        '{"value": 7}',
        '```json\n{"value": 7}\n```',
        '```\n{"value": 7}\n```',
        'Here is the result: {"value": 7}. Let me know if you need changes!',
        '  \n {"value": 7}\n ',
    ],
)
def test_json_is_recovered_from_the_shapes_models_actually_emit(raw):
    assert Tiny.model_validate(extract_json(raw)).value == 7


def test_extraction_fails_loudly_when_there_is_no_json():
    with pytest.raises(ValueError, match="no JSON value found"):
        extract_json("I'm afraid I can't help with that.")


# --- Prompt templates -------------------------------------------------------


def test_the_plan_prompt_renders_with_every_placeholder_filled():
    messages = prompts.render(
        "plan_generator",
        job_title="Staff Engineer",
        job_description="Own the ledger.",
        resume_context="[experience] Kafka migration",
        question_count=6,
        duration_minutes=30,
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    # The JSON shape example survives .format() -- its braces are doubled.
    assert '"questions"' in messages[1]["content"]
    assert "{job_title}" not in messages[1]["content"]


def test_the_prompt_labels_the_resume_as_untrusted():
    """A resume is written by the person being evaluated. The model is told."""
    messages = prompts.render(
        "plan_generator",
        job_title="x",
        job_description="y",
        resume_context="z",
        question_count=3,
        duration_minutes=10,
    )
    assert "untrusted" in messages[0]["content"].lower()
    assert "untrusted" in messages[1]["content"].lower()


def test_a_missing_placeholder_raises_rather_than_sending_a_hole():
    with pytest.raises(KeyError):
        prompts.render("plan_generator", job_title="only this one")


def test_a_within_tolerance_sum_is_still_made_exact():
    """0.9999 is "close enough" to accept but not to store.

    The scorer's weighted mean assumes the weights sum to exactly 1.0, so the
    normaliser must not early-return on a near miss.
    """
    plan = GeneratedPlan.model_validate(
        _plan([_criterion("aa", "0.4"), _criterion("bb", "0.3"), _criterion("cc", "0.2999")])
    )
    assert sum(c.weight for c in plan.criteria) == Decimal("1")


def test_exact_weights_are_not_needlessly_rewritten():
    """Rescaling 0.4/0.3/0.3 through a division would make them 0.4000/... for
    no benefit, so the readable numbers survive."""
    plan = GeneratedPlan.model_validate(_plan(THREE_EVEN))
    assert [str(c.weight) for c in plan.criteria] == ["0.4", "0.3", "0.3"]
