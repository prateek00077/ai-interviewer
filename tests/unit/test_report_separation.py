"""The candidate must never learn their score. Enforced at five layers.

This is the highest-consequence property in the reporting slice, so it is
tested structurally rather than by example. An example test ("this particular
PDF contains no digits") passes right up until someone adds a field; these
assert that there is no field to add.

The five layers, each tested below:

1. The table has no score column.
2. The view model has no score attribute.
3. The generator's signature accepts no score-bearing type.
4. The template references no score variable.
5. The response schema has no score field.

Any one of these would probably be enough. All five, because the failure is
silent, irreversible, and lands on a person.
"""

import inspect

import pytest

from app.models.report import CandidateReport, RecruiterReport
from app.modules.reports import candidate, recruiter, renderer
from app.schemas.report import CandidateFeedbackRead

# Substrings that must not appear in any candidate-facing name. Kept broad on
# purpose: "band" and "rank" are not currently used anywhere, and that is the
# point -- this list is a tripwire for concepts, not just for today's fields.
FORBIDDEN = (
    "score",
    "overall",
    "recommendation",
    "verdict",
    "weight",
    "rubric",
    "band",
    "rank",
    "percentile",
    "grade",
)


def _offending(names) -> list[str]:
    return [n for n in names if any(word in n.lower() for word in FORBIDDEN)]


# --- 1. The table ------------------------------------------------------------


def test_the_candidate_report_table_has_no_score_column():
    columns = [c.name for c in CandidateReport.__table__.columns]
    assert _offending(columns) == [], f"candidate_reports carries {_offending(columns)}"


def test_the_recruiter_report_table_is_the_one_that_may_reference_scoring():
    """The mirror assertion. If this ever fails, the two tables have been
    merged and every other test in this file is testing nothing."""
    assert CandidateReport.__tablename__ != RecruiterReport.__tablename__


# --- 2. The view model -------------------------------------------------------


def test_the_candidate_view_has_no_score_attribute():
    fields = candidate.CandidateView.__dataclass_fields__
    assert _offending(fields) == [], f"CandidateView carries {_offending(fields)}"


def test_the_candidate_view_cannot_be_given_one():
    """``slots=True`` is load-bearing here, not a micro-optimisation: without
    it, ``view.overall = 4.2`` silently succeeds and the template could print
    it."""
    view = candidate.CandidateView(candidate_name="A", job_title="B", summary="c")
    with pytest.raises(AttributeError):
        view.overall = 4.2  # type: ignore[attr-defined]


# --- 3. The generator's signature -------------------------------------------


def test_the_feedback_generator_accepts_no_score_bearing_argument():
    """The strongest of the five. A value that is never passed in cannot leak
    out, whatever the model decides to write."""
    parameters = inspect.signature(candidate.generate).parameters
    assert _offending(parameters) == [], f"generate() takes {_offending(parameters)}"
    assert set(parameters) == {"job_title", "topic_names", "turns"}


def test_the_candidate_module_never_imports_the_score_models():
    """Checked against the parsed imports, not the raw text.

    The module's own docstring explains at length that it must not touch
    ``Score``, so a substring search over the source finds the prose describing
    the rule and reports it as a violation of it.
    """
    import ast

    tree = ast.parse(inspect.getsource(candidate))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)

    assert not any("score" in name.lower() for name in imported), imported
    assert not any("proctoring" in name.lower() for name in imported), imported


def test_only_criterion_names_cross_from_the_recruiter_side():
    """``topic_names`` is the whole bridge between the two audiences."""
    view = recruiter.RecruiterView(
        candidate_name="A",
        candidate_email="a@e.com",
        job_title="Staff Engineer",
        interview_id=__import__("uuid").uuid4(),
        status="COMPLETED",
        criteria=[
            recruiter.CriterionView(
                name="System Design",
                weight=__import__("decimal").Decimal("0.6"),
                score=__import__("decimal").Decimal("4.00"),
                rationale="strong",
                evidence=[{"quote": "we sharded", "turn_ordinal": 1}],
            )
        ],
    )
    assert recruiter.topic_names(view) == ["System Design"]


# --- 4. The template ---------------------------------------------------------


def test_the_candidate_template_references_no_score_variable():
    source = (renderer.TEMPLATE_DIR / renderer.CANDIDATE_TEMPLATE).read_text()
    # Strip Jinja comments first: the template explains WHY it has no score,
    # and that prose would otherwise trip the check it is describing.
    stripped = __import__("re").sub(r"\{#.*?#\}", "", source, flags=__import__("re").DOTALL)
    offenders = [
        word
        for word in FORBIDDEN
        if f"v.{word}" in stripped or f"{{{{ {word}" in stripped
    ]
    assert offenders == [], f"candidate template reads {offenders}"


def test_the_two_templates_are_different_files():
    assert renderer.CANDIDATE_TEMPLATE != renderer.RECRUITER_TEMPLATE
    for name in (renderer.CANDIDATE_TEMPLATE, renderer.RECRUITER_TEMPLATE):
        assert (renderer.TEMPLATE_DIR / name).exists()


def test_the_candidate_template_includes_no_shared_partial():
    """A shared "score block" the candidate template merely declines to call is
    exactly the arrangement that leaks on the next layout refactor."""
    source = (renderer.TEMPLATE_DIR / renderer.CANDIDATE_TEMPLATE).read_text()
    assert "{% include" not in source
    assert "{% extends" not in source


# --- 5. The response schema --------------------------------------------------


def test_the_candidate_schema_has_no_score_field():
    fields = CandidateFeedbackRead.model_fields
    assert _offending(fields) == [], f"CandidateFeedbackRead carries {_offending(fields)}"


def test_the_candidate_schema_does_not_inherit_from_the_recruiter_one():
    """A shared base is the mechanism by which a field added "for the recruiter
    view" silently appears in the candidate one."""
    from app.schemas.report import RecruiterReportRead

    assert not issubclass(CandidateFeedbackRead, RecruiterReportRead)
    assert not issubclass(RecruiterReportRead, CandidateFeedbackRead)


# --- And the prompt ----------------------------------------------------------


def test_the_feedback_prompt_is_never_given_a_score():
    """Rendering with every real input and asserting the result is clean.

    Belt and braces over the signature test: this catches a future template
    that reaches for a score through some other route.
    """
    from app.modules import prompts

    messages = prompts.render(
        "candidate_feedback",
        job_title="Staff Engineer",
        topics="- System Design\n- Communication",
        transcript="[CANDIDATE] We sharded on tenant id.",
    )
    rendered = " ".join(m["content"] for m in messages).lower()
    # "scored", "scoring" and "score" all appear in the *instructions* telling
    # the model it has not been given one, so the check is on digits-as-marks
    # rather than on the word.
    for forbidden in ("/ 5", "out of 5", "recommendation:", "strong_hire", "no_hire"):
        assert forbidden not in rendered, f"prompt contains {forbidden!r}"
