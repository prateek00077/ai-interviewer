"""LLM: job description + resume -> question plan + rubric.

Everything the model returns is validated before it is trusted, and the checks
are not cosmetic:

- Weights must end up summing to 1.0. The scorer computes a weighted mean, so a
  rubric summing to 0.8 silently produces scores 20% too low with nothing
  downstream to notice. Ordinary drift is rescaled rather than rejected -- see
  ``_normalise_weights`` for why, and for the measurement behind it.
- Question competencies must name real criteria. A question tagged with a
  criterion that does not exist produces evidence nobody scores.
- Criterion names must be unique. They are the join key between a question and
  its criterion, so duplicates make that join ambiguous.

Validation failures go back to the model once as a repair turn (see
``nim_client.complete_structured``) before the generation is failed.
"""

from __future__ import annotations

from decimal import Decimal

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator

from app.integrations import nim_client
from app.models.question_plan import WEIGHT_SCALE, WEIGHT_TOLERANCE
from app.modules import prompts
from app.modules.voice.nvidia.catalog import get_service

log = structlog.get_logger(__name__)

DEFAULT_QUESTION_COUNT = 8
DEFAULT_DURATION_MINUTES = 30

MIN_CRITERIA = 3
MAX_CRITERIA = 6
# Bands the prompt asks for. A missing band leaves the scorer interpolating,
# which is exactly the unfalsifiable scoring the rubric exists to prevent.
REQUIRED_BANDS = ("1", "3", "5")


class GeneratedQuestion(BaseModel):
    body: str = Field(min_length=10, max_length=2000)
    competency: str | None = Field(default=None, max_length=120)
    follow_up_hints: list[str] = Field(default_factory=list, max_length=6)
    time_budget_secs: int = Field(default=180, ge=30, le=1800)


class GeneratedCriterion(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    # Decimal, not float: these are summed and compared against 1.0, and binary
    # floats make that comparison a coin toss at the last digit.
    weight: Decimal = Field(gt=0, le=1)
    descriptors: dict[str, str] = Field(default_factory=dict)

    @field_validator("descriptors")
    @classmethod
    def _has_required_bands(cls, v: dict[str, str]) -> dict[str, str]:
        missing = [band for band in REQUIRED_BANDS if not v.get(band, "").strip()]
        if missing:
            raise ValueError(f"descriptors missing bands {missing}; required: {REQUIRED_BANDS}")
        return v


class GeneratedPlan(BaseModel):
    questions: list[GeneratedQuestion] = Field(min_length=1, max_length=40)
    criteria: list[GeneratedCriterion] = Field(min_length=MIN_CRITERIA, max_length=MAX_CRITERIA)

    @model_validator(mode="after")
    def _check_rubric(self) -> GeneratedPlan:
        names = [c.name.strip() for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError("criterion names must be unique; they are the join key for questions")

        self._normalise_weights()

        unknown = {
            q.competency
            for q in self.questions
            if q.competency and q.competency.strip() not in set(names)
        }
        if unknown:
            raise ValueError(
                f"questions reference criteria that do not exist: {sorted(unknown)}. "
                f"Valid names: {names}"
            )
        return self

    def _normalise_weights(self) -> None:
        """Rescale weights to sum to exactly 1.0.

        MEASURED BEHAVIOUR: Nemotron-3-Nano returns weights summing to 1.05 and,
        shown that exact validation error, returns the same numbers again. The
        repair turn works mechanically; the model simply cannot do the
        arithmetic reliably.

        Rejecting on that would throw away a perfectly good rubric over a
        constant we can compute ourselves. What the model actually contributes
        is the *relative* importance of the criteria, and proportional rescaling
        preserves that exactly.

        The bound below is the part still worth rejecting: a total far from 1.0
        means the model misunderstood the scale rather than fumbled the rounding,
        and silently rescaling that would hide a real failure.
        """
        total = sum((c.weight for c in self.criteria), Decimal(0))
        if not (Decimal("0.5") <= total <= Decimal("1.5")):
            raise ValueError(
                f"criterion weights sum to {total}; expected roughly 1.0. "
                "Use decimal fractions, not percentages or arbitrary points."
            )

        # Rescale only when the drift is real. Within tolerance the model's own
        # numbers are kept, because rescaling 0.4/0.3/0.3 through a division
        # would turn readable weights into 0.4000/0.3000/0.3000 for nothing.
        if abs(total - Decimal(1)) > WEIGHT_TOLERANCE:
            quantum = Decimal(1).scaleb(-WEIGHT_SCALE)
            for criterion in self.criteria:
                criterion.weight = (criterion.weight / total).quantize(quantum)

        # Then make the sum EXACT, in both branches. Rescaling leaves a few
        # ten-thousandths of rounding behind, and a within-tolerance total like
        # 0.9999 is still not 1.0 -- the scorer's weighted mean and every
        # assertion downstream assume exactness, so "close enough" cannot be
        # allowed to reach the database.
        drift = Decimal(1) - sum((c.weight for c in self.criteria), Decimal(0))
        if drift:
            max(self.criteria, key=lambda c: c.weight).weight += drift

    @property
    def uncovered_criteria(self) -> list[str]:
        """Criteria no question produces evidence for.

        A warning rather than a rejection: the live interview is a conversation
        and may still surface the evidence. Rejecting the whole generation over
        it would trade a usable plan for no plan.
        """
        covered = {q.competency.strip() for q in self.questions if q.competency}
        return [c.name for c in self.criteria if c.name.strip() not in covered]


async def generate(
    *,
    job_title: str,
    job_description: str,
    resume_context: str = "",
    question_count: int = DEFAULT_QUESTION_COUNT,
    duration_minutes: int = DEFAULT_DURATION_MINUTES,
) -> tuple[GeneratedPlan, str]:
    """Produce a validated plan. Returns it alongside the model that made it."""
    messages = prompts.render(
        "plan_generator",
        job_title=job_title,
        job_description=job_description,
        # An empty resume is normal -- a candidate may never upload one -- and
        # the model is told that rather than being handed a blank block.
        resume_context=resume_context.strip() or "(no resume was provided)",
        question_count=question_count,
        duration_minutes=duration_minutes,
    )

    plan = await nim_client.complete_structured(messages, GeneratedPlan)
    model_name = get_service("llm").model

    uncovered = plan.uncovered_criteria
    if uncovered:
        log.warning("plan_criteria_uncovered", criteria=uncovered)

    log.info(
        "plan_generated",
        questions=len(plan.questions),
        criteria=len(plan.criteria),
        model=model_name,
    )
    return plan, model_name
