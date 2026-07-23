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

import re
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator

from app.integrations import nim_client
from app.models.question_plan import WEIGHT_SCALE
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
    # The words from the resume this question is about, copied rather than
    # paraphrased. Not persisted -- it exists so grounding can be CHECKED
    # instead of asked for politely. A model that has to quote the line it is
    # asking about cannot invent the line without the quote failing to match.
    resume_evidence: str = Field(default="", max_length=1000)
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

    @model_validator(mode="before")
    @classmethod
    def _trim_overlong_rubric(cls, data: object) -> object:
        """Keep the heaviest MAX_CRITERIA and drop the rest, rather than reject.

        MEASURED: asked for 3 to 6 criteria, Nemotron-3-Nano returns 8, and
        shown ``List should have at most 6 items`` it returns 8 again. Third
        instance of the same pattern -- the model is good at naming dimensions
        and bad at arithmetic and counting.

        Dropped by ascending weight, so what goes is what the model itself rated
        least important. Weight normalisation runs afterwards and rescales the
        survivors back to 1.0, so the rubric stays coherent.

        A ``mode="before"`` validator because ``max_length`` on the field is
        checked before any ``mode="after"`` hook could intervene.

        The cap is a product constraint, not a technical one: more than six
        dimensions cannot be scored reliably in one conversation, and each one
        costs another model call at scoring time. Trimming honours the cap;
        raising it would quietly abandon it.
        """
        if not isinstance(data, dict):
            return data
        criteria = data.get("criteria")
        if not isinstance(criteria, list) or len(criteria) <= MAX_CRITERIA:
            return data

        def _weight(item: object) -> Decimal:
            try:
                return Decimal(str(item.get("weight", 0)))  # type: ignore[attr-defined]
            except (AttributeError, ArithmeticError, TypeError, ValueError):
                return Decimal(0)

        kept = sorted(criteria, key=_weight, reverse=True)[:MAX_CRITERIA]
        dropped = [c.get("name") for c in criteria if c not in kept]
        log.warning(
            "plan_rubric_trimmed",
            returned=len(criteria),
            kept=MAX_CRITERIA,
            dropped=dropped,
        )
        # Original ordering preserved among survivors: the model puts the
        # criteria in a deliberate order and reordering them by weight would
        # reshuffle the rubric a recruiter is about to read.
        return {**data, "criteria": [c for c in criteria if c in kept]}

    @model_validator(mode="after")
    def _check_rubric(self) -> GeneratedPlan:
        names = [c.name.strip() for c in self.criteria]
        if len(set(names)) != len(names):
            raise ValueError("criterion names must be unique; they are the join key for questions")

        self._normalise_weights()

        self._drop_unmatched_competencies(set(names))
        return self

    def _drop_unmatched_competencies(self, names: set[str]) -> None:
        """Clear competency tags that name no criterion, rather than rejecting.

        MEASURED BEHAVIOUR, same shape as the weights problem below: Nemotron
        invents question-level tags that do not appear in the rubric it just
        wrote -- ``testing_async_sqlalchemy`` alongside criteria called
        ``async_processing`` and ``schema_design`` -- and shown the exact
        validation error, it does it again. Two calls, one plan thrown away.

        Rejecting was also inconsistent with what the rest of the system already
        does. ``PlannedQuestion.competency`` is documented as a soft join key
        precisely because the LLM emits both halves in one pass, and the scorer
        already treats an unmatched competency as ungraded. The validator was
        the only component that treated a tagging slip as fatal.

        So an unmatched tag is dropped and the question is kept. The question is
        the valuable part -- it still gets asked, still produces evidence, and
        the criterion it half-belonged to is scored from the transcript as a
        whole. What is lost is a hint, and the loss is logged.
        """
        for question in self.questions:
            if question.competency and question.competency.strip() not in names:
                log.warning(
                    "plan_competency_unmatched",
                    competency=question.competency,
                    valid=sorted(names),
                )
                question.competency = None

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

        # QUANTIZE UNCONDITIONALLY, at the precision the column actually stores.
        #
        # This used to run only when the total drifted past WEIGHT_TOLERANCE, on
        # the reasoning that rescaling 0.4/0.3/0.3 through a division buys
        # nothing. That reasoning was wrong, and the bug it caused survived
        # every unit test:
        #
        # Nemotron returns six criteria at 0.16666666666666666. In Python those
        # sum to 0.99999999999999996 -- within tolerance, so the old code left
        # them alone and the assertion below passed at full Decimal precision.
        # But ``RubricCriterion.weight`` is Numeric(5,4), so Postgres rounds
        # each one to 0.1667 on INSERT, and six of those sum to 1.0002.
        #
        # Quantization does not distribute over addition. Enforcing the
        # invariant at a precision finer than the one you store it at enforces
        # nothing. OBSERVED end to end: every rubric with a weight that is not
        # exactly representable in four places reached the database summing to
        # something other than 1.0, and the scorer's weighted mean was computed
        # against it.
        quantum = Decimal(1).scaleb(-WEIGHT_SCALE)
        for criterion in self.criteria:
            criterion.weight = (criterion.weight / total).quantize(quantum)

        # Then make the sum EXACT. Quantizing six sixths leaves a rounding
        # remainder no matter how it is done, and the scorer's weighted mean and
        # every assertion downstream assume exactness -- so the remainder is put
        # somewhere deliberate rather than left to land wherever.
        drift = Decimal(1) - sum((c.weight for c in self.criteria), Decimal(0))
        if drift:
            # Onto the heaviest criterion, where a ten-thousandth is the
            # smallest relative distortion available.
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


# A question is grounded when this fraction of the distinctive words in its
# quoted evidence actually appear in the resume. Not 1.0: the model
# re-capitalises, drops a bullet's punctuation and occasionally joins two
# fragments, none of which mean it invented the experience. Well below 1.0 and
# a question about Kubernetes "grounded" in the word "the" would pass.
GROUNDING_THRESHOLD = 0.6
# Below this, dropping ungrounded questions would leave an interview too short
# to score anything, so they are kept and the failure is logged instead.
MIN_GROUNDED_QUESTIONS = 3
# Words that carry no evidence of anything. Overlap on these is noise.
_STOPWORDS = frozenset(
    "a an and the of to in on for with at by from as is was were be been using "
    "used built build developed development work worked experience project "
    "projects it its this that their they i my our we".split()
)
_WORD = re.compile(r"[a-z0-9+#.]+")


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def _ungrounded(plan: GeneratedPlan, resume_context: str) -> list[GeneratedQuestion]:
    """Questions whose quoted evidence is not in the resume.

    THE CHECK IS AGAINST THE RESUME TEXT, not against the evidence field being
    non-empty. A model asked to cite its source will happily cite one it made
    up, and an unverified citation is worth less than none -- it makes an
    invented question look checked.

    Word overlap rather than substring: the model reformats what it copies
    (case, punctuation, a bullet split across two lines) far more often than it
    fabricates, and a substring test fails all of that as if it were invention.
    """
    haystack = set(_words(resume_context))
    if not haystack:
        return []

    offenders = []
    for question in plan.questions:
        needle = _words(question.resume_evidence)
        # No evidence at all is ungrounded by definition: nothing was claimed,
        # so nothing was checked.
        if not needle:
            offenders.append(question)
            continue
        hits = sum(1 for w in needle if w in haystack)
        if hits / len(needle) < GROUNDING_THRESHOLD:
            offenders.append(question)
    return offenders


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

    # One grounding repair, and only when there is a resume to be grounded in.
    # The schema was satisfied -- this is the check the schema cannot make --
    # so it needs its own turn rather than riding on complete_structured's.
    if resume_context.strip():
        offenders = _ungrounded(plan, resume_context)
        if offenders:
            log.warning(
                "plan_questions_ungrounded",
                count=len(offenders),
                total=len(plan.questions),
                bodies=[q.body[:120] for q in offenders],
            )
            repaired = await nim_client.complete_structured(
                [
                    *messages,
                    {"role": "assistant", "content": plan.model_dump_json()},
                    {
                        "role": "user",
                        "content": (
                            "These questions are about experience that does not "
                            "appear in THE CANDIDATE section:\n"
                            + "\n".join(f"- {q.body}" for q in offenders)
                            + "\n\nRewrite EVERY one of them to be about work the "
                            "candidate actually describes, and quote the exact "
                            "words from THE CANDIDATE in resume_evidence. Keep "
                            "the questions that were already grounded as they "
                            "are. Return the complete JSON object again."
                        ),
                    },
                ],
                GeneratedPlan,
            )
            still = _ungrounded(repaired, resume_context)
            log.info("plan_grounding_repaired", before=len(offenders), after=len(still))
            plan = repaired

            # WHAT THE REPAIR COULD NOT GROUND IS DROPPED, not asked.
            #
            # MEASURED on a real CV: six questions, one ungrounded ("how would
            # you approach a payment reconciliation service using Celery and S3"
            # of a candidate with neither on their resume), and the repair turn
            # returned it unchanged. Same shape as the weights and the
            # competency tags -- shown its own error, the model repeats itself.
            #
            # Asking it anyway costs three minutes of a fixed-length interview
            # and produces "I have not used that" as the evidence a criterion
            # gets scored on. A shorter interview is strictly better.
            #
            # Down to MIN_GROUNDED_QUESTIONS and no further: if almost nothing
            # survives, the resume text is probably the problem (a scanned PDF
            # that parsed to noise), and an odd interview still beats none.
            if still and len(plan.questions) - len(still) >= MIN_GROUNDED_QUESTIONS:
                dropped = {id(q) for q in still}
                plan.questions = [q for q in plan.questions if id(q) not in dropped]
                log.warning(
                    "plan_questions_dropped_ungrounded",
                    dropped=len(still),
                    remaining=len(plan.questions),
                )
            elif still:
                log.warning(
                    "plan_grounding_gave_up",
                    ungrounded=len(still),
                    total=len(plan.questions),
                )

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
