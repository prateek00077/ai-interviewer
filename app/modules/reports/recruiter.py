"""Recruiter report: scores, evidence, timestamps, proctoring, recommendation.

The mirror image of ``candidate.py`` -- this one is handed everything, and the
two share no serializer, no template and no code path. That duplication is the
design: a shared builder with an audience flag is one wrong boolean away from
sending a hire recommendation to the person it is about.

This module assembles; it does not judge. Every number here was decided in
``modules/scoring``; every proctoring finding in ``modules/proctoring``. What
this adds is the framing a reviewer needs to disagree with them:

- Each criterion travels with the evidence that produced it, and each quote
  with the offset in the recording where it was said. A score a recruiter
  cannot check is a score they have to take on faith.
- Rubric coverage is shown next to the overall, because "3.6 across the whole
  rubric" and "3.6 across the 60% we got to" are different findings.
- Delivery signals are in their own block, labelled as observations, with the
  reason they are not scored stated on the page. A recruiter reading "long
  pauses" needs to know nobody marked the candidate down for it.
- A proctoring verdict never appears without its reasons.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from decimal import Decimal

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.interview import Interview
from app.models.proctoring import ProctoringVerdict
from app.models.score import Score
from app.modules.interview import transcript as transcript_module
from app.modules.jobs import service as jobs_service
from app.modules.scoring import service as scoring_service
from app.modules.users import service as users_service

log = structlog.get_logger(__name__)

# Signals that are measurements of delivery rather than metadata about the run.
# The rest of the JSONB blob (rubric_coverage, criteria_graded) is shown in the
# scoring block instead, where it belongs.
DELIVERY_SIGNAL_KEYS = (
    "words",
    "words_per_minute",
    "speaking_seconds",
    "median_pitch_hz",
    "pitch_variation",
    "pause_count",
    "median_pause_ms",
    "longest_pause_ms",
    "filler_count",
    "fillers_per_100_words",
)


@dataclass(slots=True)
class CriterionView:
    name: str
    weight: Decimal
    score: Decimal | None
    rationale: str | None
    evidence: list[dict] = field(default_factory=list)

    @property
    def is_graded(self) -> bool:
        return self.score is not None


@dataclass(slots=True)
class RecruiterView:
    candidate_name: str
    candidate_email: str
    job_title: str
    interview_id: uuid.UUID
    status: str
    started_at: object = None
    completed_at: object = None

    overall: Decimal | None = None
    recommendation: str | None = None
    rubric_coverage: float | None = None
    criteria_graded: int = 0
    criteria_total: int = 0
    scored_by: str | None = None
    criteria: list[CriterionView] = field(default_factory=list)

    proctoring_verdict: str | None = None
    proctoring_reasons: list[str] = field(default_factory=list)
    frames_analysed: int = 0

    # None means the interview was conducted against a plan no human reviewed.
    # Printed on the report, because a score is only as defensible as the rubric
    # behind it and a recruiter should not have to go looking.
    plan_approved_at: object = None

    delivery_signals: dict = field(default_factory=dict)
    turns: list = field(default_factory=list)
    has_recording: bool = False

    @property
    def is_assessed(self) -> bool:
        return self.overall is not None

    def as_dict(self) -> dict:
        return asdict(self)


async def build(session: AsyncSession, interview_id: uuid.UUID) -> RecruiterView:
    """Gather everything about one interview into a single render model.

    Every part is optional. An interview whose plan generation failed has no
    score; one that ran without a camera has no proctoring verdict. The report
    still renders and says so, because a recruiter looking at a blank page
    cannot tell "not assessed" from "the page is broken".
    """
    interview = (
        await session.execute(select(Interview).where(Interview.id == interview_id))
    ).scalar_one()

    candidate = await users_service.get_candidate(session, interview.candidate_id)

    job_title = "(no role attached)"
    if interview.job_id is not None:
        # NotFoundError only. A deleted job must not fail the report, but a
        # broader except would swallow a MissingGreenlet and make it look like
        # the job was simply missing.
        try:
            job_title = (await jobs_service.get_job(session, interview.job_id)).title
        except NotFoundError:
            log.warning("reports.job_missing", interview_id=str(interview_id))

    view = RecruiterView(
        candidate_name=candidate.full_name or candidate.email,
        candidate_email=candidate.email,
        job_title=job_title,
        interview_id=interview.id,
        status=interview.status.value,
        started_at=interview.started_at,
        completed_at=interview.completed_at,
        has_recording=interview.recording_key is not None,
    )

    from app.modules.question_plan import service as plan_service

    plan = await plan_service.get_for_interview(session, interview_id)
    if plan is not None:
        view.plan_approved_at = plan.approved_at

    score = await scoring_service.get_for_interview(session, interview_id)
    if score is not None:
        _apply_score(view, score)

    verdict = (
        await session.execute(
            select(ProctoringVerdict).where(ProctoringVerdict.interview_id == interview_id)
        )
    ).scalar_one_or_none()
    if verdict is not None:
        view.proctoring_verdict = verdict.verdict.value
        # A verdict never travels without its reasons. "FLAGGED" alone is an
        # accusation with no way to check it.
        view.proctoring_reasons = list(verdict.reasons)
        view.frames_analysed = verdict.frames_analysed

    view.turns = await transcript_module.list_turns(session, interview_id)
    return view


def _apply_score(view: RecruiterView, score: Score) -> None:
    signals = score.confidence_signals or {}
    view.overall = score.overall
    view.recommendation = score.recommendation.value if score.recommendation else None
    view.scored_by = score.scored_by
    view.rubric_coverage = signals.get("rubric_coverage")
    view.criteria_graded = signals.get("criteria_graded", 0)
    view.criteria_total = signals.get("criteria_total", 0)
    view.delivery_signals = {
        key: signals[key] for key in DELIVERY_SIGNAL_KEYS if signals.get(key) is not None
    }
    view.criteria = [
        CriterionView(
            name=criterion.name,
            weight=criterion.weight,
            score=criterion.score,
            rationale=criterion.rationale,
            evidence=list(criterion.evidence or []),
        )
        for criterion in score.criteria
    ]


def topic_names(view: RecruiterView) -> list[str]:
    """The criterion names, and nothing else about them.

    This is the ONLY thing the candidate report is allowed to take from here --
    the names of what the interview set out to cover. Weights, scores,
    rationales, evidence and descriptors all stay on this side.
    """
    return [criterion.name for criterion in view.criteria]
