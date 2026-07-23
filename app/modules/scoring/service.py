"""Reading and writing the score row. No judgement lives here.

The scoring job is at-least-once, like every other task in this codebase, so
storing has to be an idempotent replace rather than an append: a redelivery must
overwrite the previous answer, not sit beside it. The unique constraint on
``interview_id`` makes that a database guarantee rather than a convention, and
criterion rows are deleted and rewritten as a set so a rubric that lost a
criterion between runs does not leave an orphan behind.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import NotFoundError
from app.models.question_plan import RubricCriterion
from app.models.score import CriterionScore, Score, ScoringStatus
from app.modules.scoring import aggregator
from app.modules.scoring.rubric_scorer import Graded

log = structlog.get_logger(__name__)


async def get_for_interview(
    session: AsyncSession, interview_id: uuid.UUID
) -> Score | None:
    return (
        await session.execute(
            select(Score)
            .where(Score.interview_id == interview_id)
            .options(selectinload(Score.criteria))
        )
    ).scalar_one_or_none()


async def require_for_interview(session: AsyncSession, interview_id: uuid.UUID) -> Score:
    score = await get_for_interview(session, interview_id)
    if score is None:
        raise NotFoundError(
            "This interview has not been scored.", interview_id=str(interview_id)
        )
    return score


async def ensure_score(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    plan_id: uuid.UUID | None = None,
) -> Score:
    """Get or create the interview's score row, in PENDING."""
    score = await get_for_interview(session, interview_id)
    if score is None:
        score = Score(org_id=org_id, interview_id=interview_id, plan_id=plan_id)
        session.add(score)
        await session.flush()
    elif plan_id is not None:
        score.plan_id = plan_id
    return score


async def mark_scoring(session: AsyncSession, score: Score) -> None:
    score.status = ScoringStatus.SCORING
    score.error = None
    await session.flush()


async def mark_failed(session: AsyncSession, score: Score, error: str) -> None:
    """Record the failure without discarding whatever was already stored.

    The previous overall is left in place on purpose: a retry that fails should
    not turn a report a recruiter has already read into a blank one.
    """
    score.status = ScoringStatus.FAILED
    score.error = error[:2000]
    await session.flush()


async def store(
    session: AsyncSession,
    *,
    score: Score,
    results: list[tuple[RubricCriterion, Graded]],
    outcome: aggregator.Outcome,
    signals: dict,
    model_name: str,
) -> Score:
    """Replace the score's contents with this run's results."""
    await session.execute(delete(CriterionScore).where(CriterionScore.score_id == score.id))
    # The identity map still holds the rows the bulk DELETE removed; without
    # this the collection below is appended to a stale list and the flush
    # re-INSERTs the deleted ordinals.
    await session.refresh(score, ["criteria"])

    for ordinal, (criterion, graded) in enumerate(results):
        session.add(
            CriterionScore(
                org_id=score.org_id,
                score_id=score.id,
                criterion_id=criterion.id,
                ordinal=ordinal,
                name=criterion.name,
                weight=criterion.weight,
                score=graded.score,
                rationale=graded.rationale,
                evidence=graded.evidence,
            )
        )

    score.overall = outcome.overall
    score.recommendation = outcome.recommendation
    score.status = ScoringStatus.READY
    score.scored_by = model_name
    score.scored_at = datetime.now(UTC)
    score.error = None
    # Coverage rides with the signals rather than in its own column: it is
    # metadata about the run, and a recruiter reads it in the same block as the
    # rest of the caveats.
    score.confidence_signals = {
        **signals,
        "rubric_coverage": float(outcome.coverage),
        "criteria_graded": outcome.graded_count,
        "criteria_total": outcome.total_count,
    }
    await session.flush()

    log.info(
        "scoring.stored",
        interview_id=str(score.interview_id),
        overall=str(outcome.overall),
        recommendation=outcome.recommendation.value,
        coverage=str(outcome.coverage),
    )
    return score


def weights_and_scores(
    results: list[tuple[RubricCriterion, Graded]],
) -> list[tuple[Decimal, Decimal | None]]:
    """Reduce ORM rows to the pairs the aggregator's arithmetic needs."""
    return [(criterion.weight, graded.score) for criterion, graded in results]
