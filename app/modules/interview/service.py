"""Interview orchestration and lifecycle operations.

Every status change goes through ``state_machine.transition``; nothing here
assigns ``interview.status`` directly. The status governs who may connect,
whether the plan may still be edited, and whether scoring should run, so one
place has to own it.

The module also subscribes to the session events the voice pipeline publishes.
That direction is deliberate: ``voice/`` announces what happened
("the session ended, reason=abandoned") and this module decides what it means
for the interview's state. The voice module never sets a status itself.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import Select, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.events import SessionEnded, SessionStarted, subscribe
from app.core.exceptions import ConflictError, NotFoundError
from app.db.session import tenant_session
from app.models.interview import Interview, InterviewStatus
from app.models.question_plan import PlanStatus
from app.modules.interview import state_machine
from app.modules.question_plan import service as plan_service

log = structlog.get_logger(__name__)

# How a voice session's reported reason maps onto a status. Unknown reasons are
# ABANDONED, which is the honest default: something ended the call and it was
# not a clean completion.
END_REASON_STATUS: dict[str, InterviewStatus] = {
    "completed": InterviewStatus.COMPLETED,
    "abandoned": InterviewStatus.ABANDONED,
    "terminated": InterviewStatus.TERMINATED,
    "timed_out": InterviewStatus.COMPLETED,
}


# --- Reads ------------------------------------------------------------------


async def get_interview(session: AsyncSession, interview_id: uuid.UUID) -> Interview:
    """One interview, or 404.

    The session is org-scoped, so another tenant's interview is simply not
    found -- the caller is never told it exists.
    """
    interview = await session.get(Interview, interview_id)
    if interview is None:
        raise NotFoundError("Interview not found.", interview_id=str(interview_id))
    return interview


async def list_interviews(
    session: AsyncSession,
    *,
    limit: int,
    offset: int,
    status: InterviewStatus | None = None,
    candidate_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
) -> tuple[list[Interview], int]:
    stmt: Select = select(Interview)
    if status is not None:
        stmt = stmt.where(Interview.status == status)
    if candidate_id is not None:
        stmt = stmt.where(Interview.candidate_id == candidate_id)
    if job_id is not None:
        stmt = stmt.where(Interview.job_id == job_id)

    total = await session.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = (
        await session.execute(
            stmt.order_by(Interview.created_at.desc(), Interview.id).limit(limit).offset(offset)
        )
    ).scalars()
    return list(rows), int(total or 0)


# --- Creation ---------------------------------------------------------------


async def create_interview(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    candidate_id: uuid.UUID,
    job_id: uuid.UUID | None = None,
    scheduled_at: datetime | None = None,
) -> Interview:
    """An interview with no invite yet.

    Separate from the invite flow because a recruiter schedules a batch first
    and sends links later, and because an interview needs to exist before a
    question plan can be generated for it.
    """
    interview = Interview(
        org_id=org_id,
        candidate_id=candidate_id,
        job_id=job_id,
        scheduled_at=scheduled_at,
        status=InterviewStatus.CREATED,
    )
    session.add(interview)
    await session.flush()
    log.info("interview.created", interview_id=str(interview.id))
    return interview


# --- Transitions ------------------------------------------------------------


async def start(session: AsyncSession, interview_id: uuid.UUID) -> Interview:
    """Mark an interview live and freeze its plan. Idempotent.

    Freezing here rather than at approval is the point: the plan must be
    immutable from the moment the first question is asked, and "approved" is a
    recruiter's opinion that can still be revised right up until then.

    A plan that does not exist is not an error. An interview can be run without
    one -- the interviewer improvises from the job description -- and refusing
    to start would strand a candidate who is already on the call.
    """
    interview = await get_interview(session, interview_id)
    if state_machine.is_terminal(interview.status):
        raise ConflictError(
            "This interview has already ended.", current_status=interview.status.value
        )

    changed = state_machine.transition(interview, InterviewStatus.IN_PROGRESS, reason="session")

    plan = await plan_service.get_for_interview(session, interview_id)
    if plan is not None and plan.status is not PlanStatus.FROZEN and plan.questions:
        await plan_service.freeze(session, plan=plan)

    await session.flush()
    if changed:
        log.info("interview.started", interview_id=str(interview_id))
    return interview


async def finish(
    session: AsyncSession,
    interview_id: uuid.UUID,
    *,
    reason: str = "completed",
) -> Interview:
    """End an interview. Idempotent, and never raises on a terminal interview.

    Called from the session-ended handler, which runs off a fire-and-forget bus:
    raising there would only be logged, and a second close event arriving after
    a clean one must not be an error.
    """
    interview = await get_interview(session, interview_id)
    target = END_REASON_STATUS.get(reason, InterviewStatus.ABANDONED)

    if state_machine.is_terminal(interview.status):
        return interview

    state_machine.transition(interview, target, reason=reason)
    await session.flush()
    return interview


async def terminate(
    session: AsyncSession, interview_id: uuid.UUID, *, reason: str = "recruiter"
) -> Interview:
    """Stop an interview deliberately -- recruiter action or a proctoring breach."""
    interview = await get_interview(session, interview_id)
    if state_machine.is_terminal(interview.status):
        raise ConflictError(
            "This interview has already ended.", current_status=interview.status.value
        )
    state_machine.transition(interview, InterviewStatus.TERMINATED, reason=reason)
    await session.flush()
    log.info("interview.terminated", interview_id=str(interview_id), reason=reason)
    return interview


# --- Expiry -----------------------------------------------------------------


async def expire_stale(session: AsyncSession, *, older_than_hours: int | None = None) -> int:
    """Expire interviews nobody ever joined. Returns how many were changed.

    A bulk UPDATE rather than a loop of transitions: the reaper may touch
    thousands of rows and does not need each one's timestamp logic. The WHERE
    clause encodes the same rule the state machine does -- only CREATED and
    INVITED have an edge to EXPIRED.
    """
    # `is None`, not `or`: 0 is a meaningful value here (expire everything) and
    # falsy, so `or` would silently substitute the 72-hour default.
    hours = settings.interview_expiry_hours if older_than_hours is None else older_than_hours
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    statement = (
        update(Interview)
        .where(
            Interview.status.in_([InterviewStatus.CREATED, InterviewStatus.INVITED]),
            Interview.created_at < cutoff,
        )
        .values(status=InterviewStatus.EXPIRED, completed_at=func.now())
        .returning(Interview.id)
    )
    count = len((await session.execute(statement)).scalars().all())
    if count:
        log.info("interview.expired_batch", count=count, cutoff=cutoff.isoformat())
    return count


# --- Bus subscriptions ------------------------------------------------------


async def _on_session_started(event: SessionStarted) -> None:
    async with tenant_session(event.org_id, "system", None) as session:
        await start(session, event.interview_id)


async def _on_session_ended(event: SessionEnded) -> None:
    async with tenant_session(event.org_id, "system", None) as session:
        await finish(session, event.interview_id, reason=event.reason)


def register() -> None:
    """Wire this module to the bus. Called once from the app lifespan."""
    subscribe(SessionStarted, _on_session_started)
    subscribe(SessionEnded, _on_session_ended)
