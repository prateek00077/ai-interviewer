"""Schedule, invite, state transitions, transcript access.

Mostly recruiter-facing. The one candidate route is ``/interviews/me``, which
resolves the interview from the token's own claim rather than from a path
parameter -- a candidate cannot ask about an interview that is not theirs
because there is nowhere to put someone else's id.

Note what is absent: no route sets IN_PROGRESS or COMPLETED. Those transitions
belong to the voice session and arrive over the event bus. A recruiter can only
terminate, which is a deliberate act rather than a lifecycle step.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import (
    Principal,
    ScopedSession,
    get_current_candidate,
    require_role,
)
from app.core.exceptions import NotFoundError
from app.models.interview import InterviewStatus
from app.models.user import UserRole
from app.modules.interview import service as interview_service
from app.modules.interview import transcript
from app.modules.users import service as users_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, Page
from app.schemas.interview import (
    CandidateInterviewRead,
    InterviewCreate,
    InterviewRead,
    TerminateRequest,
    TranscriptRead,
    TurnRead,
)

router = APIRouter(prefix="/interviews", tags=["interviews"])

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]
CurrentCandidate = Annotated[Principal, Depends(get_current_candidate)]


# --- Candidate-facing -------------------------------------------------------
#
# Declared before /{interview_id} so "me" is not swallowed by the UUID path
# parameter.


@router.get(
    "/me",
    response_model=CandidateInterviewRead,
    summary="The candidate's own interview",
)
async def get_own_interview(
    db: ScopedSession, principal: CurrentCandidate
) -> CandidateInterviewRead:
    if principal.interview_id is None:
        raise NotFoundError("This token is not tied to an interview.")
    interview = await interview_service.get_interview(db, principal.interview_id)
    return CandidateInterviewRead.model_validate(interview)


# --- Recruiter-facing -------------------------------------------------------


@router.get("", response_model=Page[InterviewRead], summary="List interviews")
async def list_interviews(
    db: ScopedSession,
    _: Recruiter,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[InterviewStatus | None, Query(alias="status")] = None,
    candidate_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
) -> Page[InterviewRead]:
    items, total = await interview_service.list_interviews(
        db,
        limit=limit,
        offset=offset,
        status=status_filter,
        candidate_id=candidate_id,
        job_id=job_id,
    )
    return Page[InterviewRead](
        items=[InterviewRead.model_validate(i) for i in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    response_model=InterviewRead,
    status_code=status.HTTP_201_CREATED,
    summary="Schedule an interview",
)
async def create_interview(
    payload: InterviewCreate, db: ScopedSession, principal: Recruiter
) -> InterviewRead:
    # Resolved rather than trusted: a candidate id from another org would
    # otherwise create an interview pointing at a row this tenant cannot see.
    await users_service.get_candidate(db, payload.candidate_id)
    interview = await interview_service.create_interview(
        db,
        org_id=principal.org_id,
        candidate_id=payload.candidate_id,
        job_id=payload.job_id,
        scheduled_at=payload.scheduled_at,
    )
    return InterviewRead.model_validate(interview)


@router.get("/{interview_id}", response_model=InterviewRead, summary="Get one interview")
async def get_interview(
    interview_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> InterviewRead:
    return InterviewRead.model_validate(
        await interview_service.get_interview(db, interview_id)
    )


@router.get(
    "/{interview_id}/transcript",
    response_model=TranscriptRead,
    summary="The interview transcript",
)
async def get_transcript(
    interview_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> TranscriptRead:
    interview = await interview_service.get_interview(db, interview_id)
    turns = await transcript.list_turns(db, interview_id)
    return TranscriptRead(
        interview_id=interview.id,
        status=interview.status,
        turns=[TurnRead.model_validate(t) for t in turns],
    )


@router.post(
    "/{interview_id}/terminate",
    response_model=InterviewRead,
    summary="Stop an interview",
)
async def terminate_interview(
    interview_id: uuid.UUID,
    payload: TerminateRequest,
    db: ScopedSession,
    _: Recruiter,
) -> InterviewRead:
    # Terminal states are absorbing, so terminating an already-finished
    # interview is a 409 rather than a silent no-op -- the recruiter should know
    # their action did nothing.
    interview = await interview_service.terminate(db, interview_id, reason=payload.reason)
    return InterviewRead.model_validate(interview)
