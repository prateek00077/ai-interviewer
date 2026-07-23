"""The score of an interview. Recruiter-only, at every layer.

There is no candidate route here at all, and RLS makes that structural rather
than a matter of routing: ``scores`` and ``criterion_scores`` are registered
USER_ONLY, so a candidate token reaching one of these queries returns zero rows
even if a future endpoint forgets the role check.

Scoring is asynchronous, so the score row exists in PENDING before there is
anything in it. That is deliberate -- a recruiter who opens the page seconds
after the call ends should see "in progress", not a 404 that reads like the
interview was never scored at all.
"""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, status

from app.api.deps import Principal, ScopedSession, require_role
from app.core.exceptions import ConflictError
from app.models.user import UserRole
from app.modules.interview import service as interview_service
from app.modules.interview import state_machine
from app.modules.scoring import service as scoring_service
from app.schemas.score import ScoreRead

log = structlog.get_logger(__name__)

router = APIRouter(tags=["scoring"])

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]


@router.get(
    "/interviews/{interview_id}/score",
    response_model=ScoreRead,
    summary="The interview's score, its criteria and their evidence",
)
async def get_score(interview_id: uuid.UUID, db: ScopedSession, _: Recruiter) -> ScoreRead:
    # Resolved through the service so a cross-org id is a 404 from the same
    # place every other interview route produces one.
    await interview_service.get_interview(db, interview_id)
    score = await scoring_service.require_for_interview(db, interview_id)
    return ScoreRead.model_validate(score)


@router.post(
    "/interviews/{interview_id}/score/rescore",
    response_model=ScoreRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Re-run scoring for an interview",
)
async def rescore(
    interview_id: uuid.UUID, db: ScopedSession, principal: Recruiter
) -> ScoreRead:
    """Queue a fresh scoring run against the transcript as it stands now.

    Useful after a failed run, or after the transcript pass has corrected the
    text. The previous score stays readable until the new one replaces it: a
    recruiter mid-review should not watch the report empty out.

    Refused while the interview is live. Scoring a conversation that is still
    happening produces a number about half an answer.
    """
    interview = await interview_service.get_interview(db, interview_id)
    if not state_machine.is_terminal(interview.status):
        raise ConflictError(
            "This interview has not finished yet.", current_status=interview.status.value
        )

    score = await scoring_service.ensure_score(
        db, org_id=principal.org_id, interview_id=interview_id
    )
    # Committed before the task is sent, so the worker cannot read a row this
    # request has not written yet.
    await db.commit()

    from app.workers.tasks import scoring_tasks

    scoring_tasks.score_interview.delay(str(principal.org_id), str(interview_id))
    log.info(
        "scoring.rescore_requested",
        interview_id=str(interview_id),
        actor=str(principal.actor_id),
    )
    return ScoreRead.model_validate(score)
