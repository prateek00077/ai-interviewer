"""Generate, review, and edit question plans + rubrics.

Recruiter-only, all of it. ``require_role`` rejects candidate tokens before any
handler runs, and the RLS policy on these tables rejects them again at the
database -- a candidate reading their own plan would know the questions and the
weights before the interview starts.

Plans are addressed by interview rather than by their own id: there is exactly
one plan per interview, and "the plan for interview X" is how every caller
actually thinks about it.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.api.deps import Principal, ScopedSession, require_role
from app.core.exceptions import NotFoundError
from app.models.user import UserRole
from app.modules.interview import service as interview_service
from app.modules.question_plan import service as plan_service
from app.schemas.question_plan import (
    ApproveRequest,
    CriteriaReplace,
    GeneratePlanRequest,
    QuestionPlanRead,
    QuestionsReplace,
)
from app.workers.tasks.plan_tasks import generate_plan

router = APIRouter(prefix="/interviews/{interview_id}/plan", tags=["question-plans"])

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]


async def _load(db: ScopedSession, interview_id: uuid.UUID):
    plan = await plan_service.get_for_interview(db, interview_id)
    if plan is None:
        raise NotFoundError("No question plan for this interview.", interview_id=str(interview_id))
    return plan


@router.get("", response_model=QuestionPlanRead, summary="The interview's plan and rubric")
async def get_plan(
    interview_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> QuestionPlanRead:
    return QuestionPlanRead.model_validate(await _load(db, interview_id))


@router.post(
    "/generate",
    response_model=QuestionPlanRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue generation (or regeneration) of the plan",
)
async def request_generation(
    interview_id: uuid.UUID,
    payload: GeneratePlanRequest,
    db: ScopedSession,
    principal: Recruiter,
) -> QuestionPlanRead:
    """202, not 201: the model call happens in a worker.

    The response is the PENDING shell, so the client has a row to poll rather
    than a job id to correlate.
    """
    interview = await interview_service.get_interview(db, interview_id)
    plan = await plan_service.ensure_plan(
        db, org_id=principal.org_id, interview_id=interview.id
    )
    if not plan.is_editable:
        raise plan_service.PlanFrozenError()

    # Commit before enqueueing, or the worker can read a row that is not yet
    # visible to it.
    await db.commit()
    generate_plan.delay(
        str(principal.org_id),
        str(interview_id),
        payload.question_count,
        payload.duration_minutes,
    )
    return QuestionPlanRead.model_validate(await plan_service.get_plan(db, plan.id))


@router.put(
    "/questions",
    response_model=QuestionPlanRead,
    summary="Replace the question list",
)
async def replace_questions(
    interview_id: uuid.UUID,
    payload: QuestionsReplace,
    db: ScopedSession,
    _: Recruiter,
) -> QuestionPlanRead:
    plan = await _load(db, interview_id)
    updated = await plan_service.replace_questions(
        db,
        plan=plan,
        questions=[q.model_dump() for q in payload.questions],
        expected_version=payload.expected_version,
    )
    return QuestionPlanRead.model_validate(updated)


@router.put(
    "/criteria",
    response_model=QuestionPlanRead,
    summary="Replace the rubric",
)
async def replace_criteria(
    interview_id: uuid.UUID,
    payload: CriteriaReplace,
    db: ScopedSession,
    _: Recruiter,
) -> QuestionPlanRead:
    # Weights must sum to 1.0. A human edit is deliberate, so this is rejected
    # rather than normalised the way model output is.
    plan = await _load(db, interview_id)
    updated = await plan_service.replace_criteria(
        db,
        plan=plan,
        criteria=[c.model_dump() for c in payload.criteria],
        expected_version=payload.expected_version,
    )
    return QuestionPlanRead.model_validate(updated)


@router.post(
    "/approve",
    response_model=QuestionPlanRead,
    summary="Mark the plan ready to interview with",
)
async def approve_plan(
    interview_id: uuid.UUID,
    payload: ApproveRequest,
    db: ScopedSession,
    _: Recruiter,
) -> QuestionPlanRead:
    plan = await _load(db, interview_id)
    approved = await plan_service.approve(
        db, plan=plan, expected_version=payload.expected_version
    )
    return QuestionPlanRead.model_validate(
        await plan_service.get_plan(db, approved.id)
    )
