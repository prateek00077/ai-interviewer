"""Recruiter report and candidate report endpoints.

Two audiences, and the separation runs all the way down: separate tables,
separate RLS policies, separate builders, separate templates, separate schemas,
separate routes. There is no endpoint here that takes an audience parameter.

The candidate routes resolve the interview from the token's own claim rather
than from a path parameter, exactly as ``/interviews/me`` does. A candidate
cannot ask about someone else's feedback because there is nowhere to put
someone else's id.
"""

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, status

from app.api.deps import Principal, ScopedSession, get_current_candidate, require_role
from app.core.config import settings
from app.core.exceptions import ConflictError, NotFoundError
from app.models.report import ReportStatus
from app.models.user import UserRole
from app.modules.interview import service as interview_service
from app.modules.interview import state_machine
from app.modules.reports import renderer
from app.modules.reports import service as reports_service
from app.schemas.report import (
    CandidateFeedbackRead,
    RecruiterReportRead,
    ReportLink,
)

log = structlog.get_logger(__name__)

router = APIRouter(tags=["reports"])

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]
CurrentCandidate = Annotated[Principal, Depends(get_current_candidate)]


def _require_ready(report: object, *, what: str) -> str:
    status_value = getattr(report, "status", None)
    key = getattr(report, "s3_key", None)
    if status_value is not ReportStatus.READY or not key:
        raise NotFoundError(
            f"The {what} is not ready yet.",
            status=getattr(status_value, "value", "UNKNOWN"),
        )
    return str(key)


# --- Recruiter --------------------------------------------------------------


@router.get(
    "/interviews/{interview_id}/report",
    response_model=RecruiterReportRead,
    summary="Status of the recruiter report",
)
async def get_recruiter_report(
    interview_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> RecruiterReportRead:
    await interview_service.get_interview(db, interview_id)
    report = await reports_service.require_recruiter_report(db, interview_id)
    return RecruiterReportRead.model_validate(report)


@router.get(
    "/interviews/{interview_id}/report/download",
    response_model=ReportLink,
    summary="A short-lived link to the recruiter report PDF",
)
async def download_recruiter_report(
    interview_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> ReportLink:
    await interview_service.get_interview(db, interview_id)
    report = await reports_service.require_recruiter_report(db, interview_id)
    key = _require_ready(report, what="report")
    return ReportLink(
        download_url=await renderer.download_url(key),
        expires_in=settings.report_download_ttl_secs,
        generated_at=report.generated_at,
    )


@router.post(
    "/interviews/{interview_id}/report/regenerate",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=RecruiterReportRead,
    summary="Re-render both reports for an interview",
)
async def regenerate(
    interview_id: uuid.UUID, db: ScopedSession, principal: Recruiter
) -> RecruiterReportRead:
    """Queue a fresh render of both PDFs.

    Both, not just the recruiter's: they are two views of one interview, and
    letting them drift to different vintages is how a candidate receives
    feedback that contradicts the report the recruiter is reading.

    The existing PDFs stay downloadable until the new ones replace them.
    """
    interview = await interview_service.get_interview(db, interview_id)
    if not state_machine.is_terminal(interview.status):
        raise ConflictError(
            "This interview has not finished yet.", current_status=interview.status.value
        )

    report = await reports_service.ensure_recruiter_report(
        db, org_id=principal.org_id, interview_id=interview_id
    )
    # Committed before the tasks are sent, so a worker cannot read a row this
    # request has not written yet.
    await db.commit()

    from app.workers.tasks import report_tasks

    args = (str(principal.org_id), str(interview_id))
    report_tasks.render_recruiter_report.delay(*args)
    report_tasks.render_candidate_report.delay(*args)

    log.info(
        "reports.regenerate_requested",
        interview_id=str(interview_id),
        actor=str(principal.actor_id),
    )
    return RecruiterReportRead.model_validate(report)


# --- Candidate --------------------------------------------------------------


@router.get(
    "/reports/me",
    response_model=CandidateFeedbackRead,
    summary="The candidate's own interview feedback",
)
async def get_own_feedback(
    db: ScopedSession, principal: CurrentCandidate
) -> CandidateFeedbackRead:
    if principal.interview_id is None:
        raise NotFoundError("This token is not tied to an interview.")
    report = await reports_service.require_candidate_report(db, principal.interview_id)
    return CandidateFeedbackRead.model_validate(report)


@router.get(
    "/reports/me/download",
    response_model=ReportLink,
    summary="A short-lived link to the candidate's own feedback PDF",
)
async def download_own_feedback(db: ScopedSession, principal: CurrentCandidate) -> ReportLink:
    if principal.interview_id is None:
        raise NotFoundError("This token is not tied to an interview.")
    report = await reports_service.require_candidate_report(db, principal.interview_id)
    key = _require_ready(report, what="feedback")
    return ReportLink(
        download_url=await renderer.download_url(key),
        expires_in=settings.report_download_ttl_secs,
        generated_at=report.generated_at,
    )
