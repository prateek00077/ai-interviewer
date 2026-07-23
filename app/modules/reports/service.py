"""Reading and writing the two report rows.

Two parallel sets of functions rather than one parameterised by audience. The
duplication is the same deliberate choice as the two tables and the two
templates: there is no ``get_report(interview_id, audience)`` here that a
caller could pass the wrong constant to.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.report import CandidateReport, RecruiterReport, ReportStatus

log = structlog.get_logger(__name__)


# --- Recruiter --------------------------------------------------------------


async def get_recruiter_report(
    session: AsyncSession, interview_id: uuid.UUID
) -> RecruiterReport | None:
    return (
        await session.execute(
            select(RecruiterReport).where(RecruiterReport.interview_id == interview_id)
        )
    ).scalar_one_or_none()


async def require_recruiter_report(
    session: AsyncSession, interview_id: uuid.UUID
) -> RecruiterReport:
    report = await get_recruiter_report(session, interview_id)
    if report is None:
        raise NotFoundError(
            "No report has been generated for this interview.",
            interview_id=str(interview_id),
        )
    return report


async def ensure_recruiter_report(
    session: AsyncSession, *, org_id: uuid.UUID, interview_id: uuid.UUID
) -> RecruiterReport:
    report = await get_recruiter_report(session, interview_id)
    if report is None:
        report = RecruiterReport(org_id=org_id, interview_id=interview_id)
        session.add(report)
        await session.flush()
    return report


# --- Candidate --------------------------------------------------------------


async def get_candidate_report(
    session: AsyncSession, interview_id: uuid.UUID
) -> CandidateReport | None:
    return (
        await session.execute(
            select(CandidateReport).where(CandidateReport.interview_id == interview_id)
        )
    ).scalar_one_or_none()


async def require_candidate_report(
    session: AsyncSession, interview_id: uuid.UUID
) -> CandidateReport:
    report = await get_candidate_report(session, interview_id)
    if report is None:
        raise NotFoundError(
            "No feedback has been prepared for this interview yet.",
            interview_id=str(interview_id),
        )
    return report


async def ensure_candidate_report(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    candidate_id: uuid.UUID,
) -> CandidateReport:
    report = await get_candidate_report(session, interview_id)
    if report is None:
        report = CandidateReport(
            org_id=org_id, interview_id=interview_id, candidate_id=candidate_id
        )
        session.add(report)
        await session.flush()
    return report


# --- Shared state transitions -----------------------------------------------
#
# These take a row of either type. That is safe in a way a shared *reader* is
# not: they touch only the status/error/key fields the two tables have in
# common, and none of them can move data across the audience boundary.


async def mark_rendering(session: AsyncSession, report: RecruiterReport | CandidateReport) -> None:
    report.status = ReportStatus.RENDERING
    report.error = None
    await session.flush()


async def mark_ready(
    session: AsyncSession, report: RecruiterReport | CandidateReport, *, s3_key: str
) -> None:
    report.s3_key = s3_key
    report.status = ReportStatus.READY
    report.error = None
    report.generated_at = datetime.now(UTC)
    await session.flush()


async def mark_failed(
    session: AsyncSession, report: RecruiterReport | CandidateReport, *, error: str
) -> None:
    """Record the failure and keep the previous PDF.

    ``s3_key`` is untouched on purpose: a re-render that fails should not turn
    a report someone is already reading into a broken link.
    """
    report.status = ReportStatus.FAILED
    report.error = error[:2000]
    await session.flush()
