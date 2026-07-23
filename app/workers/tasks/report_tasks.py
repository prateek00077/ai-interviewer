"""Render both PDF reports, then notify.

TWO TASKS, NOT ONE. Rendering both audiences in a single task would mean a
failure in the candidate half leaves the recruiter half unrendered on retry, or
-- worse -- that the retry re-renders the recruiter report with a partially
built candidate view in the same process. Separate tasks fail separately.

The candidate task is the sensitive one. It calls
``reports.candidate.generate``, whose parameter list contains no score-bearing
type, and it never loads a ``Score``. The recruiter view IS built first, but
only ``recruiter.topic_names`` crosses over -- the criterion names, which are
the subject headings of the interview and not its results.

Notification emails go last and never fail the task: the PDF is the work, the
email is the announcement.
"""

from __future__ import annotations

import uuid

import structlog
from celery import shared_task

from app.db.session import tenant_session
from app.integrations import email
from app.modules.interview import service as interview_service
from app.modules.reports import candidate as candidate_report
from app.modules.reports import recruiter as recruiter_report
from app.modules.reports import renderer
from app.modules.reports import service as reports_service
from app.modules.users import service as users_service
from app.workers.celery_app import run_async

log = structlog.get_logger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECS = 30


# --- Recruiter --------------------------------------------------------------


@shared_task(
    bind=True,
    name="reports.render_recruiter",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def render_recruiter_report(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    return run_async(_render_recruiter(uuid.UUID(org_id), uuid.UUID(interview_id)))


async def _render_recruiter(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        report = await reports_service.ensure_recruiter_report(
            session, org_id=org_id, interview_id=interview_id
        )
        await reports_service.mark_rendering(session, report)
        view = await recruiter_report.build(session, interview_id)
        report_id = report.id
        # Detached before the render: WeasyPrint takes seconds, and holding a
        # pooled connection through it starves the API.
        session.expunge_all()

    try:
        pdf = await renderer.render_pdf(renderer.RECRUITER_TEMPLATE, view)
        key = await renderer.publish(
            org_id=org_id, interview_id=interview_id, audience="recruiter", pdf=pdf
        )
    except Exception as exc:
        async with tenant_session(org_id, "system", None) as session:
            report = await reports_service.require_recruiter_report(session, interview_id)
            await reports_service.mark_failed(session, report, error=str(exc))
        raise

    async with tenant_session(org_id, "system", None) as session:
        report = await reports_service.require_recruiter_report(session, interview_id)
        await reports_service.mark_ready(session, report, s3_key=key)

    await _notify_recruiters(org_id, view)
    return {"interview_id": str(interview_id), "report_id": str(report_id), "s3_key": key}


async def _notify_recruiters(org_id: uuid.UUID, view: recruiter_report.RecruiterView) -> None:
    """Tell the org's active staff the report exists. Carries no findings.

    Wrapped whole: the report is rendered and stored by this point, and an
    unreachable mail server must not retry a task that would re-render a PDF
    each time.
    """
    try:
        async with tenant_session(org_id, "system", None) as session:
            recipients, _ = await users_service.list_users(session, limit=100, offset=0)
            addresses = [u.email for u in recipients if u.is_active]

        for address in addresses:
            await email.send_report_ready(
                to=address,
                candidate_name=view.candidate_name,
                job_title=view.job_title,
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("reports.recruiter_notify_failed", error=str(exc)[:200])


# --- Candidate --------------------------------------------------------------


@shared_task(
    bind=True,
    name="reports.render_candidate",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def render_candidate_report(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    return run_async(_render_candidate(uuid.UUID(org_id), uuid.UUID(interview_id)))


async def _render_candidate(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)
        person = await users_service.get_candidate(session, interview.candidate_id)
        candidate_name = person.full_name or ""
        candidate_email = person.email

        report = await reports_service.ensure_candidate_report(
            session,
            org_id=org_id,
            interview_id=interview_id,
            candidate_id=interview.candidate_id,
        )
        await reports_service.mark_rendering(session, report)

        # The ONLY things that cross from the assessment side: the role, the
        # subject headings, and the transcript. No score is loaded in this
        # function, and `generate` below has no parameter that could take one.
        source = await recruiter_report.build(session, interview_id)
        job_title = source.job_title
        topics = recruiter_report.topic_names(source)
        turns = list(source.turns)
        session.expunge_all()

    feedback = await candidate_report.generate(
        job_title=job_title, topic_names=topics, turns=turns
    )
    view = candidate_report.build_view(
        candidate_name=candidate_name, job_title=job_title, feedback=feedback
    )

    try:
        pdf = await renderer.render_pdf(renderer.CANDIDATE_TEMPLATE, view)
        key = await renderer.publish(
            org_id=org_id, interview_id=interview_id, audience="candidate", pdf=pdf
        )
    except Exception as exc:
        async with tenant_session(org_id, "system", None) as session:
            report = await reports_service.require_candidate_report(session, interview_id)
            await reports_service.mark_failed(session, report, error=str(exc))
        raise

    async with tenant_session(org_id, "system", None) as session:
        report = await reports_service.require_candidate_report(session, interview_id)
        report.summary = feedback.summary
        report.strengths = [item.model_dump() for item in feedback.strengths]
        report.growth_areas = [item.model_dump() for item in feedback.growth_areas]
        await reports_service.mark_ready(session, report, s3_key=key)

    await email.send_candidate_feedback_ready(
        to=candidate_email, candidate_name=candidate_name, job_title=job_title
    )
    return {
        "interview_id": str(interview_id),
        "s3_key": key,
        "strengths": len(feedback.strengths),
        "growth_areas": len(feedback.growth_areas),
    }
