"""Generate a question plan and rubric.

Async because it is slow and because it must not be on the request path: a
recruiter clicking "invite" should not wait 20 seconds for a model, and a model
outage should delay the plan rather than fail the invite.

Idempotent on ``interview_id``. A duplicate delivery finds a READY plan and
returns without spending another generation.
"""

from __future__ import annotations

import uuid

import structlog
from celery import shared_task

from app.core.exceptions import NotFoundError
from app.db.session import tenant_session
from app.models.question_plan import PlanGenerationStatus
from app.modules.interview import service as interview_service
from app.modules.jobs import service as jobs_service
from app.modules.question_plan import generator
from app.modules.question_plan import service as plan_service
from app.modules.resume import retriever
from app.workers.celery_app import run_async

log = structlog.get_logger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 30

RETRIEVAL_TOP_K = 8


@shared_task(
    bind=True,
    name="plan.generate",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def generate_plan(  # type: ignore[no-untyped-def]
    self,
    org_id: str,
    interview_id: str,
    question_count: int = generator.DEFAULT_QUESTION_COUNT,
    duration_minutes: int = generator.DEFAULT_DURATION_MINUTES,
) -> dict:
    return run_async(
        _generate(
            uuid.UUID(org_id),
            uuid.UUID(interview_id),
            question_count=question_count,
            duration_minutes=duration_minutes,
        )
    )


async def _generate(
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    *,
    question_count: int,
    duration_minutes: int,
) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        plan = await plan_service.get_for_interview(session, interview_id)
        if plan is None:
            return {"interview_id": str(interview_id), "skipped": "no plan row"}
        if plan.generation_status is PlanGenerationStatus.READY:
            return {"plan_id": str(plan.id), "skipped": "already generated"}
        if not plan.is_editable:
            return {"plan_id": str(plan.id), "skipped": "frozen"}

        plan.generation_status = PlanGenerationStatus.GENERATING
        plan.error = None
        await session.flush()

        context = await _gather_context(session, plan, interview_id)
        plan_id = plan.id

    # The model call is outside the transaction on purpose: it takes tens of
    # seconds, and holding a pooled connection through it starves the API.
    try:
        generated, model_name = await generator.generate(
            job_title=context["job_title"],
            job_description=context["job_description"],
            resume_context=context["resume_context"],
            question_count=question_count,
            duration_minutes=duration_minutes,
        )
    except Exception as exc:
        async with tenant_session(org_id, "system", None) as session:
            plan = await plan_service.get_plan(session, plan_id)
            await plan_service.mark_failed(session, plan=plan, error=str(exc))
        log.warning("plan_generation_failed", plan_id=str(plan_id), error=str(exc))
        # Re-raised so Celery retries: unlike an unparseable resume, a model
        # failure is usually transient and worth another attempt.
        raise

    async with tenant_session(org_id, "system", None) as session:
        plan = await plan_service.get_plan(session, plan_id)
        await plan_service.apply_generated(
            session, plan=plan, generated=generated, model_name=model_name
        )

    return {
        "plan_id": str(plan_id),
        "questions": len(generated.questions),
        "criteria": len(generated.criteria),
    }


async def _gather_context(session, plan, interview_id: uuid.UUID) -> dict[str, str]:
    """Job description and retrieved resume chunks.

    Both are optional. An interview with no job attached and no resume uploaded
    still gets a plan -- a generic one, which is better than none and is what a
    recruiter who skipped both steps has asked for.
    """
    job_title = "the role"
    job_description = "(no job description was provided)"

    # Loaded explicitly rather than through plan.interview: that relationship is
    # unloaded here, and a lazy load under asyncio is a MissingGreenlet, not a
    # query.
    interview = await interview_service.get_interview(session, interview_id)

    if interview.job_id is not None:
        # NotFoundError only. A deleted job must not fail the plan, but a
        # broader except would swallow real errors -- it previously hid a
        # MissingGreenlet raised on the line above.
        try:
            job = await jobs_service.get_job(session, interview.job_id)
        except NotFoundError:
            log.warning("plan_job_missing", interview_id=str(interview_id))
        else:
            job_title = job.title
            description = await jobs_service.get_active_description(session, job.id)
            if description is not None:
                job_description = description.content
                plan.job_description_id = description.id

    resume_context = ""
    resume = await retriever.latest_ready_resume(session, interview.candidate_id)
    if resume is not None:
        plan.resume_id = resume.id
        chunks = await retriever.search(
            session,
            resume_id=resume.id,
            # No plan exists yet, so there is no question to retrieve against;
            # the role and its description stand in for one.
            query=f"{job_title}. {job_description}"[:2000],
            top_k=RETRIEVAL_TOP_K,
        )
        resume_context = "\n\n".join(c.content for c in chunks)

    await session.flush()
    return {
        "job_title": job_title,
        "job_description": job_description,
        "resume_context": resume_context,
    }
