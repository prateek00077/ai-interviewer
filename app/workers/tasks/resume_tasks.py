"""Parse, chunk, and embed an uploaded resume.

One task, not three. Parsing and chunking are pure CPU on bytes already in hand
and take milliseconds; splitting them across queue hops would add three
round-trips and two more partial-failure states to save nothing. The embedding
call is the only slow, failure-prone step, and it is already independently
retryable because ``embed_pending`` only touches rows still missing a vector.

The task takes ``org_id`` explicitly. A worker has no request and therefore no
token to derive tenancy from, so the caller passes it and the session is opened
with ``actor_kind="system"`` -- which the policies admit as staff without
pretending a person did the work.
"""

from __future__ import annotations

import uuid

import structlog
from celery import shared_task
from sqlalchemy import select

from app.core.config import settings
from app.db.session import tenant_session
from app.integrations import storage
from app.models.resume import ResumeStatus
from app.modules.resume import embedder, parser, service
from app.modules.resume.chunker import chunk_sections
from app.workers.celery_app import run_async

log = structlog.get_logger(__name__)

# Retries cover a NIM shed or an S3 blip. A parse failure is NOT retried -- the
# document will not become parseable on a second attempt.
MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 30


@shared_task(
    bind=True,
    name="resume.process",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def process_resume(self, org_id: str, resume_id: str) -> dict:  # type: ignore[no-untyped-def]
    """UPLOADED -> READY. Idempotent on ``resume_id``."""
    return run_async(_process(uuid.UUID(org_id), uuid.UUID(resume_id)))


async def _process(org_id: uuid.UUID, resume_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        resume = await service.get_resume(session, resume_id)

        # Already done, or never uploaded. Either way there is nothing to do, and
        # returning rather than raising keeps a duplicate delivery cheap.
        if resume.status in (ResumeStatus.READY, ResumeStatus.PENDING):
            return {"resume_id": str(resume_id), "status": resume.status.value, "skipped": True}

        resume.status = ResumeStatus.PARSING
        resume.error = None
        await session.flush()
        s3_key, content_type = resume.s3_key, resume.content_type

    data = await storage.get_bytes(bucket=settings.s3_bucket_resumes, key=s3_key)

    # Parse outside the transaction above: it is pure CPU and can be slow on a
    # large PDF, and holding a pooled connection through it starves the API.
    try:
        parsed = parser.parse(data, content_type)
    except parser.ResumeParseError as exc:
        async with tenant_session(org_id, "system", None) as session:
            resume = await service.get_resume(session, resume_id)
            resume.status = ResumeStatus.FAILED
            resume.error = str(exc)
        log.warning("resume_parse_failed", resume_id=str(resume_id), error=str(exc))
        # Not re-raised: an unparseable document is a final answer, and letting
        # Celery retry it would burn three attempts to reach the same conclusion.
        return {"resume_id": str(resume_id), "status": "FAILED", "error": str(exc)}

    chunks = chunk_sections(parsed.sections)

    async with tenant_session(org_id, "system", None) as session:
        resume = await service.get_resume(session, resume_id)
        resume.parsed = parsed.as_dict()
        await embedder.store_chunks(
            session, org_id=org_id, resume_id=resume_id, chunks=chunks
        )
        # If this raises, the task retries and embeds only the rows still missing
        # a vector -- the parse and the chunk rows survive.
        embedded = await embedder.embed_pending(session, resume_id=resume_id)
        resume.status = ResumeStatus.READY

    log.info(
        "resume_ready", resume_id=str(resume_id), chunks=len(chunks), embedded=embedded
    )
    regenerated = await _regenerate_plans_without_this_resume(org_id, resume_id)
    return {
        "plans_regenerated": regenerated,
        "resume_id": str(resume_id),
        "status": "READY",
        "chunks": len(chunks),
        "embedded": embedded,
    }


async def _regenerate_plans_without_this_resume(
    org_id: uuid.UUID, resume_id: uuid.UUID
) -> int:
    """Re-plan any of this candidate's interviews that were planned without a CV.

    THE ORDERING IS UNAVOIDABLE. Plan generation starts when the invite is
    created, because a candidate can redeem the link seconds later and an
    interview with no plan is worse than one planned from the job description
    alone. But the candidate uploads their resume *after* redeeming -- so the
    first generation almost always runs before the CV exists.

    OBSERVED: every question_plan row had ``resume_id = NULL`` while the
    candidate's resume sat READY, and the questions were generic because the
    prompt never saw their background. The whole point of ingesting a resume is
    that the questions reference the candidate's actual projects.

    So the resume finishing is what triggers a re-plan. Only for plans that do
    not already have one, and only while they are still editable -- a frozen
    plan is the record of what an in-flight interview is being conducted
    against and must not move underneath it.
    """
    from app.models.interview import Interview
    from app.models.question_plan import PlanStatus, QuestionPlan
    from app.workers.tasks.plan_tasks import generate_plan

    async with tenant_session(org_id, "system", None) as session:
        resume = await service.get_resume(session, resume_id)
        rows = (
            await session.execute(
                select(QuestionPlan.interview_id)
                .join(Interview, Interview.id == QuestionPlan.interview_id)
                .where(
                    Interview.candidate_id == resume.candidate_id,
                    QuestionPlan.resume_id.is_(None),
                    QuestionPlan.status != PlanStatus.FROZEN,
                )
            )
        ).scalars().all()

    for interview_id in rows:
        log.info(
            "resume_triggered_replan",
            resume_id=str(resume_id),
            interview_id=str(interview_id),
        )
        # force: this replan has the CV the first one lacked, so it must not be
        # dropped as a duplicate of the generation already running.
        generate_plan.delay(str(org_id), str(interview_id), force=True)
    return len(rows)
