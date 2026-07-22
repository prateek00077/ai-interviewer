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
    return {
        "resume_id": str(resume_id),
        "status": "READY",
        "chunks": len(chunks),
        "embedded": embedded,
    }
