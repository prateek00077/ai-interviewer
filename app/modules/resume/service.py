"""Resume upload handshake and lifecycle.

The interesting part is ``complete``: it is the boundary between what a client
claims and what is true. A candidate holding a presigned URL can PUT anything,
or nothing, and then call /complete regardless. So /complete asks S3 directly --
does this key exist, how big is it, what type is it -- and only promotes the row
to UPLOADED if the answer is acceptable. A resume that never gets past PENDING
is invisible to everything downstream.
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import ConflictError, NotFoundError
from app.integrations import storage
from app.integrations.storage import RESUME_CONTENT_TYPES
from app.models.resume import Resume, ResumeStatus

log = structlog.get_logger(__name__)


async def start_upload(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    candidate_id: uuid.UUID,
    filename: str,
    content_type: str,
    declared_size: int | None = None,
) -> tuple[Resume, storage.PresignedUpload]:
    """Reserve a row and issue a presigned PUT.

    The row is created PENDING *before* the URL is handed out, so the key is
    server-chosen and already recorded. A client cannot invent a key, and an
    abandoned upload leaves an obviously-incomplete row rather than an orphaned
    object nobody can attribute.
    """
    if declared_size is not None and declared_size > settings.max_resume_bytes:
        raise ConflictError(
            f"Resume exceeds the {settings.max_resume_bytes // (1024 * 1024)}MB limit.",
            declared_size=declared_size,
        )

    extension = RESUME_CONTENT_TYPES[content_type]
    key = storage.resume_key(org_id, candidate_id, extension)

    resume = Resume(
        org_id=org_id,
        candidate_id=candidate_id,
        s3_key=key,
        filename=filename,
        content_type=content_type,
        status=ResumeStatus.PENDING,
    )
    session.add(resume)
    await session.flush()

    upload = await storage.presign_put(
        bucket=settings.s3_bucket_resumes,
        key=key,
        content_type=content_type,
        max_bytes=settings.max_resume_bytes,
    )
    log.info("resume_upload_started", resume_id=str(resume.id), candidate_id=str(candidate_id))
    return resume, upload


async def complete_upload(
    session: AsyncSession, *, resume_id: uuid.UUID, candidate_id: uuid.UUID | None = None
) -> tuple[Resume, bool]:
    """Verify the object with S3, then promote the row.

    Returns ``(resume, transitioned)``. The flag is what the caller keys the
    "enqueue processing" side effect off: a client retrying after a dropped
    response gets the same row back with ``transitioned=False``, so the pipeline
    is queued exactly once. Returning only the row is not enough -- its status is
    UPLOADED on both calls, which cannot distinguish them.

    ``candidate_id`` narrows the lookup when a candidate calls this. RLS already
    confines them to their own rows, so this is defence in depth rather than the
    primary control.
    """
    resume = await get_resume(session, resume_id)
    if candidate_id is not None and resume.candidate_id != candidate_id:
        raise NotFoundError("Resume not found.", resume_id=str(resume_id))

    # Already completed, or already processed further. Nothing to do.
    if resume.status is not ResumeStatus.PENDING:
        return resume, False

    info = await storage.head_object(bucket=settings.s3_bucket_resumes, key=resume.s3_key)
    if info is None:
        raise ConflictError("No file was uploaded to the issued URL.", resume_id=str(resume_id))
    if info.size > settings.max_resume_bytes:
        # The upload happened despite the limit -- a plain presigned PUT cannot
        # enforce size. Delete it rather than leaving a paid-for object behind.
        await storage.delete_object(bucket=settings.s3_bucket_resumes, key=resume.s3_key)
        resume.status = ResumeStatus.FAILED
        resume.error = f"Uploaded file is {info.size} bytes, over the limit."
        await session.flush()
        raise ConflictError("Uploaded file exceeds the size limit.", size=info.size)

    resume.size_bytes = info.size
    resume.status = ResumeStatus.UPLOADED
    await session.flush()

    log.info("resume_uploaded", resume_id=str(resume.id), size=info.size)
    return resume, True


async def get_resume(session: AsyncSession, resume_id: uuid.UUID) -> Resume:
    resume = await session.get(Resume, resume_id)
    if resume is None:
        raise NotFoundError("Resume not found.", resume_id=str(resume_id))
    return resume


async def list_for_candidate(session: AsyncSession, candidate_id: uuid.UUID) -> list[Resume]:
    rows = (
        await session.execute(
            select(Resume)
            .where(Resume.candidate_id == candidate_id)
            .order_by(Resume.created_at.desc())
        )
    ).scalars()
    return list(rows)


async def download_url(session: AsyncSession, resume_id: uuid.UUID) -> tuple[str, int]:
    """A time-limited link to the original file, for a recruiter."""
    resume = await get_resume(session, resume_id)
    if resume.status is ResumeStatus.PENDING:
        raise ConflictError("This resume was never uploaded.", resume_id=str(resume_id))
    url = await storage.presign_get(bucket=settings.s3_bucket_resumes, key=resume.s3_key)
    return url, settings.s3_presign_ttl_secs
