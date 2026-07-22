"""Embed chunks via NIM and store in pgvector.

Chunk rows are written first with a null embedding, then filled in. That split is
what makes the step retryable: an embedding call that fails halfway leaves the
extracted text in place, and a retry embeds only the rows still missing a vector
instead of re-parsing the document.
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations import nim_client
from app.models.resume import ResumeChunk
from app.modules.resume.chunker import Chunk

log = structlog.get_logger(__name__)


async def store_chunks(
    session: AsyncSession, *, org_id: uuid.UUID, resume_id: uuid.UUID, chunks: list[Chunk]
) -> list[ResumeChunk]:
    """Replace this resume's chunks with a fresh set, embeddings still null.

    Delete-then-insert rather than upsert: chunk boundaries shift when the
    chunker changes, so ordinal N is not the same span across two runs and
    updating in place would leave stale rows past the new end.
    """
    existing = (
        (await session.execute(select(ResumeChunk).where(ResumeChunk.resume_id == resume_id)))
        .scalars()
        .all()
    )
    for row in existing:
        await session.delete(row)
    await session.flush()

    rows = [
        ResumeChunk(
            org_id=org_id,
            resume_id=resume_id,
            ordinal=chunk.ordinal,
            section=chunk.section,
            content=chunk.content,
        )
        for chunk in chunks
    ]
    session.add_all(rows)
    await session.flush()
    return rows


async def embed_pending(session: AsyncSession, *, resume_id: uuid.UUID) -> int:
    """Fill in every missing embedding for a resume. Returns how many were written.

    Idempotent: called twice, the second call finds nothing pending and is a
    no-op, which is what lets the Celery task retry safely.
    """
    pending = (
        (
            await session.execute(
                select(ResumeChunk)
                .where(ResumeChunk.resume_id == resume_id, ResumeChunk.embedding.is_(None))
                .order_by(ResumeChunk.ordinal)
            )
        )
        .scalars()
        .all()
    )

    if not pending:
        return 0

    # "passage", not "query": nv-embedqa is asymmetric, and stored text belongs on
    # the passage side. Getting this backwards degrades every later search
    # silently, with nothing to notice at write time.
    vectors = await nim_client.embed([row.content for row in pending], input_type="passage")

    for row, vector in zip(pending, vectors, strict=True):
        row.embedding = vector
    await session.flush()

    log.info("resume_chunks_embedded", resume_id=str(resume_id), count=len(pending))
    return len(pending)
