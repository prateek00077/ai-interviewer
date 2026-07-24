"""Similarity search over resume chunks for interview context.

Consumers: the question-plan generator (offline) and the live interview context
builder (on the turn budget). Both ask the same question -- "which parts of this
CV are relevant to X" -- so both go through here.

``<=>`` is pgvector's cosine distance operator, and it must match the operator
class the HNSW index was built with (``vector_cosine_ops``). A mismatch does not
error; it silently falls back to a sequential scan.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import Float, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.integrations import nim_client
from app.models.resume import Resume, ResumeChunk, ResumeStatus

log = structlog.get_logger(__name__)

DEFAULT_TOP_K = 6

# Cosine distance, so lower is closer.
#
# MEASURED, not guessed. Against nv-embedqa-e5-v5 on a real CV, distances come
# back compressed into a narrow band rather than spread over [0, 2]:
#
#   clearly relevant   0.52 - 0.67   ("Kafka experience" -> the Kafka role)
#   unrelated          0.75 - 0.84   ("underwater basket weaving" -> anything)
#
# So this is a backstop against pathological mismatch, NOT a relevance gate --
# ranking does the real work, and top_k does the trimming. An earlier value of
# 0.65 sat inside the relevant band and silently dropped the education chunk from
# "what did you study at university?", which ranked first.
#
# Re-measure before trusting this if the embedding model ever changes.
MAX_DISTANCE = 0.80


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    content: str
    section: str | None
    distance: float


async def latest_ready_resume(session: AsyncSession, candidate_id: uuid.UUID) -> Resume | None:
    """The resume to interview against: newest that finished processing.

    Explicitly not "newest resume" -- a candidate who starts a second upload two
    minutes before joining must not blank out the context that already works.
    """
    return (
        await session.execute(
            select(Resume)
            .where(Resume.candidate_id == candidate_id, Resume.status == ResumeStatus.READY)
            .order_by(Resume.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def has_incoming_resume(session: AsyncSession, candidate_id: uuid.UUID) -> bool:
    """True when a CV is uploaded and still being processed -- UPLOADED or PARSING.

    The point is the race the plan generator loses: a recruiter (or the invite)
    starts generation while the candidate's upload is mid-flight, so
    ``latest_ready_resume`` finds nothing and the plan is written from the job
    description alone. That plan then has to be thrown away and regenerated once
    the CV lands, and the recruiter sees generic questions in between.

    NOT PENDING: a presigned URL that was issued but never uploaded may never be,
    so waiting on it would stall a plan forever. NOT READY/FAILED either -- those
    are settled, and there is nothing left to wait for.
    """
    row = await session.execute(
        select(Resume.id)
        .where(
            Resume.candidate_id == candidate_id,
            Resume.status.in_((ResumeStatus.UPLOADED, ResumeStatus.PARSING)),
        )
        .limit(1)
    )
    return row.scalar_one_or_none() is not None


async def search(
    session: AsyncSession,
    *,
    resume_id: uuid.UUID,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = MAX_DISTANCE,
) -> list[RetrievedChunk]:
    """Top-k chunks of one resume by cosine similarity to ``query``."""
    # "query", not "passage" -- the other half of the asymmetric pair the stored
    # chunks were embedded with.
    vector = await nim_client.embed_one(query, input_type="query")
    return await search_by_vector(
        session, resume_id=resume_id, vector=vector, top_k=top_k, max_distance=max_distance
    )


async def search_by_vector(
    session: AsyncSession,
    *,
    resume_id: uuid.UUID,
    vector: list[float],
    top_k: int = DEFAULT_TOP_K,
    max_distance: float = MAX_DISTANCE,
) -> list[RetrievedChunk]:
    """The embedding-free half, so callers with a cached vector skip the API call."""
    distance = ResumeChunk.embedding.cosine_distance(vector).cast(Float).label("distance")

    rows = (
        await session.execute(
            select(ResumeChunk.content, ResumeChunk.section, distance)
            .where(ResumeChunk.resume_id == resume_id, ResumeChunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )
    ).all()

    results = [
        RetrievedChunk(content=r.content, section=r.section, distance=float(r.distance))
        for r in rows
        if r.distance is not None and float(r.distance) <= max_distance
    ]
    log.debug("resume_search", resume_id=str(resume_id), candidates=len(rows), kept=len(results))
    return results


async def full_text(session: AsyncSession, *, resume_id: uuid.UUID) -> str:
    """Every chunk of one resume, in document order.

    NOT a search. Similarity retrieval is the right tool on the turn budget,
    where the question is "which part of this CV is about what was just asked".
    It is the wrong tool for writing the question plan, and that mismatch is
    what shipped interviews whose questions named nothing the candidate had
    done: top-k against the job description ranks the chunks that sound like
    the vacancy, so a candidate's actual projects lose to whichever section
    happens to echo the JD's vocabulary, and the model then invents plausible
    experience to fill the gap.

    A resume is one or two pages. The whole thing fits in the prompt with room
    to spare, and the model cannot ground a question in a section it was never
    shown.
    """
    rows = (
        await session.execute(
            select(ResumeChunk.content)
            .where(ResumeChunk.resume_id == resume_id)
            .order_by(ResumeChunk.ordinal)
        )
    ).scalars().all()
    log.debug("resume_full_text", resume_id=str(resume_id), chunks=len(rows))
    return "\n\n".join(rows)


async def context_for(
    session: AsyncSession,
    *,
    candidate_id: uuid.UUID,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> str:
    """Retrieved chunks as one prompt-ready block, or empty if there is nothing.

    Empty is a normal outcome, not an error: a candidate may join without ever
    uploading a CV, and the interview must still run -- just without resume
    grounding.
    """
    resume = await latest_ready_resume(session, candidate_id)
    if resume is None:
        return ""

    chunks = await search(session, resume_id=resume.id, query=query, top_k=top_k)
    return "\n\n".join(chunk.content for chunk in chunks)
