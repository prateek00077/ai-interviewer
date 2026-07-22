"""Chunk storage and vector retrieval, against real Postgres and the live NIM
embedding endpoint.

Marked ``nim`` because it spends real API calls. What it protects is the part of
the pipeline that fails *silently*: an embedding written to the wrong side of the
asymmetric model, or a distance threshold tuned into the relevant band, produces
no error anywhere -- just quietly worse interview questions.
"""

import uuid

import pytest

from app.models.resume import Resume, ResumeChunk, ResumeStatus
from app.modules.resume import embedder, retriever
from app.modules.resume.chunker import Chunk

pytestmark = [pytest.mark.integration, pytest.mark.nim]

# A miniature CV with clearly separated topics, so a ranking failure is
# unambiguous rather than a judgement call.
CHUNKS = [
    Chunk(0, "experience", "[experience] Senior Engineer at Northwind. Migrated the event bus "
                           "from RabbitMQ to Kafka with zero downtime."),
    Chunk(1, "experience", "[experience] Engineer at Contoso. Built the billing pipeline on "
                           "Postgres and Airflow, cutting month-end close to eleven minutes."),
    Chunk(2, "education", "[education] B.Tech in Computer Science, IIT Madras, 2014 to 2018."),
    Chunk(3, "skills", "[skills] Python, Go, Postgres, Kafka, Kubernetes, Terraform, gRPC."),
]


@pytest.fixture
async def embedded_resume(tenant_session, org_a):
    """A resume with four real embeddings, torn down with its org."""
    resume_id = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(
            Resume(
                id=resume_id,
                org_id=org_a.org_id,
                candidate_id=org_a.candidate_id,
                s3_key=f"{org_a.org_id}/{resume_id}.pdf",
                filename="cv.pdf",
                content_type="application/pdf",
                status=ResumeStatus.READY,
            )
        )
        await s.flush()
        await embedder.store_chunks(
            s, org_id=org_a.org_id, resume_id=resume_id, chunks=CHUNKS
        )
        written = await embedder.embed_pending(s, resume_id=resume_id)
        assert written == len(CHUNKS)
    return resume_id


async def test_embeddings_are_stored_with_the_expected_width(
    tenant_session, org_a, embedded_resume
):
    from sqlalchemy import select

    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        rows = (
            (await s.execute(select(ResumeChunk).where(ResumeChunk.resume_id == embedded_resume)))
            .scalars()
            .all()
        )
    assert len(rows) == len(CHUNKS)
    assert all(len(r.embedding) == 1024 for r in rows)


async def test_embedding_is_idempotent(tenant_session, org_a, embedded_resume):
    """The retry path: a second run must find nothing pending and spend nothing."""
    async with tenant_session(org_a.org_id, "system", None) as s:
        assert await embedder.embed_pending(s, resume_id=embedded_resume) == 0


async def test_restoring_chunks_replaces_rather_than_accumulates(
    tenant_session, org_a, embedded_resume
):
    """Chunk boundaries move when the chunker changes, so ordinal N is not the
    same span across runs. Stale rows past the new end would linger."""
    from sqlalchemy import func, select

    async with tenant_session(org_a.org_id, "system", None) as s:
        await embedder.store_chunks(
            s, org_id=org_a.org_id, resume_id=embedded_resume, chunks=CHUNKS[:2]
        )
        count = await s.scalar(
            select(func.count())
            .select_from(ResumeChunk)
            .where(ResumeChunk.resume_id == embedded_resume)
        )
    assert count == 2


@pytest.mark.parametrize(
    "query,expected_section",
    [
        ("Tell me about your experience with event streaming and Kafka.", "experience"),
        ("What did you study at university?", "education"),
        ("Which programming languages do you know?", "skills"),
    ],
)
async def test_the_right_section_ranks_first(
    tenant_session, org_a, embedded_resume, query, expected_section
):
    """Ranking, not the absolute distance, is what this pipeline relies on."""
    async with tenant_session(org_a.org_id, "system", None) as s:
        hits = await retriever.search(s, resume_id=embedded_resume, query=query, top_k=3)

    assert hits, f"nothing retrieved for {query!r}"
    assert hits[0].section == expected_section
    # Sorted nearest-first.
    assert [h.distance for h in hits] == sorted(h.distance for h in hits)


async def test_the_distance_floor_does_not_cut_into_the_relevant_band(
    tenant_session, org_a, embedded_resume
):
    """A regression guard on MAX_DISTANCE.

    An earlier value of 0.65 sat inside the relevant band and silently dropped
    the education chunk from a question about education -- which ranked first.
    """
    async with tenant_session(org_a.org_id, "system", None) as s:
        hits = await retriever.search(
            s, resume_id=embedded_resume, query="What did you study at university?", top_k=1
        )
    assert hits, "the relevant chunk was filtered out by the distance floor"
    assert hits[0].section == "education"


async def test_context_for_returns_empty_when_there_is_no_ready_resume(
    tenant_session, org_a
):
    """A candidate may join without ever uploading a CV; the interview still runs."""
    async with tenant_session(org_a.org_id, "system", None) as s:
        context = await retriever.context_for(
            s, candidate_id=org_a.candidate_id, query="anything at all"
        )
    assert context == ""


async def test_context_for_builds_a_prompt_block(tenant_session, org_a, embedded_resume):
    async with tenant_session(org_a.org_id, "system", None) as s:
        context = await retriever.context_for(
            s, candidate_id=org_a.candidate_id, query="Kafka and event streaming", top_k=2
        )
    assert "Kafka" in context
    assert context.count("\n\n") >= 1
