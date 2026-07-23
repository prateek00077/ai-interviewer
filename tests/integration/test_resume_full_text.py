"""The plan generator reads the WHOLE resume, not a slice ranked by the vacancy.

OBSERVED: retrieval ranked the candidate's chunks by similarity to the job
description, so on a MERN CV against a Python role the sections describing what
the person had actually built lost to whichever lines echoed the JD's
vocabulary. The model was then asked to write grounded questions from context
that did not contain the grounding, and it filled the gap by inventing
experience.

No embeddings here on purpose: ordering and completeness are the properties
under test, and both hold whether or not the vectors were ever written.
"""

import uuid

import pytest

from app.models.resume import Resume, ResumeStatus
from app.modules.resume import embedder, retriever
from app.modules.resume.chunker import Chunk

pytestmark = pytest.mark.integration

CHUNKS = [
    Chunk(0, "summary", "[summary] Full-stack developer, MERN."),
    Chunk(1, "projects", "[projects] TypingArena - multiplayer typing game, Socket.IO."),
    Chunk(2, "projects", "[projects] DailyOrbit - habit tracker on Cloudflare Workers."),
    Chunk(3, "skills", "[skills] React, Node.js, MongoDB, Redis, n8n."),
]


@pytest.fixture
async def resume_id(tenant_session, org_a):
    new_id = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(
            Resume(
                id=new_id,
                org_id=org_a.org_id,
                candidate_id=org_a.candidate_id,
                s3_key=f"{org_a.org_id}/{new_id}.pdf",
                filename="cv.pdf",
                content_type="application/pdf",
                status=ResumeStatus.READY,
            )
        )
        await s.flush()
        await embedder.store_chunks(
            s, org_id=org_a.org_id, resume_id=new_id, chunks=CHUNKS
        )
    return new_id


async def test_every_chunk_reaches_the_prompt(tenant_session, org_a, resume_id):
    """Including the ones a job description would never rank highly. A project
    the model was not shown is a project it can only invent around."""
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        text = await retriever.full_text(s, resume_id=resume_id)

    for chunk in CHUNKS:
        assert chunk.content in text


async def test_chunks_come_back_in_document_order(tenant_session, org_a, resume_id):
    """A CV read out of order reads as a different CV: the summary lands after
    the skills list, and dates stop lining up with the roles they belong to."""
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        text = await retriever.full_text(s, resume_id=resume_id)

    positions = [text.index(c.content) for c in CHUNKS]
    assert positions == sorted(positions)


async def test_another_resume_is_not_included(tenant_session, org_a, resume_id):
    """The filter is per-resume, not per-candidate: a candidate who uploaded
    twice must not be interviewed against both CVs at once."""
    other = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(
            Resume(
                id=other,
                org_id=org_a.org_id,
                candidate_id=org_a.candidate_id,
                s3_key=f"{org_a.org_id}/{other}.pdf",
                filename="old.pdf",
                content_type="application/pdf",
                status=ResumeStatus.READY,
            )
        )
        await s.flush()
        await embedder.store_chunks(
            s,
            org_id=org_a.org_id,
            resume_id=other,
            chunks=[Chunk(0, "projects", "[projects] An entirely different career.")],
        )
        text = await retriever.full_text(s, resume_id=resume_id)

    assert "entirely different career" not in text
