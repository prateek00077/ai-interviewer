"""The candidate resume upload handshake, end to end against MinIO and Postgres.

The property under test is that the server never takes the client's word for it.
A candidate holding a presigned URL can upload nothing, upload something huge, or
skip the upload and call /complete anyway -- each of those has a test, and the
expected outcome is never "row promoted to UPLOADED".
"""

import uuid

import httpx
import pytest

from app.core.config import settings
from app.integrations import storage

pytestmark = pytest.mark.integration

PDF = "application/pdf"
FIXTURE_PDF = b"%PDF-1.4 fake but adequately sized payload for the size checks " * 4


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


@pytest.fixture(autouse=True)
def _no_worker(monkeypatch):
    """Capture the enqueue instead of dispatching to a real broker.

    The pipeline itself is exercised by test_resume_pipeline; here the only
    question is *whether* it is queued, and on which transition.
    """
    calls: list[tuple] = []
    monkeypatch.setattr(
        "app.api.v1.candidates.process_resume.delay",
        lambda *args: calls.append(args),
    )
    return calls


async def _candidate_token(api_client, org: dict) -> tuple[dict[str, str], str]:
    """An interview token, which is the only credential a candidate ever holds."""
    invite = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(org),
        json={"candidate_email": f"cand-{uuid.uuid4().hex[:8]}@example.com"},
    )
    assert invite.status_code == 201, invite.text
    redeemed = await api_client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": invite.json()["invite_token"]}
    )
    assert redeemed.status_code == 200, redeemed.text
    body = redeemed.json()
    return {"Authorization": f"Bearer {body['interview_token']}"}, body["candidate_id"]


async def _presign(api_client, headers: dict[str, str], **overrides) -> dict:
    payload = {"filename": "cv.pdf", "content_type": PDF} | overrides
    response = await api_client.post(
        "/api/v1/candidates/me/resume/presign", headers=headers, json=payload
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- The happy path ---------------------------------------------------------


async def test_presign_upload_complete(api_client, registered_org, _no_worker):
    headers, candidate_id = await _candidate_token(api_client, registered_org)
    presigned = await _presign(api_client, headers)

    # The browser PUTs straight to storage; the API never sees the bytes.
    async with httpx.AsyncClient() as client:
        put = await client.put(
            presigned["upload_url"],
            content=FIXTURE_PDF,
            headers={"content-type": presigned["content_type"]},
        )
    assert put.status_code == 200, put.text

    completed = await api_client.post(
        f"/api/v1/candidates/me/resume/{presigned['resume_id']}/complete", headers=headers
    )
    assert completed.status_code == 200, completed.text
    assert completed.json()["status"] == "UPLOADED"
    assert _no_worker == [(registered_org["org_id"], presigned["resume_id"])]


async def test_completing_twice_does_not_queue_the_pipeline_twice(
    api_client, registered_org, _no_worker
):
    """A retried /complete after a dropped response must be a no-op."""
    headers, _ = await _candidate_token(api_client, registered_org)
    presigned = await _presign(api_client, headers)
    async with httpx.AsyncClient() as client:
        await client.put(
            presigned["upload_url"],
            content=FIXTURE_PDF,
            headers={"content-type": presigned["content_type"]},
        )

    url = f"/api/v1/candidates/me/resume/{presigned['resume_id']}/complete"
    assert (await api_client.post(url, headers=headers)).status_code == 200
    second = await api_client.post(url, headers=headers)

    assert second.status_code == 200
    assert second.json()["status"] == "UPLOADED"
    assert len(_no_worker) == 1, "the pipeline was queued twice"


# --- What the server refuses to take on trust -------------------------------


async def test_completing_without_uploading_anything_is_rejected(
    api_client, registered_org, _no_worker
):
    headers, _ = await _candidate_token(api_client, registered_org)
    presigned = await _presign(api_client, headers)

    # No PUT at all -- just a claim that the upload happened.
    response = await api_client.post(
        f"/api/v1/candidates/me/resume/{presigned['resume_id']}/complete", headers=headers
    )
    assert response.status_code == 409
    assert _no_worker == []


async def test_an_oversized_upload_is_rejected_and_deleted(
    api_client, registered_org, monkeypatch, _no_worker
):
    """A plain presigned PUT cannot bind a size limit, so the HEAD check must."""
    monkeypatch.setattr(settings, "max_resume_bytes", 64)
    headers, _ = await _candidate_token(api_client, registered_org)
    presigned = await _presign(api_client, headers)

    async with httpx.AsyncClient() as client:
        put = await client.put(
            presigned["upload_url"],
            content=b"x" * 5000,
            headers={"content-type": presigned["content_type"]},
        )
    assert put.status_code == 200, "storage accepted it; the API is the only gate"

    response = await api_client.post(
        f"/api/v1/candidates/me/resume/{presigned['resume_id']}/complete", headers=headers
    )
    assert response.status_code == 409
    assert _no_worker == []


async def test_an_oversized_declared_size_is_refused_before_a_url_is_issued(
    api_client, registered_org
):
    headers, _ = await _candidate_token(api_client, registered_org)
    response = await api_client.post(
        "/api/v1/candidates/me/resume/presign",
        headers=headers,
        json={
            "filename": "huge.pdf",
            "content_type": PDF,
            "declared_size": settings.max_resume_bytes + 1,
        },
    )
    assert response.status_code == 409


async def test_an_unsupported_content_type_is_refused(api_client, registered_org):
    headers, _ = await _candidate_token(api_client, registered_org)
    response = await api_client.post(
        "/api/v1/candidates/me/resume/presign",
        headers=headers,
        json={"filename": "photo.png", "content_type": "image/png"},
    )
    assert response.status_code == 422


async def test_a_filename_cannot_carry_a_path(api_client, registered_org):
    headers, candidate_id = await _candidate_token(api_client, registered_org)
    presigned = await _presign(api_client, headers, filename="../../etc/passwd")

    listing = await api_client.get("/api/v1/candidates/me/resume", headers=headers)
    stored = listing.json()[0]["filename"]
    assert "/" not in stored and "\\" not in stored
    # And the storage key is server-generated regardless of what was sent.
    assert presigned["resume_id"] in [r["id"] for r in listing.json()]


async def test_the_storage_key_is_org_prefixed_and_unguessable(
    api_client, registered_org
):
    headers, candidate_id = await _candidate_token(api_client, registered_org)
    key = storage.resume_key(
        uuid.UUID(registered_org["org_id"]), uuid.UUID(candidate_id), "pdf"
    )
    assert key.startswith(f"{registered_org['org_id']}/{candidate_id}/")
    # Knowing both ids is not enough to derive the key.
    assert key != storage.resume_key(
        uuid.UUID(registered_org["org_id"]), uuid.UUID(candidate_id), "pdf"
    )


# --- Who may reach what -----------------------------------------------------


async def test_a_candidate_cannot_complete_another_candidates_upload(
    api_client, registered_org, _no_worker
):
    first_headers, _ = await _candidate_token(api_client, registered_org)
    presigned = await _presign(api_client, first_headers)

    second_headers, _ = await _candidate_token(api_client, registered_org)
    response = await api_client.post(
        f"/api/v1/candidates/me/resume/{presigned['resume_id']}/complete",
        headers=second_headers,
    )
    # Same org, different candidate: RLS hides the row entirely.
    assert response.status_code == 404


async def test_a_candidate_sees_only_their_own_uploads(api_client, registered_org):
    first_headers, _ = await _candidate_token(api_client, registered_org)
    await _presign(api_client, first_headers)

    second_headers, _ = await _candidate_token(api_client, registered_org)
    listing = await api_client.get("/api/v1/candidates/me/resume", headers=second_headers)
    assert listing.json() == []


async def test_a_recruiter_token_cannot_use_the_candidate_upload_routes(
    api_client, registered_org
):
    response = await api_client.post(
        "/api/v1/candidates/me/resume/presign",
        headers=_auth(registered_org),
        json={"filename": "cv.pdf", "content_type": PDF},
    )
    assert response.status_code == 403


async def test_a_recruiter_sees_the_upload_and_can_get_a_download_link(
    api_client, registered_org, _no_worker
):
    headers, candidate_id = await _candidate_token(api_client, registered_org)
    presigned = await _presign(api_client, headers)
    async with httpx.AsyncClient() as client:
        await client.put(
            presigned["upload_url"],
            content=FIXTURE_PDF,
            headers={"content-type": presigned["content_type"]},
        )
    await api_client.post(
        f"/api/v1/candidates/me/resume/{presigned['resume_id']}/complete", headers=headers
    )

    listing = await api_client.get(
        f"/api/v1/candidates/{candidate_id}/resumes", headers=_auth(registered_org)
    )
    assert listing.status_code == 200
    assert [r["id"] for r in listing.json()] == [presigned["resume_id"]]

    download = await api_client.get(
        f"/api/v1/candidates/{candidate_id}/resumes/{presigned['resume_id']}/download",
        headers=_auth(registered_org),
    )
    assert download.status_code == 200
    async with httpx.AsyncClient() as client:
        fetched = await client.get(download.json()["url"])
    assert fetched.content == FIXTURE_PDF


async def test_the_candidate_view_hides_parsed_fields_and_errors(
    api_client, registered_org, _no_worker
):
    """The extraction briefs the interviewer. Showing it invites re-uploading
    until the summary flatters."""
    headers, _ = await _candidate_token(api_client, registered_org)
    await _presign(api_client, headers)

    listing = await api_client.get("/api/v1/candidates/me/resume", headers=headers)
    row = listing.json()[0]
    assert "parsed" not in row
    assert "error" not in row
