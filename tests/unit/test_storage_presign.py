"""Presigned URLs must carry the host the *browser* reaches, not the host the
API reaches.

These are the same host on a laptop and diverge the moment the API runs in
Docker: the API talks to MinIO at ``minio:9000`` while the browser can only
resolve ``localhost:9000``. A presigned URL signed with the internal name loads
into the browser and dies as ``TypeError: Failed to fetch`` -- the request never
leaves, because ``minio`` is not a name the host can resolve. So presigning is
bound to ``s3_public_endpoint_url``, and everything else to ``s3_endpoint_url``.
"""

from urllib.parse import urlparse

import pytest

from app.core.config import settings
from app.integrations import storage


@pytest.fixture(autouse=True)
def _reset_clients():
    # The clients are cached per process; drop them so each test's endpoint
    # settings actually take effect, and leave a clean cache behind.
    storage.reset_client()
    yield
    storage.reset_client()


async def test_presign_uses_public_endpoint_when_it_differs(monkeypatch):
    """The Docker case: internal and browser-facing hosts are not the same."""
    monkeypatch.setattr(settings, "s3_endpoint_url", "http://minio:9000")
    monkeypatch.setattr(settings, "s3_public_endpoint_url", "http://localhost:9000")

    put = await storage.presign_put(
        bucket="resumes", key="o/c/x.pdf", content_type="application/pdf"
    )
    get = await storage.presign_get(bucket="resumes", key="o/c/x.pdf")

    assert urlparse(put.url).netloc == "localhost:9000"
    assert urlparse(get).netloc == "localhost:9000"


async def test_presign_falls_back_to_endpoint_when_public_unset(monkeypatch):
    """The host-local case: no public override, so presign == the plain endpoint."""
    monkeypatch.setattr(settings, "s3_endpoint_url", "http://localhost:9000")
    monkeypatch.setattr(settings, "s3_public_endpoint_url", None)

    put = await storage.presign_put(
        bucket="resumes", key="o/c/x.pdf", content_type="application/pdf"
    )

    assert urlparse(put.url).netloc == "localhost:9000"
