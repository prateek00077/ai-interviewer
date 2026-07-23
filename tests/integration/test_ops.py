"""Liveness, readiness, metrics, and the drain.

The distinction under test is the one that matters operationally: /health must
stay green through a dependency outage (it drives restarts) while /ready must
go red (it drives load-balancer membership). Wiring them the same way is how a
database blip restarts every pod at once.
"""

import pytest

from app.api import ops

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _not_draining():
    """The flag is a module global; a test that sets it must not leak."""
    ops.set_draining(False)
    yield
    ops.set_draining(False)


# --- Liveness ---------------------------------------------------------------


async def test_health_is_always_ok_while_the_process_runs(api_client):
    response = await api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_health_stays_green_when_a_dependency_is_down(api_client, monkeypatch):
    """It drives restarts. Wiring it to Postgres would mean a database blip
    restarts every API pod simultaneously -- turning a recoverable outage into
    a thundering herd against a database that is already struggling."""

    async def _explode() -> None:
        raise RuntimeError("postgres is gone")

    monkeypatch.setattr(ops, "_check_postgres", _explode)
    assert (await api_client.get("/health")).status_code == 200


async def test_health_needs_no_credentials(api_client):
    """An orchestrator issuing the probe has none."""
    response = await api_client.get("/health")
    assert response.status_code != 401


# --- Readiness --------------------------------------------------------------


async def test_ready_reports_every_dependency_by_name(api_client):
    response = await api_client.get("/ready")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert set(body["checks"]) == {"postgres", "redis", "storage"}
    assert all(v == "ok" for v in body["checks"].values()), body["checks"]


async def test_ready_goes_red_when_a_dependency_fails(api_client, monkeypatch):
    async def _explode() -> None:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(ops, "_check_postgres", _explode)

    response = await api_client.get("/ready")
    # 503, not 500: this instance is temporarily unfit, not broken, and that is
    # what tells a load balancer to retry elsewhere.
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert "connection refused" in body["checks"]["postgres"]


async def test_one_probe_reports_everything_that_is_wrong(api_client, monkeypatch):
    """All checks run even when the first fails, so an operator does not have
    to fix one thing to discover the next."""

    async def _explode() -> None:
        raise RuntimeError("down")

    monkeypatch.setattr(ops, "_check_postgres", _explode)
    monkeypatch.setattr(ops, "_check_storage", _explode)

    body = (await api_client.get("/ready")).json()
    assert body["checks"]["postgres"] != "ok"
    assert body["checks"]["storage"] != "ok"
    assert body["checks"]["redis"] == "ok", "a healthy check was skipped after a failure"


async def test_a_hanging_dependency_does_not_hang_the_probe(api_client, monkeypatch):
    """A probe that hangs is worse than one that fails: the orchestrator waits
    out its own timeout on every check instead of getting a fast negative."""
    import asyncio

    async def _hang() -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(ops, "CHECK_TIMEOUT_SECS", 0.1)
    monkeypatch.setattr(ops, "_check_redis", lambda _request: _hang())

    response = await asyncio.wait_for(api_client.get("/ready"), timeout=5)
    assert response.status_code == 503
    assert "timed out" in response.json()["checks"]["redis"]


# --- Draining ---------------------------------------------------------------


async def test_a_draining_instance_reports_not_ready(api_client):
    ops.set_draining(True)
    response = await api_client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "draining"


async def test_a_draining_instance_is_still_alive(api_client):
    """Draining means "stop sending me new work", not "kill me". The live voice
    sessions still need to finish."""
    ops.set_draining(True)
    assert (await api_client.get("/health")).status_code == 200


async def test_a_draining_instance_refuses_a_new_voice_session(api_client, registered_org):
    """A candidate connecting during shutdown would otherwise get a pipeline
    built against a Redis pool that is seconds from closing -- failing a minute
    later mid-answer rather than at the door."""
    import uuid

    headers = {"Authorization": f"Bearer {registered_org['tokens']['access_token']}"}
    invite = (
        await api_client.post(
            "/api/v1/auth/invites",
            headers=headers,
            json={"candidate_email": f"c-{uuid.uuid4().hex[:8]}@example.com"},
        )
    ).json()
    redeemed = (
        await api_client.post(
            "/api/v1/auth/invites/redeem", json={"invite_token": invite["invite_token"]}
        )
    ).json()

    ops.set_draining(True)
    response = await api_client.post(
        "/api/v1/webrtc/offer",
        headers={"Authorization": f"Bearer {redeemed['interview_token']}"},
        json={"sdp": "v=0\r\n", "type": "offer"},
    )
    assert response.status_code == 409
    assert "shutting down" in response.text


# --- Metrics ----------------------------------------------------------------


async def test_metrics_reports_the_numbers_worth_paging_on(api_client):
    body = (await api_client.get("/metrics")).json()
    assert body["uptime_seconds"] >= 0
    assert body["draining"] is False
    # Live calls with real people on them: a deploy that drops these is visible
    # to candidates.
    assert body["voice_sessions_active"] == 0
    assert body["event_bus_pending"] == 0


async def test_metrics_reflects_the_drain(api_client):
    ops.set_draining(True)
    assert (await api_client.get("/metrics")).json()["draining"] is True
