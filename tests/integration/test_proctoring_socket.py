"""The proctoring WebSocket, driven by a real client.

The socket is the only place in this API where the caller IS the person being
assessed, so this file is almost entirely about what a hostile client cannot
do: forge a severity, claim a server-derived signal, or reset its own
escalation by reconnecting.

Synchronous ``TestClient`` rather than the async ``api_client``: httpx's ASGI
transport does not speak WebSocket. It runs the app in its own event loop, and
the app's lifespan disposes the database engine on exit, so the pool is rebuilt
per loop rather than being shared across two.

Assertions go through the recruiter report endpoint rather than a direct
database read, which keeps everything on one loop and tests the path a real
reviewer would use.
"""

import uuid

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """Function-scoped deliberately.

    A module-scoped TestClient does not survive across tests here: its anyio
    portal and the app lifespan are torn down after the first test, and every
    later socket connects to an app whose receive loop never runs -- producing
    an ack but persisting nothing, which is a confusing way to fail.
    """
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _clear_rate_limits():
    """The registration limiter is real and every test here registers from the
    same address, so without this the module trips its own 429 after five tests.

    A synchronous client on purpose: this file has no event loop of its own, and
    borrowing the app's async Redis would mean reaching into TestClient's.
    """
    import redis as sync_redis

    from app.core.config import settings

    connection = sync_redis.Redis.from_url(settings.redis_url)
    try:
        keys = list(connection.scan_iter(match="rl:*"))
        if keys:
            connection.delete(*keys)
    finally:
        connection.close()


@pytest.fixture
def session(client):
    """A recruiter, a job, an interview, and a candidate token -- all over HTTP."""
    slug = f"ws-{uuid.uuid4().hex[:10]}"
    org = client.post(
        "/api/v1/auth/register-org",
        json={
            "org_name": "WS Co",
            "slug": slug,
            "admin_email": f"admin@{slug}.example.com",
            "admin_password": "correct-horse-battery-staple",
        },
    ).json()
    recruiter = {"Authorization": f"Bearer {org['tokens']['access_token']}"}

    job = client.post(
        "/api/v1/jobs", headers=recruiter, json={"title": "Staff Engineer"}
    ).json()
    invite = client.post(
        "/api/v1/auth/invites",
        headers=recruiter,
        json={"candidate_email": f"cand@{slug}.example.com", "job_id": job["id"]},
    ).json()
    redeemed = client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": invite["invite_token"]}
    ).json()

    return {
        "recruiter": recruiter,
        "job_id": job["id"],
        "interview_id": invite["interview_id"],
        "invite_token": invite["invite_token"],
        "token": redeemed["interview_token"],
    }


def _report(client, session) -> dict:
    response = client.get(
        f"/api/v1/interviews/{session['interview_id']}/proctoring",
        headers=session["recruiter"],
    )
    assert response.status_code == 200, response.text
    return response.json()


def _types(report: dict) -> list[str]:
    return [e["event_type"] for e in report["events"]]


def _url(session) -> str:
    return f"/api/v1/proctoring/ws?token={session['token']}"


def _send(ws, message) -> int:
    """Send one event and wait for the server's ack. Returns the accepted count.

    Waiting matters: without it the socket can close before the server has read
    what was queued, and the test would assert on events the server never saw.
    """
    ws.send_json(message)
    return ws.receive_json()["accepted"]


# --- Authentication ----------------------------------------------------------


@pytest.mark.parametrize("token", ["", "not-a-token", "a.b.c"])
def test_the_socket_rejects_an_unusable_token(client, token):
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/api/v1/proctoring/ws?token={token}"):
            pass


def test_the_socket_rejects_a_recruiter_access_token(client, session):
    """Only an interview token opens this socket. An access token is signed with
    a different derived key and fails at signature verification, not at a claim
    comparison."""
    token = session["recruiter"]["Authorization"].removeprefix("Bearer ")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/api/v1/proctoring/ws?token={token}"):
            pass


def test_a_terminated_interview_refuses_the_socket(client, session):
    client.post(
        f"/api/v1/interviews/{session['interview_id']}/terminate",
        headers=session["recruiter"],
        json={},
    )
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(_url(session)):
            pass


# --- What a well-behaved client can do ---------------------------------------


def test_reported_events_reach_the_recruiter_report(client, session):
    with client.websocket_connect(_url(session)) as ws:
        _send(ws, {"type": "TAB_BLUR", "offset_ms": 1000})
        _send(ws, {"type": "TAB_FOCUS", "offset_ms": 1500})

    report = _report(client, session)
    assert _types(report) == ["TAB_BLUR", "TAB_FOCUS"]
    assert report["events"][0]["offset_ms"] == 1000


def test_severity_is_assigned_by_the_server(client, session):
    """The rules decide what an event is worth, not the browser."""
    with client.websocket_connect(_url(session)) as ws:
        _send(ws, {"type": "TAB_BLUR"})
        _send(ws, {"type": "PASTE"})

    by_type = {e["event_type"]: e["severity"] for e in _report(client, session)["events"]}
    assert by_type["TAB_BLUR"] == "INFO"
    # Pasting into a spoken interview means text arrived from somewhere.
    assert by_type["PASTE"] == "WARN"


def test_the_server_timestamps_every_event(client, session):
    """A client-supplied time would let a candidate backdate an event out of the
    interview window entirely."""
    with client.websocket_connect(_url(session)) as ws:
        _send(ws, {"type": "TAB_BLUR", "at": "1999-01-01T00:00:00Z"})

    events = _report(client, session)["events"]
    assert events[0]["at"].startswith("20"), "a client timestamp was stored"


# --- What a hostile client cannot do -----------------------------------------


def test_a_client_cannot_forge_its_own_severity(client, session):
    """A candidate grading their own conduct is the whole failure this stops."""
    with client.websocket_connect(_url(session)) as ws:
        _send(ws, {"type": "PASTE", "severity": "INFO"})

    events = _report(client, session)["events"]
    assert [e["severity"] for e in events] == ["WARN"]


def test_a_client_cannot_claim_a_server_derived_signal(client, session):
    """SECOND_SPEAKER and MULTIPLE_FACES come from audio and vision. A client
    claiming them -- in either direction -- fabricates evidence about itself."""
    with client.websocket_connect(_url(session)) as ws:
        for forged in ("SECOND_SPEAKER", "MULTIPLE_FACES", "FACE_ABSENT", "ANOMALOUS_SILENCE"):
            assert _send(ws, {"type": forged}) == 0, f"{forged} was accepted"
        assert _send(ws, {"type": "TAB_BLUR"}) == 1  # a legitimate one, sent last

    # Only the legitimate event survived, and the socket stayed usable.
    assert _types(_report(client, session)) == ["TAB_BLUR"]


@pytest.mark.parametrize(
    "message",
    [
        {"type": "NOT_A_REAL_EVENT"},
        {"nonsense": True},
        {"type": 12345},
        {"type": "TAB_BLUR", "offset_ms": -5},
        {},
    ],
)
def test_malformed_messages_are_dropped_without_closing_the_socket(client, session, message):
    with client.websocket_connect(_url(session)) as ws:
        assert _send(ws, message) == 0, "a malformed message was accepted"
        assert _send(ws, {"type": "TAB_BLUR"}) == 1, "the socket stopped working"

    assert _types(_report(client, session)) == ["TAB_BLUR"]


def test_escalation_survives_a_reconnect(client, session):
    """Otherwise a candidate resets their own escalation simply by reconnecting,
    which the multi-use invite makes trivial."""
    client.put(
        f"/api/v1/jobs/{session['job_id']}/proctoring-policy",
        headers=session["recruiter"],
        json={"blur_limit": 1},
    )

    with client.websocket_connect(_url(session)) as ws:
        _send(ws, {"type": "TAB_BLUR"})  # 1st: within the limit, INFO

    # A fresh socket, as a candidate reopening the tab would produce.
    with client.websocket_connect(_url(session)) as ws:
        _send(ws, {"type": "TAB_BLUR"})  # 2nd overall: over the limit

    severities = [e["severity"] for e in _report(client, session)["events"]]
    assert severities == ["INFO", "WARN"], (
        "escalation restarted on reconnect; counts were not primed from earlier events"
    )


def test_a_flood_is_rate_limited(client, session, monkeypatch):
    """A candidate must not be able to bury a real signal under synthetic ones,
    or use the socket to exhaust the database."""
    from app.modules.proctoring import collector

    monkeypatch.setattr(collector.settings, "proctor_events_per_minute", 5)

    with client.websocket_connect(_url(session)) as ws:
        for _ in range(40):
            _send(ws, {"type": "TAB_FOCUS"})

    stored = len(_report(client, session)["events"])
    assert stored <= 5, f"rate limit did not hold: {stored} events stored"


# --- Auto-termination --------------------------------------------------------
#
# Only the "off by default" half is exercised here. The enabled path closes the
# socket from the server side mid-exchange, and TestClient's close handshake
# deadlocks against that rather than raising -- a harness limitation, not an
# application one. The behaviour itself is covered in test_proctoring.py, at the
# layer where the decision is actually made.


def test_auto_termination_is_off_unless_a_recruiter_enables_it(client, session):
    with client.websocket_connect(_url(session)) as ws:
        for _ in range(12):  # well past any threshold
            _send(ws, {"type": "TAB_BLUR"})


    interview = client.get(
        f"/api/v1/interviews/{session['interview_id']}", headers=session["recruiter"]
    ).json()
    assert interview["status"] != "TERMINATED"
