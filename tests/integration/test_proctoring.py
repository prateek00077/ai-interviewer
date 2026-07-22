"""Proctoring policy, report, and frame upload over HTTP.

The WebSocket lives in test_proctoring_socket.py, which needs a synchronous
client and therefore its own event loop.
"""

import uuid

import pytest

from app.db.session import tenant_session
from app.models.interview import InterviewStatus
from app.models.proctoring import ProctorEventType, ProctoringEvent, ProctorSeverity
from app.modules.proctoring import verdict as verdict_module

pytestmark = pytest.mark.integration

T = ProctorEventType
S = ProctorSeverity


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


@pytest.fixture
async def invited(api_client, registered_org):
    """A job, an interview against it, and a redeemed candidate token."""
    job = await api_client.post(
        "/api/v1/jobs", headers=_auth(registered_org), json={"title": "Staff Engineer"}
    )
    job_id = job.json()["id"]

    invite = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={
            "candidate_email": f"cand-{uuid.uuid4().hex[:8]}@example.com",
            "job_id": job_id,
        },
    )
    body = invite.json()
    redeemed = await api_client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": body["invite_token"]}
    )
    return {
        **body,
        "job_id": job_id,
        "org_id": registered_org["org_id"],
        "headers": {"Authorization": f"Bearer {redeemed.json()['interview_token']}"},
    }


# --- Policy ------------------------------------------------------------------


async def test_a_recruiter_can_set_and_read_a_policy(api_client, registered_org, invited):
    url = f"/api/v1/jobs/{invited['job_id']}/proctoring-policy"
    assert (await api_client.get(url, headers=_auth(registered_org))).status_code == 404

    created = await api_client.put(
        url,
        headers=_auth(registered_org),
        json={"blur_limit": 1, "fullscreen_required": True, "auto_terminate": False},
    )
    assert created.status_code == 200, created.text
    assert created.json()["blur_limit"] == 1

    fetched = await api_client.get(url, headers=_auth(registered_org))
    assert fetched.json()["fullscreen_required"] is True


async def test_the_policy_defaults_keep_auto_termination_off(api_client, registered_org, invited):
    """Ending a real person's interview on a heuristic is a human's decision."""
    created = await api_client.put(
        f"/api/v1/jobs/{invited['job_id']}/proctoring-policy",
        headers=_auth(registered_org),
        json={},
    )
    assert created.json()["auto_terminate"] is False


async def test_a_candidate_cannot_read_the_policy(api_client, invited):
    """Knowing the blur limit is knowing exactly how much you can get away with."""
    response = await api_client.get(
        f"/api/v1/jobs/{invited['job_id']}/proctoring-policy", headers=invited["headers"]
    )
    assert response.status_code == 403


async def test_another_org_cannot_touch_the_policy(api_client, registered_org, invited):
    slug = f"rival-{uuid.uuid4().hex[:10]}"
    rival = (
        await api_client.post(
            "/api/v1/auth/register-org",
            json={
                "org_name": "Rival",
                "slug": slug,
                "admin_email": f"admin@{slug}.example.com",
                "admin_password": "correct-horse-battery-staple",
            },
        )
    ).json()
    url = f"/api/v1/jobs/{invited['job_id']}/proctoring-policy"
    assert (await api_client.get(url, headers=_auth(rival))).status_code == 404
    assert (await api_client.put(url, headers=_auth(rival), json={})).status_code == 404


# --- Report ------------------------------------------------------------------


async def test_a_candidate_cannot_read_their_own_proctoring_report(api_client, invited):
    response = await api_client.get(
        f"/api/v1/interviews/{invited['interview_id']}/proctoring",
        headers=invited["headers"],
    )
    assert response.status_code == 403


async def test_the_report_carries_the_verdict_and_its_events(
    api_client, registered_org, invited
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    async with tenant_session(org_id, "system", None) as s:
        for _ in range(2):
            s.add(
                ProctoringEvent(
                    org_id=org_id,
                    interview_id=interview_id,
                    event_type=T.TAB_BLUR,
                    severity=S.INFO,
                )
            )
    async with tenant_session(org_id, "system", None) as s:
        await verdict_module.finalise(s, org_id=org_id, interview_id=interview_id)

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/proctoring", headers=_auth(registered_org)
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["events"]) == 2
    assert body["verdict"]["verdict"] == "CLEAN"
    assert body["verdict"]["reasons"], "a verdict was returned without its reasons"


async def test_the_verdict_is_recomputed_not_accumulated(registered_org, invited):
    """Re-running after a rule change must give the answer the current rules
    imply, not a fossil of the rules that were live during the interview."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    async with tenant_session(org_id, "system", None) as s:
        s.add(
            ProctoringEvent(
                org_id=org_id,
                interview_id=interview_id,
                event_type=T.SECOND_SPEAKER,
                severity=S.CRITICAL,
            )
        )

    async with tenant_session(org_id, "system", None) as s:
        first = await verdict_module.finalise(s, org_id=org_id, interview_id=interview_id)
        assert first.verdict.value == "FLAGGED"
        first_id = first.id

    async with tenant_session(org_id, "system", None) as s:
        second = await verdict_module.finalise(s, org_id=org_id, interview_id=interview_id)
        assert second.id == first_id, "a re-run created a second verdict row"
        assert second.verdict.value == "FLAGGED"


async def test_an_interview_with_no_events_is_no_data(registered_org, invited):
    org_id = uuid.UUID(registered_org["org_id"])
    async with tenant_session(org_id, "system", None) as s:
        stored = await verdict_module.finalise(
            s, org_id=org_id, interview_id=uuid.UUID(invited["interview_id"])
        )
    assert stored.verdict.value == "NO_DATA"
    assert stored.reasons


# --- Frame upload ------------------------------------------------------------


async def test_a_candidate_gets_a_server_chosen_frame_key(api_client, invited):
    """A client that could name the key could overwrite another interview's
    evidence."""
    response = await api_client.post(
        "/api/v1/proctoring/frames/presign",
        headers=invited["headers"],
        json={"content_type": "image/jpeg"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["s3_key"].startswith(f"{invited['org_id']}/{invited['interview_id']}/")
    assert body["upload_url"].startswith("http")


async def test_a_recruiter_cannot_use_the_candidate_frame_route(
    api_client, registered_org, invited
):
    response = await api_client.post(
        "/api/v1/proctoring/frames/presign",
        headers=_auth(registered_org),
        json={"content_type": "image/jpeg"},
    )
    assert response.status_code == 403


async def test_a_non_image_content_type_is_refused(api_client, invited):
    response = await api_client.post(
        "/api/v1/proctoring/frames/presign",
        headers=invited["headers"],
        json={"content_type": "application/pdf"},
    )
    assert response.status_code == 422


# --- Auto-termination --------------------------------------------------------


async def test_a_critical_event_terminates_when_the_policy_says_so(registered_org, invited):
    """The socket's termination path, at the layer that decides it.

    Driven here rather than through the WebSocket because the server closes the
    socket as it terminates, and TestClient's close handshake deadlocks against
    that. What matters is that a CRITICAL event under an auto-terminate policy
    ends the interview THROUGH THE STATE MACHINE -- not that the socket shuts
    in a particular order.
    """
    from app.modules.interview import service as interview_service
    from app.modules.proctoring import collector, rules

    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    thresholds = rules.Thresholds(blur_limit=1, auto_terminate=True)
    counters = collector.SessionCounters()

    async with tenant_session(org_id, "system", None) as s:
        terminated = False
        for _ in range(4):
            event = await collector.record(
                s,
                org_id=org_id,
                interview_id=interview_id,
                event_type=T.TAB_BLUR,
                thresholds=thresholds,
                counters=counters,
            )
            if rules.should_terminate(event.severity, thresholds):
                await interview_service.terminate(s, interview_id, reason="proctoring")
                terminated = True
                break

    assert terminated, "repeated blurs never reached CRITICAL under a strict policy"

    async with tenant_session(org_id, "system", None) as s:
        interview = await interview_service.get_interview(s, interview_id)
        assert interview.status is InterviewStatus.TERMINATED
        assert interview.completed_at is not None


async def test_auto_termination_stays_off_under_the_default_policy(registered_org, invited):
    from app.modules.interview import service as interview_service
    from app.modules.proctoring import collector, rules

    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    thresholds = rules.Thresholds()  # org defaults
    counters = collector.SessionCounters()

    async with tenant_session(org_id, "system", None) as s:
        for _ in range(12):  # well past every threshold
            event = await collector.record(
                s,
                org_id=org_id,
                interview_id=interview_id,
                event_type=T.TAB_BLUR,
                thresholds=thresholds,
                counters=counters,
            )
            assert not rules.should_terminate(event.severity, thresholds)

    async with tenant_session(org_id, "system", None) as s:
        interview = await interview_service.get_interview(s, interview_id)
        assert interview.status is not InterviewStatus.TERMINATED
