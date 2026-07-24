"""Live-context assembly and the WebRTC entry point.

The property this file exists to protect is that the system prompt never leaves
the server. Everything the interviewer knows -- the plan, the rubric's
existence, the resume extraction -- is assembled here and handed to the model
in-process. If any of it reached the browser, a candidate could read the
questions before answering them.
"""

import uuid

import pytest

from app.models.interview import Interview, InterviewStatus
from app.models.job import Job, JobDescription
from app.modules.interview import service as interview_service
from app.modules.question_plan import service as plan_service
from app.modules.question_plan.generator import GeneratedPlan
from app.modules.voice import context as context_builder

pytestmark = pytest.mark.integration


GENERATED = GeneratedPlan.model_validate(
    {
        "criteria": [
            {
                "name": "kafka_depth",
                "weight": "0.5",
                "descriptors": {"1": "a", "3": "b", "5": "c"},
            },
            {"name": "comms", "weight": "0.3", "descriptors": {"1": "a", "3": "b", "5": "c"}},
            {"name": "owner", "weight": "0.2", "descriptors": {"1": "a", "3": "b", "5": "c"}},
        ],
        "questions": [
            {
                "body": "Walk me through the RabbitMQ to Kafka migration.",
                "competency": "kafka_depth",
                "follow_up_hints": ["press on exactly-once semantics"],
            },
            {"body": "How did you explain that to the team?", "competency": "comms"},
        ],
    }
)


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


@pytest.fixture
async def interview_with_plan(tenant_session, org_a):
    """An interview with a job, an active description, and a populated plan."""
    interview_id = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        job = Job(org_id=org_a.org_id, title="Staff Backend Engineer")
        s.add(job)
        await s.flush()
        s.add(
            JobDescription(
                org_id=org_a.org_id,
                job_id=job.id,
                content="Own the payments ledger and the Kafka event bus.",
                is_active=True,
            )
        )
        s.add(
            Interview(
                id=interview_id,
                org_id=org_a.org_id,
                candidate_id=org_a.candidate_id,
                job_id=job.id,
                status=InterviewStatus.INVITED,
            )
        )

    async with tenant_session(org_a.org_id, "system", None) as s:
        plan = await plan_service.ensure_plan(
            s, org_id=org_a.org_id, interview_id=interview_id
        )
        await plan_service.apply_generated(
            s, plan=plan, generated=GENERATED, model_name="test-model"
        )
    return interview_id


async def _build(tenant_session, org_a, interview_id):
    async with tenant_session(org_a.org_id, "system", None) as s:
        return await context_builder.build(s, interview_id)


# --- What the model is told --------------------------------------------------


async def test_the_plan_and_job_reach_the_prompt(tenant_session, org_a, interview_with_plan):
    built = await _build(tenant_session, org_a, interview_with_plan)
    text = "\n".join(m["content"] for m in built.messages)

    assert "Staff Backend Engineer" in text
    assert "payments ledger" in text
    assert "RabbitMQ to Kafka migration" in text
    assert built.question_count == 2


async def test_follow_up_hints_are_included(tenant_session, org_a, interview_with_plan):
    """The difference between a probe that lands and one the model improvises."""
    built = await _build(tenant_session, org_a, interview_with_plan)
    text = "\n".join(m["content"] for m in built.messages)
    assert "exactly-once semantics" in text


async def test_the_rubric_is_never_given_to_the_live_model(
    tenant_session, org_a, interview_with_plan
):
    """The interviewer's job is to elicit evidence. Telling it how answers are
    weighted invites steering the candidate toward a better score, which
    corrupts the evidence it is collecting."""
    built = await _build(tenant_session, org_a, interview_with_plan)
    text = "\n".join(m["content"] for m in built.messages)

    # No criterion names, no weights, no descriptors.
    assert "kafka_depth" not in text
    assert "0.5" not in text
    for descriptor in ("descriptors", "weight"):
        assert descriptor not in text.lower().replace("weighted", "")


async def test_an_interview_with_no_plan_still_produces_a_usable_prompt(
    tenant_session, org_a
):
    """Refusing to start would strand a candidate already on the call."""
    interview_id = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(
            Interview(
                id=interview_id, org_id=org_a.org_id, candidate_id=org_a.candidate_id
            )
        )

    built = await _build(tenant_session, org_a, interview_id)
    text = "\n".join(m["content"] for m in built.messages)

    assert built.question_count == 0
    assert "No plan was prepared" in text
    assert "(no job description was provided)" in text
    assert "(no resume was provided)" in text


async def test_the_prompt_records_the_plan_version(
    tenant_session, org_a, interview_with_plan
):
    """So a resumed session can tell whether the plan moved under it."""
    built = await _build(tenant_session, org_a, interview_with_plan)
    assert built.plan_version is not None


# --- The WebRTC entry point --------------------------------------------------


@pytest.fixture
async def candidate_token(api_client, registered_org):
    invite = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={"candidate_email": f"cand-{uuid.uuid4().hex[:8]}@example.com"},
    )
    redeemed = await api_client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": invite.json()["invite_token"]}
    )
    return {
        "headers": {"Authorization": f"Bearer {redeemed.json()['interview_token']}"},
        "interview_id": invite.json()["interview_id"],
    }


async def test_a_recruiter_token_cannot_open_a_voice_session(api_client, registered_org):
    response = await api_client.post(
        "/api/v1/webrtc/offer",
        headers=_auth(registered_org),
        json={"sdp": "v=0\r\n", "type": "offer"},
    )
    assert response.status_code == 403


async def test_an_unauthenticated_offer_is_rejected(api_client):
    response = await api_client.post(
        "/api/v1/webrtc/offer", json={"sdp": "v=0\r\n", "type": "offer"}
    )
    assert response.status_code == 401


async def test_an_oversized_sdp_is_rejected_before_parsing(api_client, candidate_token):
    """The endpoint hands SDP to a parser; an unbounded field is an invitation."""
    response = await api_client.post(
        "/api/v1/webrtc/offer",
        headers=candidate_token["headers"],
        json={"sdp": "v=0\r\n" + "a" * 70_000, "type": "offer"},
    )
    assert response.status_code == 422


async def test_the_offer_carries_no_interview_identifier(api_client, candidate_token):
    """The interview comes from the token's signed claim, so there is no field
    for a candidate to point at someone else's session."""
    from app.schemas.webrtc import WebRTCOffer

    assert set(WebRTCOffer.model_fields) == {"sdp", "type", "pc_id"}


async def test_a_terminated_interview_refuses_a_new_session(
    api_client, registered_org, candidate_token
):
    await api_client.post(
        f"/api/v1/interviews/{candidate_token['interview_id']}/terminate",
        headers=_auth(registered_org),
        json={},
    )
    response = await api_client.post(
        "/api/v1/webrtc/offer",
        headers=candidate_token["headers"],
        json={"sdp": "v=0\r\n", "type": "offer"},
    )
    assert response.status_code == 409


# --- Session bookkeeping -----------------------------------------------------


async def test_stopping_an_unknown_session_is_a_no_op():
    """Called from a disconnect handler, a watchdog and an explicit end -- any
    of which can arrive twice or for an interview that never had a session."""
    from app.modules.voice import session_manager

    assert await session_manager.stop(uuid.uuid4()) is False
    assert session_manager.active_count() == 0


async def test_interview_start_is_driven_by_the_event_not_the_route(
    tenant_session, org_a, interview_with_plan
):
    """The voice module never writes an interview status; it announces, and
    interview/service decides. This pins that the decision path works."""
    async with tenant_session(org_a.org_id, "system", None) as s:
        interview = await interview_service.start(s, interview_with_plan)
        assert interview.status is InterviewStatus.IN_PROGRESS

    # And starting froze the plan.
    async with tenant_session(org_a.org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_with_plan)
        assert plan.status.value == "FROZEN"


async def test_a_dropped_connection_ends_the_session(monkeypatch):
    """Without this wiring, a candidate who closes their browser leaves the
    session running until the 45-minute watchdog -- holding an ASR stream open,
    the interview stuck IN_PROGRESS, and the recording never written.

    A drop no longer ends the session on the spot: it starts a grace timer, and
    only a window that lapses with no reconnect abandons the interview (the
    reconnect path itself is pinned in tests/unit/test_voice_reconnect.py). What
    this test guards is that the real connection's drop events are wired at all,
    so a pipecat rename surfaces here rather than as a session that never ends --
    and that the three events pipecat fires for one drop collapse to a single
    abandonment, not three.
    """
    import asyncio

    from app.core.config import settings
    from app.modules.voice import session_manager, transport

    monkeypatch.setattr(settings, "voice_reconnect_grace_secs", 0.05)
    session_manager._pending_abandon.clear()

    connection = transport.create_connection()
    session = session_manager.VoiceSession(
        interview_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        candidate_id=uuid.uuid4(),
        connection=connection,
        built=None,  # type: ignore[arg-type]
        runner=None,  # type: ignore[arg-type]
    )

    stopped: list[tuple] = []

    async def _fake_stop(interview_id, *, reason="completed"):
        stopped.append((interview_id, reason))
        return True

    monkeypatch.setattr(session_manager, "stop", _fake_stop)
    session_manager._wire_disconnect(session)

    # Called the way pipecat calls it: no extra args, the emitting object is
    # supplied by the dispatcher. All three fire for one drop.
    for event in ("disconnected", "closed", "failed"):
        await connection._call_event_handler(event)

    # Before the grace window: nothing abandoned yet -- the candidate might be
    # reconnecting.
    await asyncio.sleep(0.01)
    assert stopped == [], "a drop abandoned the interview before the grace window"

    # After it lapses with nobody back: exactly one abandonment, not three.
    await asyncio.sleep(0.1)
    assert len(stopped) == 1, f"one drop must abandon once, got: {stopped}"
    assert stopped[0][1] == "abandoned", (
        "a dropped call must be distinguishable from one that finished"
    )
