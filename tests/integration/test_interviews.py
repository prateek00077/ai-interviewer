"""Interview lifecycle over HTTP, and the bus wiring behind it.

The two properties that matter here are that a candidate can only ever reach
their own interview, and that the transitions the voice pipeline drives -- via
events, not endpoints -- land correctly in the database.
"""

import uuid

import pytest

from app.core.events import EventBus, SessionEnded, SessionStarted, TurnCompleted
from app.db.session import tenant_session
from app.models.interview import InterviewStatus
from app.modules.interview import service as interview_service
from app.modules.interview import transcript
from app.modules.question_plan import service as plan_service
from app.modules.question_plan.generator import GeneratedPlan

pytestmark = pytest.mark.integration


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


GENERATED = GeneratedPlan.model_validate(
    {
        "criteria": [
            {"name": "depth", "weight": "0.5", "descriptors": {"1": "a", "3": "b", "5": "c"}},
            {"name": "comms", "weight": "0.3", "descriptors": {"1": "a", "3": "b", "5": "c"}},
            {"name": "owner", "weight": "0.2", "descriptors": {"1": "a", "3": "b", "5": "c"}},
        ],
        "questions": [{"body": "Walk me through the migration.", "competency": "depth"}],
    }
)


@pytest.fixture
async def invited(api_client, registered_org):
    """An INVITED interview plus a redeemed candidate token."""
    invite = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={"candidate_email": f"cand-{uuid.uuid4().hex[:8]}@example.com"},
    )
    assert invite.status_code == 201, invite.text
    body = invite.json()

    redeemed = await api_client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": body["invite_token"]}
    )
    assert redeemed.status_code == 200, redeemed.text
    return {
        **body,
        "candidate_headers": {
            "Authorization": f"Bearer {redeemed.json()['interview_token']}"
        },
    }


# --- Redemption no longer starts the interview ------------------------------


async def test_redeeming_an_invite_does_not_mark_the_interview_live(
    api_client, registered_org, invited
):
    """Intent to join is not joining. A candidate can redeem and never connect,
    and "redeemed but never attended" must stay distinguishable from an
    interview that actually ran."""
    response = await api_client.get(
        f"/api/v1/interviews/{invited['interview_id']}", headers=_auth(registered_org)
    )
    assert response.json()["status"] == "INVITED"
    assert response.json()["started_at"] is None


async def test_an_invite_for_a_terminated_interview_is_unusable(
    api_client, registered_org, invited
):
    """The link outlived the interview. Same opaque 410 as every other case."""
    await api_client.post(
        f"/api/v1/interviews/{invited['interview_id']}/terminate",
        headers=_auth(registered_org),
        json={"reason": "withdrew"},
    )
    response = await api_client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": invited["invite_token"]}
    )
    assert response.status_code == 410


# --- The candidate's own view -----------------------------------------------


async def test_a_candidate_reads_their_own_interview(api_client, invited):
    response = await api_client.get(
        "/api/v1/interviews/me", headers=invited["candidate_headers"]
    )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == invited["interview_id"]
    # The narrow view: no org, no job, no candidate id, nothing about scoring.
    assert set(body) == {"id", "status", "scheduled_at", "started_at"}


async def test_a_candidate_cannot_reach_the_recruiter_interview_routes(api_client, invited):
    headers = invited["candidate_headers"]
    interview_id = invited["interview_id"]

    assert (await api_client.get("/api/v1/interviews", headers=headers)).status_code == 403
    assert (
        await api_client.get(f"/api/v1/interviews/{interview_id}", headers=headers)
    ).status_code == 403
    assert (
        await api_client.get(f"/api/v1/interviews/{interview_id}/transcript", headers=headers)
    ).status_code == 403
    assert (
        await api_client.post(
            f"/api/v1/interviews/{interview_id}/terminate", headers=headers, json={}
        )
    ).status_code == 403


async def test_a_recruiter_token_cannot_use_the_candidate_route(api_client, registered_org):
    response = await api_client.get("/api/v1/interviews/me", headers=_auth(registered_org))
    assert response.status_code == 403


# --- Recruiter operations ---------------------------------------------------


async def test_scheduling_an_interview_for_another_orgs_candidate_is_a_404(
    api_client, registered_org
):
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
    victim = await api_client.post(
        "/api/v1/candidates",
        headers=_auth(registered_org),
        json={"email": f"victim-{uuid.uuid4().hex[:8]}@example.com"},
    )

    response = await api_client.post(
        "/api/v1/interviews",
        headers=_auth(rival),
        json={"candidate_id": victim.json()["id"]},
    )
    assert response.status_code == 404


async def test_terminating_twice_is_a_conflict(api_client, registered_org, invited):
    """Terminal states are absorbing, and the recruiter should learn their
    second click did nothing."""
    url = f"/api/v1/interviews/{invited['interview_id']}/terminate"
    first = await api_client.post(url, headers=_auth(registered_org), json={})
    assert first.status_code == 200
    assert first.json()["status"] == "TERMINATED"
    assert first.json()["completed_at"] is not None

    second = await api_client.post(url, headers=_auth(registered_org), json={})
    assert second.status_code == 409


async def test_interviews_are_filterable_by_status(api_client, registered_org, invited):
    response = await api_client.get(
        "/api/v1/interviews", headers=_auth(registered_org), params={"status": "INVITED"}
    )
    assert response.json()["total"] >= 1
    assert all(i["status"] == "INVITED" for i in response.json()["items"])


async def test_another_org_sees_no_interviews(api_client, registered_org, invited):
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
    response = await api_client.get("/api/v1/interviews", headers=_auth(rival))
    assert response.json()["total"] == 0


# --- Session lifecycle, driven by events ------------------------------------


@pytest.fixture
def local_bus(monkeypatch):
    """A private bus with the real handlers, so tests do not cross-talk."""
    bus = EventBus()
    monkeypatch.setattr("app.core.events.bus", bus)
    monkeypatch.setattr("app.core.events.publish", bus.publish)
    monkeypatch.setattr("app.core.events.subscribe", bus.subscribe)

    bus.subscribe(SessionStarted, interview_service._on_session_started)
    bus.subscribe(SessionEnded, interview_service._on_session_ended)
    bus.subscribe(TurnCompleted, transcript._on_turn_completed)
    return bus


async def test_a_session_started_event_marks_the_interview_live(
    api_client, registered_org, invited, local_bus
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    local_bus.publish(SessionStarted(org_id=org_id, interview_id=interview_id))
    await local_bus.drain()

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}", headers=_auth(registered_org)
    )
    assert response.json()["status"] == "IN_PROGRESS"
    assert response.json()["started_at"] is not None


async def test_starting_an_interview_freezes_its_plan(
    api_client, registered_org, invited, local_bus
):
    """The plan must be immutable from the first question asked -- not from
    approval, which a recruiter can still revise."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    async with tenant_session(org_id, "system", None) as s:
        plan = await plan_service.ensure_plan(s, org_id=org_id, interview_id=interview_id)
        await plan_service.apply_generated(
            s, plan=plan, generated=GENERATED, model_name="test"
        )

    local_bus.publish(SessionStarted(org_id=org_id, interview_id=interview_id))
    await local_bus.drain()

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/plan", headers=_auth(registered_org)
    )
    assert response.json()["status"] == "FROZEN"


async def test_an_interview_with_no_plan_still_starts(
    api_client, registered_org, invited, local_bus
):
    """Refusing would strand a candidate already on the call."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    local_bus.publish(SessionStarted(org_id=org_id, interview_id=interview_id))
    await local_bus.drain()

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}", headers=_auth(registered_org)
    )
    assert response.json()["status"] == "IN_PROGRESS"


@pytest.mark.parametrize(
    "reason,expected",
    [
        ("completed", "COMPLETED"),
        ("abandoned", "ABANDONED"),
        ("terminated", "TERMINATED"),
        ("timed_out", "COMPLETED"),
        ("something nobody defined", "ABANDONED"),
    ],
)
async def test_session_end_reasons_map_to_statuses(
    api_client, registered_org, invited, local_bus, reason, expected
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    local_bus.publish(SessionStarted(org_id=org_id, interview_id=interview_id))
    await local_bus.drain()
    local_bus.publish(
        SessionEnded(org_id=org_id, interview_id=interview_id, reason=reason)
    )
    await local_bus.drain()

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}", headers=_auth(registered_org)
    )
    assert response.json()["status"] == expected


async def test_a_duplicate_session_end_is_not_an_error(
    api_client, registered_org, invited, local_bus
):
    """The bus is fire-and-forget; a second close event must not raise into a
    handler nobody is awaiting."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    local_bus.publish(SessionStarted(org_id=org_id, interview_id=interview_id))
    await local_bus.drain()
    for _ in range(2):
        local_bus.publish(
            SessionEnded(org_id=org_id, interview_id=interview_id, reason="completed")
        )
    await local_bus.drain()

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}", headers=_auth(registered_org)
    )
    assert response.json()["status"] == "COMPLETED"


# --- Transcript -------------------------------------------------------------


async def test_turn_events_become_transcript_rows(
    api_client, registered_org, invited, local_bus
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    for ordinal, (speaker, content) in enumerate(
        [
            ("interviewer", "Tell me about the migration."),
            ("candidate", "We moved from RabbitMQ to Kafka."),
        ]
    ):
        local_bus.publish(
            TurnCompleted(
                org_id=org_id,
                interview_id=interview_id,
                ordinal=ordinal,
                speaker=speaker,
                content=content,
                started_offset_ms=ordinal * 1000,
                ended_offset_ms=ordinal * 1000 + 800,
            )
        )
    await local_bus.drain()

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/transcript", headers=_auth(registered_org)
    )
    assert response.status_code == 200
    turns = response.json()["turns"]
    assert [t["speaker"] for t in turns] == ["INTERVIEWER", "CANDIDATE"]
    assert turns[1]["content"] == "We moved from RabbitMQ to Kafka."
    assert turns[0]["is_final"] is False


async def test_a_replayed_turn_updates_rather_than_duplicating(
    api_client, registered_org, invited, local_bus
):
    """A reconnecting session replays its last turn. The unique constraint would
    otherwise raise into a handler nobody awaits."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    for content in ("first attempt", "corrected text"):
        local_bus.publish(
            TurnCompleted(
                org_id=org_id,
                interview_id=interview_id,
                ordinal=0,
                speaker="candidate",
                content=content,
            )
        )
        await local_bus.drain()

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/transcript", headers=_auth(registered_org)
    )
    turns = response.json()["turns"]
    assert len(turns) == 1, "a replayed turn duplicated the transcript line"
    assert turns[0]["content"] == "corrected text"


async def test_an_offline_corrected_turn_is_not_overwritten_by_a_late_replay(
    registered_org, invited, local_bus
):
    """The offline pass produces the better text. A stale live-ASR event
    arriving afterwards must not undo it."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    local_bus.publish(
        TurnCompleted(
            org_id=org_id, interview_id=interview_id, ordinal=0,
            speaker="candidate", content="live asr text",
        )
    )
    await local_bus.drain()

    async with tenant_session(org_id, "system", None) as s:
        turns = await transcript.list_turns(s, interview_id)
        turns[0].content = "offline corrected text"
        turns[0].is_final = True

    local_bus.publish(
        TurnCompleted(
            org_id=org_id, interview_id=interview_id, ordinal=0,
            speaker="candidate", content="stale replay",
        )
    )
    await local_bus.drain()

    async with tenant_session(org_id, "system", None) as s:
        turns = await transcript.list_turns(s, interview_id)
    assert turns[0].content == "offline corrected text"


async def test_next_ordinal_resumes_where_the_transcript_left_off(
    registered_org, invited, local_bus
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    async with tenant_session(org_id, "system", None) as s:
        assert await transcript.next_ordinal(s, interview_id) == 0

    for ordinal in range(3):
        local_bus.publish(
            TurnCompleted(
                org_id=org_id, interview_id=interview_id, ordinal=ordinal,
                speaker="candidate", content=f"turn {ordinal}",
            )
        )
    await local_bus.drain()

    async with tenant_session(org_id, "system", None) as s:
        assert await transcript.next_ordinal(s, interview_id) == 3


async def test_an_unknown_speaker_is_kept_rather_than_dropped(
    registered_org, invited, local_bus
):
    """Losing a line of transcript is worse than mislabelling one, and a
    mislabelled line is visible in review."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    local_bus.publish(
        TurnCompleted(
            org_id=org_id, interview_id=interview_id, ordinal=0,
            speaker="narrator", content="something was said",
        )
    )
    await local_bus.drain()

    async with tenant_session(org_id, "system", None) as s:
        turns = await transcript.list_turns(s, interview_id)
    assert len(turns) == 1
    assert turns[0].content == "something was said"


# --- Expiry -----------------------------------------------------------------


async def test_the_reaper_expires_only_unstarted_interviews(
    registered_org, invited, local_bus
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    # Nothing is old enough yet.
    async with tenant_session(org_id, "system", None) as s:
        assert await interview_service.expire_stale(s, older_than_hours=72) == 0

    async with tenant_session(org_id, "system", None) as s:
        assert await interview_service.expire_stale(s, older_than_hours=0) == 1
        interview = await interview_service.get_interview(s, interview_id)
        assert interview.status is InterviewStatus.EXPIRED
        assert interview.completed_at is not None


async def test_the_reaper_leaves_live_interviews_alone(registered_org, invited, local_bus):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    local_bus.publish(SessionStarted(org_id=org_id, interview_id=interview_id))
    await local_bus.drain()

    async with tenant_session(org_id, "system", None) as s:
        assert await interview_service.expire_stale(s, older_than_hours=0) == 0
        interview = await interview_service.get_interview(s, interview_id)
        assert interview.status is InterviewStatus.IN_PROGRESS
