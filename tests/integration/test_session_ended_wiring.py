"""What happens on the bus when a voice session ends.

Three things have to happen and none of them is the voice module's decision: the
interview reaches a terminal status, the recording key is captured, and the
post-interview chain is enqueued. Before this slice the key was announced and
thrown away, which left the transcript pass and the delivery signals with
nothing to read.
"""

import uuid

import pytest

from app.core import events
from app.core.events import SessionEnded, SessionStarted
from app.db.session import tenant_session
from app.models.interview import InterviewStatus
from app.modules.interview import service as interview_service
from app.modules.interview import state_machine

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _subscribed():
    """The bus is wired by the app lifespan, which these tests do not start.

    Cleared first: subscriptions are global, so registering per test without
    clearing would stack handlers and each event would be handled N times --
    which passes for a while and then produces baffling duplicate writes.
    """
    events.bus.clear()
    interview_service.register()
    yield
    events.bus.clear()


@pytest.fixture
def captured(monkeypatch) -> list[tuple]:
    """Intercept the enqueue so no broker is involved."""
    from app.workers import pipeline

    calls: list[tuple] = []
    monkeypatch.setattr(pipeline, "enqueue", lambda org, interview: calls.append((org, interview)))
    return calls


@pytest.fixture
async def live_interview(tenant_session, org_a):
    """An interview in IN_PROGRESS, as one is when a session is running."""
    async with tenant_session(org_a.org_id, "system", None) as session:
        interview = await interview_service.create_interview(
            session, org_id=org_a.org_id, candidate_id=org_a.candidate_id
        )
        interview_id = interview.id
        # Via INVITED, because that is the only edge into IN_PROGRESS: an
        # interview nobody was invited to cannot have a session.
        state_machine.transition(interview, InterviewStatus.INVITED)

    async with tenant_session(org_a.org_id, "system", None) as session:
        await interview_service.start(session, interview_id)
    return interview_id


async def _end(org_id: uuid.UUID, interview_id: uuid.UUID, **kwargs) -> None:
    events.publish(SessionEnded(org_id=org_id, interview_id=interview_id, **kwargs))
    # Fire-and-forget: the handler is a task, not an await.
    await events.bus.drain()


async def test_the_recording_key_is_captured(tenant_session, org_a, live_interview, captured):
    await _end(org_a.org_id, live_interview, reason="completed", recording_key="org/int/abc.wav")

    async with tenant_session(org_a.org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, live_interview)
    assert interview.recording_key == "org/int/abc.wav"
    assert interview.status is InterviewStatus.COMPLETED


async def test_the_post_interview_chain_is_enqueued(org_a, live_interview, captured):
    await _end(org_a.org_id, live_interview, reason="completed", recording_key="k.wav")
    assert captured == [(org_a.org_id, live_interview)]


async def test_a_session_with_no_recording_still_enqueues(
    tenant_session, org_a, live_interview, captured
):
    """A candidate who joined and never spoke produces no recording. The
    pipeline still runs and reports INSUFFICIENT_EVIDENCE rather than silently
    never producing a score row at all."""
    await _end(org_a.org_id, live_interview, reason="abandoned")

    async with tenant_session(org_a.org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, live_interview)
    assert interview.recording_key is None
    assert interview.status is InterviewStatus.ABANDONED
    assert captured == [(org_a.org_id, live_interview)]


async def test_a_superseded_session_does_not_enqueue_anything(
    tenant_session, org_a, live_interview, captured
):
    """A reconnecting candidate supersedes their own session. The interview is
    not over, and scoring a conversation still in progress would produce a
    number about half of it.

    The voice module does not publish this reason at all -- but the two sides
    communicate by string, and this is the exact mismatch that once locked a
    reconnecting candidate out of their own interview.
    """
    await _end(org_a.org_id, live_interview, reason="superseded")

    async with tenant_session(org_a.org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, live_interview)
    assert interview.status is InterviewStatus.IN_PROGRESS
    assert captured == [], "a reconnect kicked off the post-interview chain"


async def test_a_reconnect_after_a_supersede_still_works(org_a, live_interview, captured):
    """The regression that motivates the previous test: the replacement session
    must be able to attach to an interview that is still live."""
    await _end(org_a.org_id, live_interview, reason="superseded")

    events.publish(SessionStarted(org_id=org_a.org_id, interview_id=live_interview))
    await events.bus.drain()

    async with tenant_session(org_a.org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, live_interview)
    assert interview.status is InterviewStatus.IN_PROGRESS


async def test_a_duplicate_end_event_does_not_enqueue_twice(org_a, live_interview, captured):
    """The bus is at-least-once in spirit. A second close arriving after a clean
    one must not run the whole pipeline again -- the tasks are idempotent, so a
    double run would be survivable rather than wrong, but it is minutes of GPU
    time either way."""
    await _end(org_a.org_id, live_interview, reason="completed", recording_key="k.wav")
    await _end(org_a.org_id, live_interview, reason="completed", recording_key="k.wav")

    assert captured == [(org_a.org_id, live_interview)]
    async with tenant_session(org_a.org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, live_interview)
    assert interview.status is InterviewStatus.COMPLETED
