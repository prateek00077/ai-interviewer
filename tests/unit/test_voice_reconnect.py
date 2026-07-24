"""A dropped call is a reconnect until proven otherwise.

THE BUG THIS PINS: a candidate whose connection blipped -- wifi, a tab reload --
was abandoned on the first drop, which terminates the interview. The rejoin a
second later then hit a terminal status and was locked out, defeating the
multi-use invite and the per-turn checkpoint that exist for exactly this case.

So a drop now starts a grace timer instead of abandoning outright. A reconnect
within the window cancels the timer and supersedes the old session (kept alive,
resumed from the checkpoint); only a window that lapses with nobody back
abandons the interview.
"""

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.modules.voice import session_manager as sm


@pytest.fixture
def fast_grace(monkeypatch):
    """A millisecond grace window and a recording stop(), so the timer logic is
    testable without a real session or a real wait."""
    monkeypatch.setattr(settings, "voice_reconnect_grace_secs", 0.05)
    calls: list = []

    async def _stop(interview_id, reason="completed"):
        calls.append((interview_id, reason))

    monkeypatch.setattr(sm, "stop", _stop)
    sm._pending_abandon.clear()
    yield calls
    for task in list(sm._pending_abandon.values()):
        task.cancel()
    sm._pending_abandon.clear()


class _FakeConn:
    """Records the handlers wired against it, like a SmallWebRTCConnection."""

    def __init__(self) -> None:
        self.handlers: dict = {}

    def add_event_handler(self, name: str, fn) -> None:
        self.handlers[name] = fn


def _session() -> sm.VoiceSession:
    return sm.VoiceSession(
        interview_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        candidate_id=uuid.uuid4(),
        connection=_FakeConn(),
        built=MagicMock(),
        runner=MagicMock(),
    )


# --- The timer itself -------------------------------------------------------


async def test_a_drop_with_no_reconnect_abandons_after_the_grace(fast_grace):
    interview_id = uuid.uuid4()
    sm._schedule_abandon(interview_id)
    await asyncio.sleep(0.12)
    assert fast_grace == [(interview_id, "abandoned")]


async def test_a_reconnect_within_the_grace_cancels_the_abandon(fast_grace):
    """THE FIX. The rejoin lands inside the window and calls off the abandon, so
    the interview never terminates and the new session can supersede the old."""
    interview_id = uuid.uuid4()
    sm._schedule_abandon(interview_id)
    sm._cancel_pending_abandon(interview_id)
    await asyncio.sleep(0.12)
    assert fast_grace == []
    assert interview_id not in sm._pending_abandon


async def test_one_drop_schedules_one_timer_despite_three_events(fast_grace):
    """pipecat fires disconnected, then closed, then failed for a single drop;
    they must not stack three abandonments."""
    interview_id = uuid.uuid4()
    sm._schedule_abandon(interview_id)
    first = sm._pending_abandon[interview_id]
    sm._schedule_abandon(interview_id)
    sm._schedule_abandon(interview_id)
    assert sm._pending_abandon[interview_id] is first
    await asyncio.sleep(0.12)
    assert fast_grace == [(interview_id, "abandoned")]


# --- The drop handler's guard -----------------------------------------------


async def test_a_live_session_drop_schedules_an_abandon(fast_grace):
    obj = _session()
    sm._wire_disconnect(obj)
    await obj.connection.handlers["disconnected"]()
    assert obj.interview_id in sm._pending_abandon


async def test_a_superseded_session_drop_does_not_schedule(fast_grace):
    """When a reconnect supersedes the old session, tearing it down closes its
    connection and fires this same handler. It must NOT abandon the interview the
    replacing session is now running."""
    obj = _session()
    obj._reason = sm.SUPERSEDED
    sm._wire_disconnect(obj)
    await obj.connection.handlers["closed"]()
    assert obj.interview_id not in sm._pending_abandon


async def test_an_already_stopped_session_drop_does_not_schedule(fast_grace):
    obj = _session()
    obj._stopped = True
    sm._wire_disconnect(obj)
    await obj.connection.handlers["failed"]()
    assert obj.interview_id not in sm._pending_abandon


# --- The invariant that keeps the grace from colliding with the idle nudge ---


def test_the_grace_is_shorter_than_the_idle_window():
    """A stale session's idle check-in must not fire into the dead connection
    during the wait, so the abandon has to resolve first."""
    assert settings.voice_reconnect_grace_secs < settings.voice_idle_nudge_secs
