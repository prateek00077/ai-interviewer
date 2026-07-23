"""What the interviewer does when nobody says anything.

Silence in a voice interview is ambiguous: the candidate may be thinking, or
their microphone may have died two minutes ago while they wait for a question
that already came. A human interviewer resolves that by asking. Sitting mute
forever is the worst option; hanging up without a word is the second worst.

Pipecat's own default is to CANCEL the pipeline after 300s idle, which is wrong
on both counts -- so these also pin that we turned that off.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.modules.voice import session_manager as sm


class _FakeTask:
    """Records queued frames and the handlers registered against it."""

    def __init__(self) -> None:
        self.frames: list = []
        self.handlers: dict = {}

    def event_handler(self, name: str):
        def register(fn):
            self.handlers[name] = fn
            return fn

        return register

    async def queue_frames(self, frames) -> None:
        self.frames.extend(frames)


@pytest.fixture
def session():
    task = _FakeTask()
    obj = sm.VoiceSession(
        interview_id=uuid.uuid4(),
        org_id=uuid.uuid4(),
        candidate_id=uuid.uuid4(),
        connection=MagicMock(),
        built=MagicMock(task=task),
        runner=MagicMock(),
    )
    sm._handle_silence(obj)
    return obj, task


@pytest.fixture
def no_teardown(monkeypatch):
    """Stop the give-up path from touching the real session registry."""
    ended: list = []

    async def _stop(interview_id, reason="completed"):
        ended.append((interview_id, reason))

    monkeypatch.setattr(sm, "stop", _stop)
    monkeypatch.setattr(sm, "_FAREWELL_GRACE_SECS", 0)
    return ended


async def test_silence_is_answered_rather_than_ignored(session):
    obj, task = session
    await task.handlers["on_idle_timeout"](None)
    assert len(task.frames) == 1
    assert task.frames[0].text == sm.IDLE_NUDGES[0]


async def test_the_check_ins_escalate_rather_than_repeat(session):
    """Saying the identical line twice is what makes a bot sound like a bot."""
    obj, task = session
    for _ in range(2):
        await task.handlers["on_idle_timeout"](None)

    spoken = [f.text for f in task.frames]
    assert spoken == list(sm.IDLE_NUDGES[:2])
    assert spoken[0] != spoken[1]


async def test_the_first_check_in_reassures_before_it_questions(session):
    """A candidate who is thinking hard should not be accused of vanishing on
    the first pause. The explicit "are you there" comes second."""
    assert "take your time" in sm.IDLE_NUDGES[0].lower()
    assert "there" in sm.IDLE_NUDGES[1].lower()


async def test_it_gives_up_eventually_and_says_why(session, no_teardown):
    obj, task = session
    for _ in range(settings.voice_max_idle_nudges + 1):
        await task.handlers["on_idle_timeout"](None)

    assert task.frames[-1].text == sm.IDLE_FAREWELL
    assert no_teardown == [(obj.interview_id, "abandoned")]


async def test_the_farewell_is_spoken_before_the_call_drops(session, monkeypatch):
    """Hanging up mid-sentence would undo the point of saying it."""
    obj, task = session
    order: list[str] = []

    async def _stop(interview_id, reason="completed"):
        order.append("stopped")

    async def _sleep(_secs):
        order.append("waited")

    monkeypatch.setattr(sm, "stop", _stop)
    monkeypatch.setattr(sm.asyncio, "sleep", _sleep)

    for _ in range(settings.voice_max_idle_nudges + 1):
        await task.handlers["on_idle_timeout"](None)

    assert order == ["waited", "stopped"], "the call dropped before the farewell played"


async def test_a_stopped_session_says_nothing(session, no_teardown):
    """The event can fire while teardown is already underway."""
    obj, task = session
    obj._stopped = True
    await task.handlers["on_idle_timeout"](None)
    assert task.frames == []


def test_the_pipeline_does_not_cancel_itself_on_idle():
    """Pipecat cancels on idle by default. Five minutes of a candidate thinking
    is not abandonment, and killing the call is not the right answer to silence.
    """
    import inspect

    from app.modules.voice import pipeline

    source = inspect.getsource(pipeline.build)
    assert "cancel_on_idle_timeout=False" in source
    assert "cancel_runner_on_idle_timeout=False" in source


def test_the_idle_window_is_shorter_than_the_session_cap():
    """A nudge that fires after the watchdog has already ended the session is
    no nudge at all."""
    assert settings.voice_idle_nudge_secs < settings.max_interview_minutes * 60
