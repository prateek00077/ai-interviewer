"""The interview state machine and the event bus.

Both are small, both are load-bearing, and both fail in ways that are invisible
at the call site: an illegal transition that silently succeeds lets a candidate
rejoin a scored interview, and a bus that swallows a handler reference loses a
transcript turn with no error anywhere.
"""

import asyncio
import uuid

import pytest

from app.core.events import (
    Event,
    EventBus,
    SessionEnded,
    TurnCompleted,
)
from app.models.interview import Interview, InterviewStatus
from app.modules.interview import state_machine
from app.modules.interview.state_machine import (
    IllegalTransitionError,
    can_transition,
    is_terminal,
    transition,
)

S = InterviewStatus


def _interview(status: S = S.CREATED) -> Interview:
    return Interview(id=uuid.uuid4(), org_id=uuid.uuid4(), candidate_id=uuid.uuid4(), status=status)


# --- The transition table ---------------------------------------------------


def test_every_status_appears_in_the_table():
    """A status added to the enum without an entry here would KeyError at the
    first transition attempt, in production."""
    assert set(state_machine.LEGAL) == set(S)


@pytest.mark.parametrize("terminal", [S.COMPLETED, S.ABANDONED, S.TERMINATED, S.EXPIRED])
def test_terminal_states_have_no_outgoing_edges(terminal):
    """The rule that stops a candidate rejoining an interview that was scored.

    Absorbing by construction, not by a check someone has to remember.
    """
    assert state_machine.LEGAL[terminal] == frozenset()
    assert is_terminal(terminal)
    for target in S:
        if target is not terminal:
            assert not can_transition(terminal, target)


@pytest.mark.parametrize("terminal", [S.COMPLETED, S.ABANDONED, S.TERMINATED, S.EXPIRED])
def test_a_terminal_interview_cannot_be_restarted(terminal):
    interview = _interview(terminal)
    with pytest.raises(IllegalTransitionError):
        transition(interview, S.IN_PROGRESS)
    assert interview.status is terminal


def test_the_happy_path():
    interview = _interview(S.CREATED)
    assert transition(interview, S.INVITED)
    assert transition(interview, S.IN_PROGRESS)
    assert transition(interview, S.COMPLETED)
    assert interview.status is S.COMPLETED


def test_an_interview_cannot_skip_straight_to_completed():
    with pytest.raises(IllegalTransitionError, match="INVITED to COMPLETED"):
        transition(_interview(S.INVITED), S.COMPLETED)


def test_re_entering_the_same_state_is_a_no_op_not_an_error():
    """Reconnecting candidates, redelivered tasks and duplicate socket closes
    all drive this, and none of them should have to check first."""
    interview = _interview(S.IN_PROGRESS)
    assert transition(interview, S.IN_PROGRESS) is False
    assert interview.status is S.IN_PROGRESS


# --- Timestamps -------------------------------------------------------------


def test_starting_stamps_started_at():
    interview = _interview(S.INVITED)
    assert interview.started_at is None
    transition(interview, S.IN_PROGRESS)
    assert interview.started_at is not None
    assert interview.completed_at is None


@pytest.mark.parametrize("ending", [S.COMPLETED, S.ABANDONED, S.TERMINATED])
def test_every_ending_stamps_completed_at(ending):
    interview = _interview(S.IN_PROGRESS)
    transition(interview, ending)
    assert interview.completed_at is not None


def test_a_timestamp_already_set_is_not_overwritten():
    """A TERMINATED interview that is later expired must keep the moment it
    actually stopped."""
    interview = _interview(S.IN_PROGRESS)
    transition(interview, S.TERMINATED)
    stopped_at = interview.completed_at

    # EXPIRED is not reachable from TERMINATED, so drive the guard directly.
    interview.status = S.INVITED
    transition(interview, S.EXPIRED)
    assert interview.completed_at == stopped_at


def test_is_live_is_true_only_while_in_progress():
    assert state_machine.is_live(S.IN_PROGRESS)
    for status in S:
        if status is not S.IN_PROGRESS:
            assert not state_machine.is_live(status)


# --- The event bus ----------------------------------------------------------


def _turn(**overrides) -> TurnCompleted:
    return TurnCompleted(
        org_id=uuid.uuid4(),
        interview_id=uuid.uuid4(),
        **{"ordinal": 0, "speaker": "candidate", "content": "hello", **overrides},
    )


async def test_a_subscriber_receives_its_event_type():
    bus = EventBus()
    seen: list[Event] = []
    bus.subscribe(TurnCompleted, lambda e: seen.append(e))

    bus.publish(_turn(content="an answer"))
    await bus.drain()

    assert [e.content for e in seen] == ["an answer"]


async def test_publish_does_not_block_the_publisher():
    """The voice pipeline calls this mid-turn on a 1.5s budget. Handlers must
    run after publish returns, not during it."""
    bus = EventBus()
    ran = asyncio.Event()

    async def _slow(_):
        await asyncio.sleep(0)
        ran.set()

    bus.subscribe(TurnCompleted, _slow)
    bus.publish(_turn())
    assert not ran.is_set(), "publish awaited its handler"

    await bus.drain()
    assert ran.is_set()


async def test_events_are_routed_by_exact_type():
    bus = EventBus()
    turns: list = []
    ends: list = []
    bus.subscribe(TurnCompleted, lambda e: turns.append(e))
    bus.subscribe(SessionEnded, lambda e: ends.append(e))

    bus.publish(_turn())
    await bus.drain()

    assert len(turns) == 1
    assert ends == []


async def test_a_failing_subscriber_cannot_break_a_live_interview():
    """A broken proctoring rule must not take the call down, and must not stop
    the transcript from being written."""
    bus = EventBus()
    survived: list = []

    def _explode(_):
        raise RuntimeError("subscriber is broken")

    bus.subscribe(TurnCompleted, _explode)
    bus.subscribe(TurnCompleted, lambda e: survived.append(e))

    bus.publish(_turn())
    await bus.drain()

    assert len(survived) == 1, "one handler's failure suppressed another"


async def test_an_async_failing_subscriber_is_also_contained():
    bus = EventBus()
    survived: list = []

    async def _explode(_):
        raise RuntimeError("async subscriber is broken")

    bus.subscribe(TurnCompleted, _explode)
    bus.subscribe(TurnCompleted, lambda e: survived.append(e))

    bus.publish(_turn())
    await bus.drain()
    assert len(survived) == 1


async def test_unsubscribe_stops_delivery():
    bus = EventBus()
    seen: list = []
    unsubscribe = bus.subscribe(TurnCompleted, lambda e: seen.append(e))

    unsubscribe()
    bus.publish(_turn())
    await bus.drain()

    assert seen == []


async def test_in_flight_handlers_are_strongly_referenced():
    """asyncio only holds weak references to tasks, so a handler that awaits can
    be garbage collected mid-flight and simply vanish with no error."""
    import gc

    bus = EventBus()
    finished: list = []

    async def _slow(_):
        await asyncio.sleep(0.01)
        finished.append(1)

    bus.subscribe(TurnCompleted, _slow)
    bus.publish(_turn())

    gc.collect()  # would collect the task if nothing held it
    await bus.drain()
    assert finished == [1]


async def test_publishing_with_no_subscribers_is_harmless():
    bus = EventBus()
    bus.publish(_turn())
    await bus.drain()


async def test_drain_returns_when_there_is_nothing_in_flight():
    await EventBus().drain()
