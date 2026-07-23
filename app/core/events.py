"""In-process pub/sub bus. The ONLY channel between voice/ and other modules.

WHY A BUS AT ALL, given everything runs in one process: the voice pipeline is
the one component that must never block. It is holding a live conversation on a
1.5-second turn budget, and if persisting a transcript turn or evaluating a
proctoring rule happened inline, a slow query would land as dead air in a real
person's ear. Subscribers therefore run as separate tasks and the publisher does
not await them.

That choice has a consequence worth stating: delivery is best-effort and
fire-and-forget. A subscriber that raises is logged and dropped, and events are
not persisted, so a process restart loses whatever was in flight. This is the
right trade for transcript turns -- the recording is the durable artifact and the
offline pass rebuilds the transcript from it -- and it is why the turn
checkpoint goes to Redis synchronously rather than riding on the bus.

A SUBSCRIBER MUST NEVER RAISE INTO THE PUBLISHER. Handlers are wrapped, so a
broken proctoring rule cannot take down a live interview.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, TypeVar

import structlog

log = structlog.get_logger(__name__)


# --- Event types ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Event:
    """Base for every event. Carries the tenancy a subscriber needs.

    Subscribers run outside the request that produced them, so they have no
    principal and no session. ``org_id`` is what lets one open a correctly
    scoped session of its own.
    """

    org_id: uuid.UUID
    interview_id: uuid.UUID
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class SessionStarted(Event):
    candidate_id: uuid.UUID | None = None


@dataclass(frozen=True, slots=True)
class SessionEnded(Event):
    # "completed", "abandoned", "terminated", "timed_out" -- the voice module
    # reports what happened; interview/service decides which status that means.
    reason: str = "completed"
    recording_key: str | None = None


@dataclass(frozen=True, slots=True)
class TurnCompleted(Event):
    """One exchange finished: the candidate spoke, the interviewer replied.

    Offsets are seconds from session start, not wall clock. They are what ties a
    transcript line to a position in the recording, and wall-clock timestamps
    would drift against the audio the moment anything buffered.
    """

    ordinal: int = 0
    speaker: str = "candidate"
    content: str = ""
    started_offset_ms: int = 0
    ended_offset_ms: int = 0
    question_ordinal: int | None = None


@dataclass(frozen=True, slots=True)
class ProctorEventRaised(Event):
    """A proctoring signal, from the browser or from the audio stream."""

    event_type: str = ""
    severity: str = "info"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VoiceSignalObserved(Event):
    """Raw acoustic observations from the live ASR stream.

    RAW, not interpreted. The voice module reports what the diarizer tagged and
    how long the gap was; ``modules/proctoring`` decides whether that means a
    second person in the room. Keeping the judgement on the proctoring side is
    what lets the thresholds change without touching the pipeline, and what
    keeps ``voice/`` from importing a proctoring model.
    """

    # Diarized speaker id -> words attributed to it, for this utterance.
    speaker_tag_counts: dict[int, int] = field(default_factory=dict)
    # Silence since the candidate's previous utterance.
    silence_gap_ms: int = 0
    offset_ms: int = 0


EventT = TypeVar("EventT", bound=Event)
Handler = Callable[[Any], Awaitable[None] | None]


# --- The bus ----------------------------------------------------------------


class EventBus:
    """Type-keyed async pub/sub.

    Instantiable rather than a module-level singleton so tests get a clean bus
    without unsubscribing each other's handlers. Application code uses the
    module-level ``bus`` below.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Handler]] = defaultdict(list)
        # Strong references to in-flight tasks. asyncio only holds weak ones, so
        # without this a handler can be garbage collected mid-await and simply
        # vanish -- with no error anywhere.
        self._tasks: set[asyncio.Task] = set()

    def subscribe(self, event_type: type[EventT], handler: Handler) -> Callable[[], None]:
        """Register a handler. Returns a function that unsubscribes it."""
        self._handlers[event_type].append(handler)

        def _unsubscribe() -> None:
            with_handler = self._handlers.get(event_type, [])
            if handler in with_handler:
                with_handler.remove(handler)

        return _unsubscribe

    def publish(self, event: Event) -> None:
        """Fire and forget. Returns immediately, before any handler has run.

        Synchronous on purpose: the voice pipeline calls this from inside a turn
        and must not await anything that could be slow.
        """
        handlers = self._handlers.get(type(event), [])
        if not handlers:
            return

        for handler in handlers:
            task = asyncio.create_task(self._run(handler, event))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run(self, handler: Handler, event: Event) -> None:
        """Run one handler, swallowing its failure.

        A broken subscriber must not take down a live interview, and it must not
        prevent the other subscribers to the same event from running.
        """
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception:
            # ``event_type``, NOT ``event``. structlog takes the message as its
            # first positional parameter *named* ``event``, so a keyword of that
            # name collides:
            #     TypeError: exception() got multiple values for argument 'event'
            #
            # OBSERVED in production logs: a handler raised ConflictError, and
            # the line meant to report it raised instead -- so the only place a
            # failed subscriber becomes visible was itself broken, and the
            # original error vanished. Second time a structlog reserved key has
            # done this here; see core/logging.py for the first.
            log.exception(
                "event_handler_failed",
                event_type=type(event).__name__,
                handler=getattr(handler, "__qualname__", repr(handler)),
                interview_id=str(event.interview_id),
            )

    async def drain(self, timeout_secs: float = 5.0) -> None:
        """Wait for in-flight handlers. For shutdown, and for tests.

        Without this a test would assert on a database row that a handler has
        not written yet, and a shutdown would cut off the final turn of every
        live interview.
        """
        pending = set(self._tasks)
        if pending:
            await asyncio.wait(pending, timeout=timeout_secs)

    def pending_count(self) -> int:
        """Handlers still in flight. Exposed for /metrics.

        A number that climbs and does not come back down means handlers are
        outliving the events that spawned them -- a stuck database call, most
        likely -- and it is the earliest visible symptom of that.
        """
        return len(self._tasks)

    def clear(self) -> None:
        """Drop every subscription. Tests only."""
        self._handlers.clear()


bus = EventBus()


def publish(event: Event) -> None:
    bus.publish(event)


def subscribe(event_type: type[EventT], handler: Handler) -> Callable[[], None]:
    return bus.subscribe(event_type, handler)
