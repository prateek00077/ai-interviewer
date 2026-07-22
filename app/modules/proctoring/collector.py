"""WebSocket ingest of browser events (blur, fullscreen exit, paste).

THE CLIENT ON THE OTHER END OF THIS SOCKET IS THE PERSON BEING ASSESSED. That
single fact shapes every decision in this module:

- The event type is validated against a closed enum. An open string field would
  let a browser invent types no rule scores and no reviewer recognises.
- Severity is assigned by ``rules``, never read from the message.
- ``at`` is the server clock. A client timestamp would let a candidate backdate
  an event outside the interview window.
- Ingest is rate-limited. A candidate cannot bury a real signal under ten
  thousand synthetic ones, and cannot use the socket to exhaust the database.
- Malformed messages are counted and dropped, not answered in detail. Telling a
  client exactly why its forgery failed is a tuning loop for the forger.

What this module deliberately does NOT do is treat absence of events as
innocence. A candidate who disables JavaScript sends nothing at all, which is
why the verdict distinguishes NO_DATA from CLEAN.

Rows are written under a SYSTEM session, not the candidate's. The candidate
authenticated the socket, but they must not hold a write handle to the table
recording their own conduct.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.events import ProctorEventRaised, publish
from app.models.proctoring import (
    ProctorEventType,
    ProctoringEvent,
    ProctoringPolicy,
    ProctorSeverity,
)
from app.modules.proctoring import rules

log = structlog.get_logger(__name__)

# Event types a browser is allowed to report. The derived ones -- FACE_ABSENT,
# MULTIPLE_FACES, SECOND_SPEAKER, ANOMALOUS_SILENCE -- are produced server-side
# from audio and vision, and a client claiming them would be fabricating
# evidence about itself in either direction.
CLIENT_REPORTABLE: frozenset[ProctorEventType] = frozenset(
    {
        ProctorEventType.TAB_BLUR,
        ProctorEventType.TAB_FOCUS,
        ProctorEventType.FULLSCREEN_EXIT,
        ProctorEventType.PASTE,
        ProctorEventType.COPY,
        ProctorEventType.DEVTOOLS_OPEN,
        ProctorEventType.WINDOW_RESIZE,
        ProctorEventType.FACE_FRAME,
    }
)

# Payload is free-form but bounded: it reaches JSONB and a recruiter's screen.
MAX_PAYLOAD_KEYS = 12
MAX_PAYLOAD_VALUE_CHARS = 500


class RateLimiter:
    """A fixed window per socket, in memory.

    Deliberately not Redis: this bounds one connection's behaviour, the state
    dies with the socket, and a round trip per event would put the database on
    the path of something whose whole purpose is to be cheap.
    """

    def __init__(self, per_minute: int | None = None) -> None:
        self._limit = per_minute or settings.proctor_events_per_minute
        self._window_started = datetime.now(UTC)
        self._count = 0
        self.dropped = 0

    def allow(self) -> bool:
        now = datetime.now(UTC)
        if (now - self._window_started).total_seconds() >= 60:
            self._window_started = now
            self._count = 0
        self._count += 1
        if self._count > self._limit:
            self.dropped += 1
            return False
        return True


@dataclass
class SessionCounters:
    """Per-type occurrence counts for one connection.

    Held in memory rather than counted with a query per event: escalation only
    needs "how many of this type so far", and a SELECT COUNT per browser event
    would make a tab-switch storm a database problem.
    """

    counts: dict[ProctorEventType, int] = field(default_factory=dict)

    def bump(self, event_type: ProctorEventType) -> int:
        """Returns the count BEFORE this occurrence."""
        prior = self.counts.get(event_type, 0)
        self.counts[event_type] = prior + 1
        return prior

    async def prime(self, session: AsyncSession, interview_id: uuid.UUID) -> None:
        """Load counts from earlier connections for the same interview.

        Without this a candidate could reset their own escalation simply by
        reconnecting, which the multi-use invite makes trivial.
        """
        rows = (
            await session.execute(
                select(ProctoringEvent.event_type, func.count())
                .where(ProctoringEvent.interview_id == interview_id)
                .group_by(ProctoringEvent.event_type)
            )
        ).all()
        self.counts = {row[0]: int(row[1]) for row in rows}


def _clean_payload(raw: object) -> dict:
    """Bound and stringify whatever the client sent.

    Nested structures are flattened to their repr: the payload is context for a
    human reading a timeline, not a document store.
    """
    if not isinstance(raw, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in list(raw.items())[:MAX_PAYLOAD_KEYS]:
        cleaned[str(key)[:64]] = str(value)[:MAX_PAYLOAD_VALUE_CHARS]
    return cleaned


def parse_event_type(raw: object) -> ProctorEventType | None:
    """The reported type, or None if it is not one a client may report."""
    if not isinstance(raw, str):
        return None
    try:
        event_type = ProctorEventType(raw.upper())
    except ValueError:
        return None
    return event_type if event_type in CLIENT_REPORTABLE else None


async def policy_for_interview(
    session: AsyncSession, interview_id: uuid.UUID
) -> ProctoringPolicy | None:
    """The policy of the job this interview belongs to, if any."""
    from app.models.interview import Interview

    interview = await session.get(Interview, interview_id)
    if interview is None or interview.job_id is None:
        return None
    return (
        await session.execute(
            select(ProctoringPolicy).where(ProctoringPolicy.job_id == interview.job_id)
        )
    ).scalar_one_or_none()


async def record(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    event_type: ProctorEventType,
    thresholds: rules.Thresholds,
    counters: SessionCounters,
    payload: object = None,
    offset_ms: int | None = None,
    s3_key: str | None = None,
    severity: ProctorSeverity | None = None,
) -> ProctoringEvent:
    """Persist one observation and announce it.

    ``severity`` is passed only by the server-side producers (vision, voice
    signals) which already know what they found. Client-reported events always
    have it computed here.
    """
    prior = counters.bump(event_type)
    resolved = severity or rules.severity_for(
        event_type, prior_count=prior, thresholds=thresholds
    )

    event = ProctoringEvent(
        org_id=org_id,
        interview_id=interview_id,
        event_type=event_type,
        severity=resolved,
        # Server clock, deliberately. See the module docstring.
        at=datetime.now(UTC),
        offset_ms=offset_ms,
        payload=_clean_payload(payload),
        s3_key=s3_key,
    )
    session.add(event)
    await session.flush()

    publish(
        ProctorEventRaised(
            org_id=org_id,
            interview_id=interview_id,
            event_type=event_type.value,
            severity=resolved.value,
            payload={"occurrence": prior + 1},
        )
    )

    if resolved is not ProctorSeverity.INFO:
        log.info(
            "proctor.event",
            interview_id=str(interview_id),
            type=event_type.value,
            severity=resolved.value,
            occurrence=prior + 1,
        )
    return event
