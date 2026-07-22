"""Per-turn Redis snapshot; resume a session after a restart.

This is the answer to the risk named in the architecture: the voice pod is the
one stateful thing in the system, and if it dies someone's interview dies with
it. The checkpoint is what turns that from "start over" into "rejoin and carry
on" -- the candidate's invite is multi-use precisely so they can.

Redis, not Postgres, and written synchronously rather than over the event bus.
Postgres already has the durable transcript; what a resuming session needs is
the *volatile* part -- which question it had reached, how far into the clock it
was -- and it needs it in under a millisecond, on the turn budget. The bus is
fire-and-forget, so a checkpoint riding on it could be lost in exactly the crash
it exists to survive.

Losing a checkpoint is survivable: a resumed session falls back to counting the
turns already persisted. Losing it silently is not, hence the TTL is generous
relative to the interview length rather than tight.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from typing import Any

import structlog
from redis.asyncio import Redis

from app.core.config import settings

log = structlog.get_logger(__name__)

# Comfortably longer than MAX_INTERVIEW_MINUTES so a checkpoint cannot expire
# under a session that is merely slow, plus room to rejoin after a crash.
CHECKPOINT_TTL_SECS = 4 * 60 * 60


def _key(interview_id: uuid.UUID) -> str:
    return f"interview:{interview_id}:checkpoint"


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """Everything needed to pick a conversation back up."""

    interview_id: str
    # Ordinal the NEXT turn should use.
    next_ordinal: int
    # Which planned question the interviewer had reached.
    question_ordinal: int
    # Milliseconds of interview already elapsed, so the time cap survives a
    # restart rather than resetting and letting a session run twice as long.
    elapsed_ms: int
    plan_version: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> Checkpoint:
        data: dict[str, Any] = json.loads(raw)
        return cls(
            interview_id=data["interview_id"],
            next_ordinal=int(data.get("next_ordinal", 0)),
            question_ordinal=int(data.get("question_ordinal", 0)),
            elapsed_ms=int(data.get("elapsed_ms", 0)),
            plan_version=data.get("plan_version"),
        )


async def save(redis: Redis, checkpoint: Checkpoint) -> None:
    """Overwrite the snapshot. Called once per turn."""
    await redis.set(
        _key(uuid.UUID(checkpoint.interview_id)),
        checkpoint.to_json(),
        ex=CHECKPOINT_TTL_SECS,
    )


async def load(redis: Redis, interview_id: uuid.UUID) -> Checkpoint | None:
    """The last snapshot, or None if there is none or it is unreadable."""
    raw = await redis.get(_key(interview_id))
    if raw is None:
        return None
    try:
        return Checkpoint.from_json(raw)
    except (ValueError, KeyError, TypeError):
        # A corrupt checkpoint must not block a rejoin. The caller falls back to
        # counting persisted turns.
        log.warning("checkpoint_unreadable", interview_id=str(interview_id))
        return None


async def clear(redis: Redis, interview_id: uuid.UUID) -> None:
    """Drop the snapshot once the interview is genuinely over."""
    await redis.delete(_key(interview_id))


def max_interview_ms() -> int:
    return settings.max_interview_minutes * 60 * 1000
