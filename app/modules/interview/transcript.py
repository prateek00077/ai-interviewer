"""Turn accumulation and persistence.

Subscribes to ``TurnCompleted`` and writes rows. It never imports anything from
``voice/`` and ``voice/`` never imports this: the bus is the whole interface,
which is what lets the voice pipeline be extracted into its own process later
without touching either side.

Writes are idempotent on ``(interview_id, ordinal)``. The bus is at-least-once
in spirit -- a reconnecting session replays its last turn, a retry re-emits --
so a duplicate must update the row rather than raise a unique violation into a
handler nobody is awaiting.
"""

from __future__ import annotations

import uuid

import structlog
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.events import TurnCompleted, subscribe
from app.db.session import tenant_session
from app.models.interview import InterviewTurn, Speaker

log = structlog.get_logger(__name__)


def _speaker(raw: str) -> Speaker:
    """Map the bus's loose string onto the enum.

    The voice module speaks in strings so it does not have to import our models.
    An unrecognised value is attributed to the interviewer rather than dropped:
    losing a line of transcript is worse than mislabelling one, and a
    mislabelled line is visible in review.
    """
    try:
        return Speaker(raw.upper())
    except ValueError:
        log.warning("unknown_speaker", speaker=raw)
        return Speaker.INTERVIEWER


async def record_turn(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    ordinal: int,
    speaker: str,
    content: str,
    started_offset_ms: int = 0,
    ended_offset_ms: int = 0,
    question_ordinal: int | None = None,
) -> None:
    """Upsert one turn.

    ON CONFLICT DO UPDATE rather than a read-then-write: two deliveries of the
    same turn can race, and a check-then-insert would let both pass the check.
    """
    statement = pg_insert(InterviewTurn).values(
        org_id=org_id,
        interview_id=interview_id,
        ordinal=ordinal,
        speaker=_speaker(speaker),
        content=content,
        started_offset_ms=started_offset_ms,
        # The CHECK constraint requires ordering; a malformed event should not
        # take down the handler.
        ended_offset_ms=max(ended_offset_ms, started_offset_ms),
        question_ordinal=question_ordinal,
    )
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=["interview_id", "ordinal"],
            set_={
                "content": statement.excluded.content,
                "ended_offset_ms": statement.excluded.ended_offset_ms,
                "question_ordinal": statement.excluded.question_ordinal,
            },
            # A turn the offline pass has already corrected must not be
            # overwritten by a late replay of the live ASR text.
            where=InterviewTurn.is_final.is_(False),
        )
    )


async def _on_turn_completed(event: TurnCompleted) -> None:
    """Bus handler. Opens its own session -- there is no request here."""
    async with tenant_session(event.org_id, "system", None) as session:
        await record_turn(
            session,
            org_id=event.org_id,
            interview_id=event.interview_id,
            ordinal=event.ordinal,
            speaker=event.speaker,
            content=event.content,
            started_offset_ms=event.started_offset_ms,
            ended_offset_ms=event.ended_offset_ms,
            question_ordinal=event.question_ordinal,
        )


def register() -> None:
    """Wire this module to the bus. Called once from the app lifespan."""
    subscribe(TurnCompleted, _on_turn_completed)


# --- Reads ------------------------------------------------------------------


async def list_turns(session: AsyncSession, interview_id: uuid.UUID) -> list[InterviewTurn]:
    rows = (
        await session.execute(
            select(InterviewTurn)
            .where(InterviewTurn.interview_id == interview_id)
            .order_by(InterviewTurn.ordinal)
        )
    ).scalars()
    return list(rows)


async def next_ordinal(session: AsyncSession, interview_id: uuid.UUID) -> int:
    """Where a resumed session should continue numbering from."""
    highest = await session.scalar(
        select(func.max(InterviewTurn.ordinal)).where(InterviewTurn.interview_id == interview_id)
    )
    return 0 if highest is None else int(highest) + 1


async def as_text(session: AsyncSession, interview_id: uuid.UUID) -> str:
    """The transcript as one block, for the scorer's prompt."""
    turns = await list_turns(session, interview_id)
    return "\n".join(f"[{t.speaker.value}] {t.content}" for t in turns)
