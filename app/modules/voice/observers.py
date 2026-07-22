"""Emits transcript/turn events onto core.events. No direct module calls.

This is the only outward edge of the voice module. It imports ``core.events``
and nothing else from the application -- no models, no services, no database.
That is what keeps the boundary real: everything downstream reacts to events,
so the pipeline can be lifted into its own process without either side
changing.

Turns are numbered here rather than by the subscriber, because ordering is a
property of the conversation and only this side knows it. Offsets are
milliseconds since session start, which is what ties a transcript line to a
position in the recording.
"""

from __future__ import annotations

import time
import uuid

import structlog
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from app.core.events import TurnCompleted, publish

log = structlog.get_logger(__name__)

CANDIDATE = "candidate"
INTERVIEWER = "interviewer"


class TranscriptObserver(BaseObserver):
    """Turns pipecat frames into ``TurnCompleted`` events.

    Two speakers arrive by different routes:

    - The candidate's words come as ``TranscriptionFrame``, already final, one
      per utterance. Each becomes a turn immediately.
    - The interviewer's words arrive as a stream of ``TTSTextFrame`` fragments
      as sentences are synthesised. Emitting one turn per fragment would shred
      a single reply across a dozen transcript lines, so they are accumulated
      and flushed when ``BotStoppedSpeakingFrame`` says the reply is finished.
    """

    def __init__(
        self,
        *,
        org_id: uuid.UUID,
        interview_id: uuid.UUID,
        start_ordinal: int = 0,
        started_at_ms: int | None = None,
    ) -> None:
        super().__init__()
        self._org_id = org_id
        self._interview_id = interview_id
        self._ordinal = start_ordinal
        # Wall clock only ever used as a subtrahend, so offsets stay relative.
        self._t0 = started_at_ms if started_at_ms is not None else self._now_ms()

        self._bot_parts: list[str] = []
        self._bot_started_ms: int | None = None
        # Which planned question the interviewer is on, for attribution. Bumped
        # once per interviewer turn, which is an approximation: a probing
        # follow-up is not a new question. Good enough to group a transcript by
        # question and deliberately not used for anything load-bearing.
        self._question_ordinal = 0

    @staticmethod
    def _now_ms() -> int:
        return int(time.monotonic() * 1000)

    def _offset_ms(self) -> int:
        return max(0, self._now_ms() - self._t0)

    @property
    def next_ordinal(self) -> int:
        """Where a resumed session should continue numbering."""
        return self._ordinal

    @property
    def question_ordinal(self) -> int:
        return self._question_ordinal

    @property
    def elapsed_ms(self) -> int:
        return self._offset_ms()

    def _emit(self, speaker: str, content: str, started_ms: int, ended_ms: int) -> None:
        text = content.strip()
        if not text:
            return
        publish(
            TurnCompleted(
                org_id=self._org_id,
                interview_id=self._interview_id,
                ordinal=self._ordinal,
                speaker=speaker,
                content=text,
                started_offset_ms=started_ms,
                ended_offset_ms=max(ended_ms, started_ms),
                question_ordinal=self._question_ordinal,
            )
        )
        self._ordinal += 1

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame

        if isinstance(frame, TranscriptionFrame):
            # Already one final utterance. InterimTranscriptionFrame is a
            # different class and is deliberately not handled -- interim results
            # are for display, not for the record.
            now = self._offset_ms()
            self._emit(CANDIDATE, frame.text, now, now)
            return

        if isinstance(frame, TTSTextFrame):
            if self._bot_started_ms is None:
                self._bot_started_ms = self._offset_ms()
            self._bot_parts.append(frame.text)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self.flush_bot_turn()

    def flush_bot_turn(self) -> None:
        """Emit the accumulated interviewer reply as one turn.

        Also called at session end, so a reply cut off mid-sentence by a
        disconnect still reaches the transcript.
        """
        if not self._bot_parts:
            return
        content = "".join(self._bot_parts)
        started = self._bot_started_ms if self._bot_started_ms is not None else self._offset_ms()
        self._bot_parts = []
        self._bot_started_ms = None

        self._emit(INTERVIEWER, content, started, self._offset_ms())
        self._question_ordinal += 1
