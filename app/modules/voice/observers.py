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

from app.core.events import TurnCompleted, VoiceSignalObserved, publish

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

    Both routes go through ``_first_sighting`` first, because this observer is
    invoked once per pipeline HOP rather than once per frame -- see that method
    for the triple-transcript it caused.
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
        # Highest frame id already accepted. See _first_sighting: the observer
        # is called once per pipeline hop, so the same frame arrives repeatedly.
        self._last_frame_id = -1
        # When the candidate last finished speaking, for the silence gap that
        # proctoring reads. None until they have spoken once.
        self._last_candidate_ms: int | None = None

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

    def _first_sighting(self, frame: object) -> bool:
        """True the first time this exact frame is seen.

        THE OBSERVER FIRES ONCE PER PIPELINE HOP, NOT ONCE PER FRAME. A
        ``TTSTextFrame`` produced by the TTS service is pushed TTS ->
        transport.output() -> audio_buffer -> assistant aggregator, so the
        observer sees the same object three times and appended its text three
        times.

        OBSERVED in a real transcript: every interviewer line stored as
        "Hi Prateek...Hi Prateek...Hi Prateek..." -- three identical copies
        concatenated. The candidate's turns were unaffected, which is why it hid
        for so long: a ``TranscriptionFrame`` is consumed by the user aggregator
        one hop after STT, so it only ever gets pushed once.

        Frame ids are assigned from a process-wide counter and never change, so
        "have I already seen this?" is just "is its id no higher than the last
        one I accepted?" -- O(1), and no set to grow for the length of a
        45-minute interview.
        """
        frame_id = getattr(frame, "id", None)
        if frame_id is None:
            return True
        if frame_id <= self._last_frame_id:
            return False
        self._last_frame_id = frame_id
        return True

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame

        if not self._first_sighting(frame):
            return

        if isinstance(frame, TranscriptionFrame):
            # Already one final utterance. InterimTranscriptionFrame is a
            # different class and is deliberately not handled -- interim results
            # are for display, not for the record.
            now = self._offset_ms()
            self._emit(CANDIDATE, frame.text, now, now)
            self._report_voice_signals(frame, now)
            self._last_candidate_ms = now
            return

        if isinstance(frame, TTSTextFrame):
            if self._bot_started_ms is None:
                self._bot_started_ms = self._offset_ms()
            self._bot_parts.append(frame.text)
            return

        if isinstance(frame, BotStoppedSpeakingFrame):
            self.flush_bot_turn()

    def _report_voice_signals(self, frame: object, now_ms: int) -> None:
        """Publish what the ASR heard, for proctoring to interpret.

        The model is sortformer, so diarization rides along with the
        transcription at no extra cost -- this is the only place those speaker
        tags are visible, and without forwarding them SECOND_SPEAKER can never
        fire. It was unwired until now: ``modules/proctoring/voice_signals``
        existed, was tested, and was imported by nothing.

        Never raises. The tag shape is the vendor's and varies with the model,
        and a proctoring signal must not be the reason a live interview stops.
        """
        try:
            counts: dict[int, int] = {}
            for word in getattr(frame, "words", None) or []:
                tag = getattr(word, "speaker_tag", None) or getattr(word, "speaker", None)
                if isinstance(tag, int) and tag >= 0:
                    counts[tag] = counts.get(tag, 0) + 1

            gap = 0 if self._last_candidate_ms is None else now_ms - self._last_candidate_ms
            if not counts and gap <= 0:
                return

            publish(
                VoiceSignalObserved(
                    org_id=self._org_id,
                    interview_id=self._interview_id,
                    speaker_tag_counts=counts,
                    silence_gap_ms=max(0, gap),
                    offset_ms=now_ms,
                )
            )
        except Exception:  # noqa: BLE001 - a signal must never break the call
            log.debug("voice.signal_report_failed", exc_info=True)

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
