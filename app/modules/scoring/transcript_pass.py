"""Full-quality re-transcription of the recording, reconciled with live turns.

WHY THIS IS NOT AN "OFFLINE ASR" CALL, as the plan assumed: the NVCF function
this deployment uses -- ``cache-aware-parakeet-rnnt-...-sortformer`` -- is
online-only. ``offline_recognize`` against it is an error, not a slower path,
which is recorded in ``config/services.cloud.yaml`` as ``streaming_only``. The
recording is therefore pushed through the *streaming* API from a file rather
than from a microphone. Riva does not require real-time pacing, so a 30-minute
call streams in well under a minute of wall clock.

WHAT THIS PASS CORRECTS AND WHAT IT LEAVES ALONE:

- Text of CANDIDATE turns: replaced. Live ASR runs under a latency budget with
  a partial-hypothesis cutoff; the same audio decoded without that pressure is
  better, and the transcript is what the scorer reads.
- Text of INTERVIEWER turns: never touched. That text is what we *sent* to TTS.
  It is already exact, and re-transcribing our own synthesised speech can only
  introduce errors into a string we already know.
- Timings: never touched. The live session stamped offsets against the audio
  clock as it ran; the re-transcription's timeline is reconstructed from a file
  and is the less trustworthy of the two.

The pass is fail-soft throughout. A missing recording, an ASR outage, or a
result that decodes to nothing all leave the live turns standing -- a slightly
worse transcript scores fine, and a missing one does not score at all.
"""

from __future__ import annotations

import asyncio
import io
import re
import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.integrations import storage
from app.models.interview import InterviewTurn, Speaker
from app.modules.voice.nvidia.catalog import ServiceSpec, get_service

log = structlog.get_logger(__name__)

TARGET_SAMPLE_RATE = 16_000
# ~200ms of 16-bit mono audio. Small enough that Riva starts decoding promptly,
# large enough that a long call is not thousands of round trips.
CHUNK_BYTES = 3_200 * 2

# A word must overlap a turn's window by at least this much of its own duration
# before it is attributed to that turn. Words straddling a boundary otherwise
# land in both, and the transcript gains duplicated phrases.
MIN_OVERLAP_RATIO = 0.5

# Below this, the re-transcription is treated as a failure to hear rather than
# as a correction. Replacing a real sentence with two words because the pass
# went wrong is worse than keeping the live text.
MIN_REPLACEMENT_RATIO = 0.5

_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    start_ms: int
    end_ms: int


# --- Audio ------------------------------------------------------------------


def decode_to_pcm16(data: bytes) -> bytes:
    """Any recording -> 16 kHz mono signed 16-bit PCM, which is what Riva wants.

    librosa rather than the ``wave`` module: the recording's sample rate follows
    the transport's negotiated rate, so assuming 16 kHz would silently
    time-shift every word whenever WebRTC settled on 48 kHz.
    """
    import librosa
    import numpy as np

    samples, _ = librosa.load(io.BytesIO(data), sr=TARGET_SAMPLE_RATE, mono=True)
    if samples.size == 0:
        return b""
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


# --- ASR --------------------------------------------------------------------


def _transcribe_blocking(pcm: bytes, spec: ServiceSpec, api_key: str) -> list[Word]:
    """Stream PCM through Riva and collect timestamped words. Blocking."""
    import riva.client

    metadata = [["authorization", f"Bearer {api_key}"]]
    if spec.function_id:
        metadata.append(["function-id", spec.function_id])

    auth = riva.client.Auth(uri=spec.grpc_server, use_ssl=spec.use_ssl, metadata_args=metadata)
    service = riva.client.ASRService(auth)

    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=TARGET_SAMPLE_RATE,
            language_code=spec.option("language", "en-US"),
            model=spec.model,
            max_alternatives=1,
            enable_automatic_punctuation=True,
            # The whole point of the pass: without offsets there is no way to
            # map corrected text back onto the turns it belongs to.
            enable_word_time_offsets=True,
        ),
        # Partials are the live pipeline's concern. Only settled text is wanted
        # here, and interim results would multiply the traffic for output that
        # is discarded.
        interim_results=False,
    )

    def _chunks():
        for start in range(0, len(pcm), CHUNK_BYTES):
            yield pcm[start : start + CHUNK_BYTES]

    words: list[Word] = []
    for response in service.streaming_response_generator(
        audio_chunks=_chunks(), streaming_config=config
    ):
        for result in response.results:
            if not result.is_final or not result.alternatives:
                continue
            words.extend(
                Word(text=w.word, start_ms=int(w.start_time), end_ms=int(w.end_time))
                for w in result.alternatives[0].words
            )
    return words


async def transcribe(recording_key: str, spec: ServiceSpec | None = None) -> list[Word]:
    """Fetch the recording and re-transcribe it. Returns [] on any failure."""
    spec = spec or get_service("stt")
    api_key = settings.nvidia_api_key.get_secret_value()
    if not api_key:
        log.warning("scoring.transcript_pass_no_key")
        return []

    try:
        data = await storage.get_bytes(bucket=settings.s3_bucket_recordings, key=recording_key)
        # Both the decode and the gRPC stream are blocking and heavy; neither
        # belongs on the worker's shared event loop.
        pcm = await asyncio.to_thread(decode_to_pcm16, data)
        if not pcm:
            return []
        return await asyncio.to_thread(_transcribe_blocking, pcm, spec, api_key)
    except Exception as exc:  # noqa: BLE001 - a better transcript is an upgrade, not a requirement
        log.warning("scoring.transcript_pass_failed", key=recording_key, error=str(exc)[:300])
        return []


# --- Reconciliation ---------------------------------------------------------


def _words_within(words: list[Word], start_ms: int, end_ms: int) -> list[Word]:
    """Words whose span lies mostly inside [start_ms, end_ms]."""
    inside: list[Word] = []
    for word in words:
        duration = max(word.end_ms - word.start_ms, 1)
        overlap = min(word.end_ms, end_ms) - max(word.start_ms, start_ms)
        if overlap / duration >= MIN_OVERLAP_RATIO:
            inside.append(word)
    return inside


def _word_count(value: str) -> int:
    return len(_WORD_RE.findall(value))


def reconcile(turns: list[InterviewTurn], words: list[Word]) -> dict[int, str]:
    """Corrected text per turn ordinal. Only candidate turns are considered.

    A turn is corrected only when the re-transcription produced a comparable
    amount of speech for its window. The check is on word count rather than on
    similarity: the pass exists precisely because the two strings should differ,
    so a similarity threshold would reject exactly the corrections worth having.
    """
    corrections: dict[int, str] = {}
    for turn in turns:
        if turn.speaker is not Speaker.CANDIDATE or turn.is_final:
            continue
        if turn.ended_offset_ms <= turn.started_offset_ms:
            continue

        matched = _words_within(words, turn.started_offset_ms, turn.ended_offset_ms)
        if not matched:
            continue

        corrected = " ".join(w.text for w in matched).strip()
        live_words = _word_count(turn.content)
        if live_words and _word_count(corrected) < live_words * MIN_REPLACEMENT_RATIO:
            log.info(
                "scoring.correction_rejected",
                ordinal=turn.ordinal,
                live_words=live_words,
                corrected_words=_word_count(corrected),
            )
            continue
        if corrected and corrected != turn.content:
            corrections[turn.ordinal] = corrected
    return corrections


async def apply(
    session: AsyncSession, *, interview_id: uuid.UUID, recording_key: str | None
) -> dict:
    """Run the pass and persist the corrections. Idempotent.

    ``is_final`` is the idempotency key: a turn this pass has already corrected
    is skipped on a redelivery, and the transcript writer's upsert refuses to
    overwrite a final turn with a late replay of the live text.
    """
    if not recording_key:
        return {"corrected": 0, "skipped": "no recording"}

    turns = list(
        (
            await session.execute(
                select(InterviewTurn)
                .where(InterviewTurn.interview_id == interview_id)
                .order_by(InterviewTurn.ordinal)
            )
        ).scalars()
    )
    if not turns:
        return {"corrected": 0, "skipped": "no turns"}

    words = await transcribe(recording_key)
    if not words:
        return {"corrected": 0, "skipped": "no words decoded"}

    corrections = reconcile(turns, words)
    for turn in turns:
        replacement = corrections.get(turn.ordinal)
        if replacement is not None:
            turn.content = replacement
            turn.is_final = True

    # Candidate turns the pass looked at but did not change are final too: they
    # have been through the better decoder and it agreed. Leaving them mutable
    # would invite a late live-ASR replay to undo that.
    await session.execute(
        update(InterviewTurn)
        .where(
            InterviewTurn.interview_id == interview_id,
            InterviewTurn.speaker == Speaker.CANDIDATE,
        )
        .values(is_final=True)
    )
    await session.flush()

    log.info(
        "scoring.transcript_reconciled",
        interview_id=str(interview_id),
        words=len(words),
        corrected=len(corrections),
    )
    return {"corrected": len(corrections), "words": len(words)}
