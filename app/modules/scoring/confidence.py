"""Delivery signals measured from the audio and the transcript.

READ THIS BEFORE USING ANY NUMBER IN HERE.

These are measurements of *how someone spoke*, not of how good their answers
were, and nothing in this module feeds the score. That is a deliberate product
decision, not an oversight:

- Pitch variance, pause length and filler rate track anxiety far more closely
  than they track competence. Folding them in would systematically mark down
  nervous candidates, non-native speakers, and anyone with a speech difference
  -- and would do it inside a single opaque number nobody could appeal.
- They are still worth showing. A recruiter reading "long pauses before the
  system-design answers" learns something real, and can weigh it as a human.

So: measured precisely, reported plainly, multiplied by nothing. The one place
this could ever change is the aggregator, and it does not import this module.

MEASURED OVER CANDIDATE WINDOWS ONLY. The recording is a merged mix of both
sides of the call, so analysing it whole would measure the TTS voice as much as
the person. The turn offsets are used to cut the candidate's spans out first,
and if there are none, the result is empty rather than a number computed from
the interviewer's speech.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import structlog

from app.models.interview import InterviewTurn, Speaker

log = structlog.get_logger(__name__)

TARGET_SAMPLE_RATE = 16_000

# Multi-word forms first so "you know" is not counted as two separate hits by a
# later single-word pattern.
FILLER_PHRASES = (
    "you know",
    "i mean",
    "sort of",
    "kind of",
    "um",
    "uh",
    "erm",
    "hmm",
    "like",
    "basically",
    "actually",
    "literally",
)

# Human speech sits roughly in this range; anything outside is octave error or
# noise rather than a voice, and including it would swamp the variance.
MIN_F0_HZ = 65.0
MAX_F0_HZ = 400.0

# Silence below this, relative to the span's peak, counts as a pause.
SILENCE_TOP_DB = 30
# Shorter gaps are the ordinary rhythm of a sentence, not hesitation.
MIN_PAUSE_MS = 400

_WORD_RE = re.compile(r"\w+")


@dataclass(slots=True)
class Signals:
    """What was measured. Every field is nullable -- absent is not zero.

    A ``median_pitch_hz`` of None means no voiced audio was analysable. Reporting
    0.0 there would put a real number in front of a recruiter that means the
    opposite of what it says.
    """

    speaking_seconds: float = 0.0
    words: int = 0
    words_per_minute: float | None = None

    median_pitch_hz: float | None = None
    # Coefficient of variation, so it is comparable across voices with very
    # different baselines -- a raw standard deviation in Hz is not.
    pitch_variation: float | None = None

    pause_count: int = 0
    median_pause_ms: float | None = None
    longest_pause_ms: float | None = None

    filler_count: int = 0
    fillers_per_100_words: float | None = None
    filler_breakdown: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --- Text-derived -----------------------------------------------------------


def candidate_text(turns: list[InterviewTurn]) -> str:
    return " ".join(t.content for t in turns if t.speaker is Speaker.CANDIDATE)


def count_fillers(text: str) -> tuple[int, dict[str, int]]:
    """Filler occurrences, longest phrase first.

    Word boundaries matter more than they look: without ``\\b`` around "um",
    "number" contains one and every technical answer scores as hesitant.
    Matched phrases are blanked out as they are found so "you know" is not
    re-counted by the later "know"-adjacent patterns.
    """
    remaining = text.lower()
    breakdown: dict[str, int] = {}
    total = 0
    for phrase in FILLER_PHRASES:
        pattern = re.compile(rf"\b{re.escape(phrase)}\b")
        hits = pattern.findall(remaining)
        if hits:
            breakdown[phrase] = len(hits)
            total += len(hits)
            remaining = pattern.sub(" ", remaining)
    return total, breakdown


# --- Audio-derived ----------------------------------------------------------


def _candidate_spans(turns: list[InterviewTurn]) -> list[tuple[int, int]]:
    return [
        (t.started_offset_ms, t.ended_offset_ms)
        for t in turns
        if t.speaker is Speaker.CANDIDATE and t.ended_offset_ms > t.started_offset_ms
    ]


def analyse_audio(data: bytes, spans: list[tuple[int, int]]) -> dict:
    """Pitch and pause statistics over the candidate's spans. Blocking.

    Returns an empty dict when there is nothing measurable, so the caller can
    tell "not measured" from "measured as zero".
    """
    import io

    import librosa
    import numpy as np

    samples, _ = librosa.load(io.BytesIO(data), sr=TARGET_SAMPLE_RATE, mono=True)
    if samples.size == 0 or not spans:
        return {}

    pieces = []
    for start_ms, end_ms in spans:
        start = int(start_ms / 1000 * TARGET_SAMPLE_RATE)
        end = min(int(end_ms / 1000 * TARGET_SAMPLE_RATE), samples.size)
        if end > start:
            pieces.append(samples[start:end])
    if not pieces:
        return {}

    result: dict = {}

    # Pauses are found per span, never across the concatenation: joining two
    # spans that were minutes apart would invent a pause at every seam.
    pauses_ms: list[float] = []
    for piece in pieces:
        if piece.size < TARGET_SAMPLE_RATE // 10:
            continue
        intervals = librosa.effects.split(piece, top_db=SILENCE_TOP_DB)
        for previous, following in zip(intervals, intervals[1:], strict=False):
            gap_ms = (following[0] - previous[1]) / TARGET_SAMPLE_RATE * 1000
            if gap_ms >= MIN_PAUSE_MS:
                pauses_ms.append(float(gap_ms))

    result["pause_count"] = len(pauses_ms)
    if pauses_ms:
        result["median_pause_ms"] = round(float(np.median(pauses_ms)), 1)
        result["longest_pause_ms"] = round(float(max(pauses_ms)), 1)

    voiced = np.concatenate(pieces)
    result["speaking_seconds"] = round(float(voiced.size / TARGET_SAMPLE_RATE), 2)

    # yin rather than pyin: pyin's HMM smoothing costs roughly an order of
    # magnitude more time for a summary statistic that does not need it.
    f0 = librosa.yin(voiced, fmin=MIN_F0_HZ, fmax=MAX_F0_HZ, sr=TARGET_SAMPLE_RATE)
    f0 = f0[np.isfinite(f0) & (f0 > MIN_F0_HZ) & (f0 < MAX_F0_HZ)]
    if f0.size:
        median = float(np.median(f0))
        result["median_pitch_hz"] = round(median, 1)
        # Coefficient of variation. Guarded because a division by a near-zero
        # median would report a monotone delivery as wildly variable.
        if median > 1.0:
            result["pitch_variation"] = round(float(np.std(f0) / median), 3)

    return result


# --- Entry point ------------------------------------------------------------


def measure(turns: list[InterviewTurn], recording: bytes | None = None) -> Signals:
    """Everything measurable, from whichever inputs are present.

    The transcript half works with no recording at all, which is the common
    degraded case -- a failed upload should cost the pitch numbers, not the
    filler rate too.
    """
    signals = Signals()

    text = candidate_text(turns)
    signals.words = len(_WORD_RE.findall(text))
    signals.filler_count, signals.filler_breakdown = count_fillers(text)

    if recording:
        try:
            measured = analyse_audio(recording, _candidate_spans(turns))
        except Exception as exc:  # noqa: BLE001 - signals are a nice-to-have
            log.warning("scoring.confidence_audio_failed", error=str(exc)[:300])
            measured = {}
        for key, value in measured.items():
            setattr(signals, key, value)

    if signals.speaking_seconds > 0 and signals.words:
        signals.words_per_minute = round(signals.words / (signals.speaking_seconds / 60), 1)
    if signals.words:
        signals.fillers_per_100_words = round(signals.filler_count / signals.words * 100, 2)

    return signals
