"""Delivery signals: what is measured, and what is deliberately not.

The most important assertion in this file is the last one. Nothing in
``confidence`` may reach a score, and the structural guarantee of that is that
``aggregator`` does not import it. A test pins that, because the day someone
"improves" the score by folding in filler rate is the day this product starts
marking down nervous candidates invisibly.
"""

import math
import struct

import pytest

from app.models.interview import InterviewTurn, Speaker
from app.modules.scoring import confidence
from app.modules.scoring.confidence import count_fillers, measure


def _turn(ordinal: int, speaker: Speaker, content: str, start: int, end: int) -> InterviewTurn:
    return InterviewTurn(
        ordinal=ordinal,
        speaker=speaker,
        content=content,
        started_offset_ms=start,
        ended_offset_ms=end,
        is_final=False,
    )


# --- Filler counting --------------------------------------------------------


def test_fillers_are_counted_and_broken_down():
    total, breakdown = count_fillers("Um, so basically, uh, I think, um, that works.")
    assert total == 4
    assert breakdown["um"] == 2
    assert breakdown["basically"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "The number of shards was fixed.",  # 'number' contains 'um'
        "We used a thumbnail cache.",  # 'thumbnail' contains 'um'
        "Aluminium housings.",
    ],
)
def test_a_word_containing_a_filler_is_not_a_filler(text):
    """Without word boundaries, every technical answer scores as hesitant."""
    total, _ = count_fillers(text)
    assert total == 0, f"a substring match fired on {text!r}"


def test_a_multi_word_filler_is_counted_once_not_twice():
    """'you know' must not also be counted by a later single-word pattern."""
    total, breakdown = count_fillers("You know, it was, you know, tricky.")
    assert breakdown["you know"] == 2
    assert total == 2


def test_counting_is_case_insensitive():
    total, _ = count_fillers("UM, ACTUALLY, Basically")
    assert total == 3


# --- Measuring without audio ------------------------------------------------


def test_the_transcript_half_works_with_no_recording_at_all():
    """A failed upload should cost the pitch numbers, not the filler rate."""
    turns = [
        _turn(0, Speaker.INTERVIEWER, "Tell me about the migration.", 0, 3_000),
        _turn(1, Speaker.CANDIDATE, "Um, we moved the write path first.", 4_000, 9_000),
    ]
    signals = measure(turns, recording=None)

    assert signals.words == 7
    assert signals.filler_count == 1
    assert signals.fillers_per_100_words == pytest.approx(14.29)
    # Not measured is not zero.
    assert signals.median_pitch_hz is None
    assert signals.pitch_variation is None


def test_only_the_candidate_is_measured():
    """The recording is a merged mix of both sides. Counting the interviewer's
    words would measure our own TTS voice as if it were the candidate."""
    turns = [
        _turn(0, Speaker.INTERVIEWER, "um um um um um um um um", 0, 3_000),
        _turn(1, Speaker.CANDIDATE, "We sharded on tenant id.", 4_000, 9_000),
    ]
    signals = measure(turns, recording=None)
    assert signals.filler_count == 0
    assert signals.words == 5


def test_an_empty_transcript_reports_nothing_rather_than_zeroes():
    signals = measure([], recording=None)
    assert signals.words == 0
    assert signals.fillers_per_100_words is None
    assert signals.words_per_minute is None


def test_unreadable_audio_degrades_to_the_transcript_signals():
    """Signals are a nice-to-have; a corrupt recording must not raise."""
    turns = [_turn(0, Speaker.CANDIDATE, "Um, yes.", 0, 2_000)]
    signals = measure(turns, recording=b"this is not audio")
    assert signals.filler_count == 1
    assert signals.median_pitch_hz is None


# --- Measuring with audio ---------------------------------------------------


def _wav_tone(hz: float, seconds: float, rate: int = 16_000) -> bytes:
    """A pure tone in a WAV container, so pitch detection has a known answer."""
    import io
    import wave

    frames = bytearray()
    for i in range(int(rate * seconds)):
        value = int(20_000 * math.sin(2 * math.pi * hz * i / rate))
        frames += struct.pack("<h", value)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(bytes(frames))
    return buffer.getvalue()


def test_pitch_is_measured_over_the_candidates_window():
    turns = [_turn(0, Speaker.CANDIDATE, "A steady sentence.", 0, 2_000)]
    signals = measure(turns, recording=_wav_tone(150.0, 2.0))

    assert signals.median_pitch_hz == pytest.approx(150.0, rel=0.05)
    # A pure tone does not vary, so the coefficient of variation is ~0. This is
    # the assertion that would catch the statistic being computed on the wrong
    # array entirely.
    assert signals.pitch_variation is not None
    assert signals.pitch_variation < 0.05
    assert signals.speaking_seconds == pytest.approx(2.0, rel=0.05)


def test_audio_outside_the_candidates_spans_is_not_measured():
    """Spans past the end of the audio must clamp, not crash or read silence."""
    turns = [_turn(0, Speaker.CANDIDATE, "Short.", 0, 60_000)]
    signals = measure(turns, recording=_wav_tone(150.0, 1.0))
    assert signals.speaking_seconds == pytest.approx(1.0, rel=0.1)


# --- The boundary that matters ----------------------------------------------


def test_the_aggregator_does_not_import_confidence():
    """Structural, not stylistic. These signals track anxiety far better than
    competence; the moment the aggregator can see them, someone will multiply by
    one and the score will quietly start penalising nervous candidates."""
    import inspect

    from app.modules.scoring import aggregator

    source = inspect.getsource(aggregator)
    assert "confidence" not in source
    assert not hasattr(aggregator, "measure")


def test_signals_serialise_to_plain_json_types():
    """They land in a JSONB column; a numpy float would fail to serialise."""
    import json

    turns = [_turn(0, Speaker.CANDIDATE, "Um, we sharded.", 0, 1_000)]
    payload = measure(turns, recording=_wav_tone(150.0, 1.0)).as_dict()
    json.dumps(payload)  # raises on anything exotic
    assert isinstance(payload["median_pitch_hz"], float)


def test_module_exposes_no_scoring_helper():
    """A ``score`` or ``rating`` function here would be the first step toward
    folding these into the number."""
    exported = [n for n in dir(confidence) if not n.startswith("_")]
    assert not any("score" in n.lower() or "rating" in n.lower() for n in exported)
