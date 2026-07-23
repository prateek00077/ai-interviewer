"""Which turns the offline pass may rewrite, and which it must not.

The pass improves the transcript the scorer reads. It can also destroy it: a
misaligned window replaces a good sentence with fragments of the next one, and
nothing downstream can tell. These tests pin the guards that stop that.

No ASR here -- ``reconcile`` is a pure function over turns and timestamped
words, which is what makes those guards testable at all.
"""

import pytest

from app.models.interview import InterviewTurn, Speaker
from app.modules.scoring.transcript_pass import Word, reconcile


def _turn(
    ordinal: int,
    speaker: Speaker,
    content: str,
    start: int,
    end: int,
    is_final: bool = False,
) -> InterviewTurn:
    return InterviewTurn(
        ordinal=ordinal,
        speaker=speaker,
        content=content,
        started_offset_ms=start,
        ended_offset_ms=end,
        is_final=is_final,
    )


def _words(spec: list[tuple[str, int, int]]) -> list[Word]:
    return [Word(text=t, start_ms=s, end_ms=e) for t, s, e in spec]


# Five words spanning 1000-6000ms, then five more spanning 10000-15000ms.
FIRST_SPAN = _words(
    [
        ("We", 1_000, 2_000),
        ("sharded", 2_000, 3_000),
        ("on", 3_000, 4_000),
        ("tenant", 4_000, 5_000),
        ("id", 5_000, 6_000),
    ]
)
SECOND_SPAN = _words(
    [
        ("It", 10_000, 11_000),
        ("took", 11_000, 12_000),
        ("three", 12_000, 13_000),
        ("whole", 13_000, 14_000),
        ("months", 14_000, 15_000),
    ]
)


# --- What gets corrected ----------------------------------------------------


def test_a_candidate_turn_is_replaced_with_the_better_decode():
    turns = [_turn(0, Speaker.CANDIDATE, "we charted on tenant ID", 1_000, 6_000)]
    assert reconcile(turns, FIRST_SPAN) == {0: "We sharded on tenant id"}


def test_words_are_attributed_to_the_turn_whose_window_they_fall_in():
    turns = [
        _turn(0, Speaker.CANDIDATE, "we charted on tenant ID", 1_000, 6_000),
        _turn(1, Speaker.CANDIDATE, "it took three hole months", 10_000, 15_000),
    ]
    corrections = reconcile(turns, FIRST_SPAN + SECOND_SPAN)
    assert corrections[0] == "We sharded on tenant id"
    assert corrections[1] == "It took three whole months"


def test_an_unchanged_turn_is_not_reported_as_a_correction():
    turns = [_turn(0, Speaker.CANDIDATE, "We sharded on tenant id", 1_000, 6_000)]
    assert reconcile(turns, FIRST_SPAN) == {}


# --- What is left alone -----------------------------------------------------


def test_the_interviewers_own_words_are_never_rewritten():
    """That text is what we SENT to TTS. It is already exact, and
    re-transcribing our own synthesised speech can only introduce errors."""
    turns = [_turn(0, Speaker.INTERVIEWER, "How did you shard it?", 1_000, 6_000)]
    assert reconcile(turns, FIRST_SPAN) == {}


def test_a_turn_already_marked_final_is_skipped():
    """Idempotency. A redelivered task must not re-decode what it already did."""
    turns = [_turn(0, Speaker.CANDIDATE, "we charted", 1_000, 6_000, is_final=True)]
    assert reconcile(turns, FIRST_SPAN) == {}


def test_a_turn_with_no_duration_is_skipped():
    """A zero-width window matches everything or nothing depending on rounding;
    neither is a correction worth making."""
    turns = [_turn(0, Speaker.CANDIDATE, "we charted", 3_000, 3_000)]
    assert reconcile(turns, FIRST_SPAN) == {}


def test_a_turn_the_pass_heard_nothing_for_keeps_its_live_text():
    turns = [_turn(0, Speaker.CANDIDATE, "something was said here", 50_000, 60_000)]
    assert reconcile(turns, FIRST_SPAN) == {}


# --- The guard against a destructive correction -----------------------------


def test_a_decode_far_shorter_than_the_live_text_is_rejected():
    """The guard that matters. Replacing a full sentence with two words because
    the pass half-failed is worse than keeping the live transcript."""
    turns = [
        _turn(
            0,
            Speaker.CANDIDATE,
            "We sharded on tenant id which cost us cross tenant reporting entirely "
            "and took three months to unwind afterwards",
            1_000,
            6_000,
        )
    ]
    assert reconcile(turns, FIRST_SPAN[:1]) == {}, "a five-word answer overwrote a twenty-word one"


def test_a_comparable_length_decode_is_accepted():
    turns = [_turn(0, Speaker.CANDIDATE, "we charted on tenant ID", 1_000, 6_000)]
    assert 0 in reconcile(turns, FIRST_SPAN)


# --- Boundary handling ------------------------------------------------------


def test_a_word_straddling_two_turns_lands_in_only_one():
    """Otherwise the phrase is duplicated across both turns."""
    turns = [
        _turn(0, Speaker.CANDIDATE, "aaa bbb ccc", 0, 3_400),
        _turn(1, Speaker.CANDIDATE, "ddd eee fff", 3_400, 7_000),
    ]
    words = _words(
        [("aaa", 0, 1_000), ("bbb", 1_000, 2_000), ("ccc", 3_000, 4_000), ("ddd", 4_000, 5_000)]
    )
    corrections = reconcile(turns, words)
    joined = " ".join(corrections.values())
    assert joined.count("ccc") == 1, f"a boundary word was counted twice: {corrections}"


@pytest.mark.parametrize("shift", [0, 500, -500])
def test_small_clock_drift_still_matches(shift):
    """Real offsets never line up exactly with word timestamps."""
    turns = [_turn(0, Speaker.CANDIDATE, "we charted on tenant ID", 1_000 + shift, 6_000 + shift)]
    assert 0 in reconcile(turns, FIRST_SPAN)
