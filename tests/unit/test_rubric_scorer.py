"""Evidence verification: the check that keeps a score defensible.

A model asked for verbatim quotes will sometimes return a fluent paraphrase and
occasionally a sentence nobody said. Both look identical to real evidence in the
recruiter's report, and both make the score unappealable by the candidate. These
tests pin what survives that check and what does not.

No model calls -- ``verify_evidence`` is a pure function over the transcript, and
that is exactly why it can be the enforcement point.
"""

from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.models.interview import InterviewTurn, Speaker
from app.modules.scoring.rubric_scorer import (
    MAX_TRANSCRIPT_CHARS,
    Citation,
    CriterionVerdict,
    render_transcript,
    verify_evidence,
)


def _turn(ordinal: int, speaker: Speaker, content: str, start: int = 0) -> InterviewTurn:
    return InterviewTurn(
        ordinal=ordinal,
        speaker=speaker,
        content=content,
        started_offset_ms=start,
        ended_offset_ms=start + 5_000,
        is_final=False,
    )


CANDIDATE_LINE = "We sharded on tenant id, which cost us cross-tenant reporting entirely."
INTERVIEWER_LINE = "How did you decide on the sharding key for that system?"

TURNS = [
    _turn(0, Speaker.INTERVIEWER, INTERVIEWER_LINE, 0),
    _turn(1, Speaker.CANDIDATE, CANDIDATE_LINE, 6_000),
    _turn(2, Speaker.CANDIDATE, "It took about three months to migrate the write path.", 20_000),
]


def _cite(quote: str, ordinal: int | None = None) -> list[Citation]:
    return [Citation(quote=quote, turn_ordinal=ordinal)]


# --- What survives verification ---------------------------------------------


def test_an_exact_quote_is_kept_with_its_offset():
    verified = verify_evidence(_cite(CANDIDATE_LINE), TURNS)
    assert len(verified) == 1
    assert verified[0]["quote"] == CANDIDATE_LINE
    assert verified[0]["offset_ms"] == 6_000, "the reviewer cannot jump to the moment"


def test_a_substring_of_a_candidate_turn_is_kept():
    verified = verify_evidence(_cite("cost us cross-tenant reporting"), TURNS)
    assert len(verified) == 1


def test_whitespace_and_case_differences_do_not_reject_a_real_quote():
    """A model reproducing a line normalises both, and rejecting on that would
    throw away genuine evidence for a formatting difference."""
    noisy = "  WE SHARDED   on tenant id,\n  which cost us cross-tenant reporting entirely. "
    assert len(verify_evidence(_cite(noisy), TURNS)) == 1


# --- What does not ----------------------------------------------------------


def test_a_paraphrase_is_rejected():
    """The whole point. A paraphrase reads as evidence and is not."""
    paraphrase = "They chose to shard by tenant identifier and lost cross-tenant reports."
    assert verify_evidence(_cite(paraphrase), TURNS) == []


def test_a_fabricated_quote_is_rejected():
    invented = "I personally designed the entire distributed consensus layer."
    assert verify_evidence(_cite(invented), TURNS) == []


def test_the_interviewer_is_never_evidence_of_the_candidates_ability():
    """Quoting our own question back would score the candidate on what we said."""
    assert verify_evidence(_cite(INTERVIEWER_LINE), TURNS) == []


def test_a_quote_too_short_to_be_distinctive_is_rejected():
    """Three words match half a transcript by accident; that is coincidence,
    not citation."""
    assert verify_evidence(_cite("we"), TURNS) == []
    assert verify_evidence(_cite("on tenant"), TURNS) == []


def test_a_quote_spanning_two_turns_is_rejected():
    """Turns are separate utterances, often minutes apart. Stitching them into
    one quote invents a sentence the candidate never said in one breath."""
    stitched = f"{CANDIDATE_LINE} It took about three months to migrate the write path."
    assert verify_evidence(_cite(stitched), TURNS) == []


def test_a_partly_fabricated_citation_list_keeps_only_the_real_quotes():
    citations = [
        Citation(quote=CANDIDATE_LINE),
        Citation(quote="I have twelve years of Kubernetes experience."),
        Citation(quote="three months to migrate the write path"),
    ]
    verified = verify_evidence(citations, TURNS)
    assert [v["turn_ordinal"] for v in verified] == [1, 2]


# --- Ordinals ---------------------------------------------------------------


def test_the_ordinal_is_taken_from_where_the_quote_was_found_not_from_the_claim():
    """Models mislabel these often enough that trusting the claim would send a
    reviewer to the wrong moment in the recording."""
    verified = verify_evidence(_cite(CANDIDATE_LINE, ordinal=7), TURNS)
    assert verified[0]["turn_ordinal"] == 1


# --- The verdict schema -----------------------------------------------------


def test_a_null_score_is_a_valid_verdict():
    """"The topic never came up" has to be expressible, or a model asked for a
    number will always invent one."""
    verdict = CriterionVerdict.model_validate({"score": None, "rationale": "never discussed"})
    assert verdict.score is None
    assert verdict.evidence == []


@pytest.mark.parametrize("bad", ["0", "0.9", "5.1", "11"])
def test_a_score_outside_the_band_is_rejected(bad):
    with pytest.raises(ValidationError):
        CriterionVerdict.model_validate({"score": bad})


def test_scores_are_quantised_to_what_the_column_stores():
    verdict = CriterionVerdict.model_validate({"score": "3.456"})
    assert verdict.score == Decimal("3.46")


# --- Transcript rendering ---------------------------------------------------


def test_the_rendered_transcript_labels_the_speaker_and_the_ordinal():
    rendered = render_transcript(TURNS)
    assert "#1 [CANDIDATE]" in rendered
    assert "#0 [INTERVIEWER]" in rendered


def test_a_long_transcript_is_trimmed_from_the_front():
    """The hard questions and the strongest evidence live at the end of an
    interview, so the opening is what gets dropped."""
    turns = [_turn(i, Speaker.CANDIDATE, "x" * 500, i * 1000) for i in range(200)]
    turns.append(_turn(200, Speaker.CANDIDATE, CANDIDATE_LINE, 200_000))

    rendered = render_transcript(turns)
    assert len(rendered) <= MAX_TRANSCRIPT_CHARS + 100
    assert CANDIDATE_LINE in rendered, "the end of the interview was trimmed away"
    assert "omitted" in rendered, "the trim was silent"
