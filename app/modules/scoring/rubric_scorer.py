"""Score one rubric criterion at a time, and verify the evidence it cites.

TWO THINGS HERE ARE LOAD-BEARING.

**One criterion per model call.** Scoring the whole rubric in one request lets a
strong answer on the first criterion colour the rest: the model forms an overall
impression and then decorates it with per-criterion numbers. Separate calls cost
more requests and are the reason a candidate who is weak overall can still score
correctly on the one thing they are genuinely good at.

**Every quote is checked against the transcript.** A model asked for verbatim
evidence will sometimes produce a fluent paraphrase instead, and occasionally a
sentence nobody said. Both are indistinguishable from real evidence to the
recruiter reading the report, and both make the score unappealable. So quotes
are matched back to the candidate's own turns, and anything that does not match
is dropped. A score whose evidence all fails verification is discarded entirely
-- an unsupported number is worse than an admitted gap, because it still reads
as authoritative.

The evidence returned carries the turn ordinal and the audio offset, which is
what lets a reviewer jump to the moment in the recording and disagree.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field, field_validator

from app.integrations import nim_client
from app.models.interview import InterviewTurn, Speaker
from app.models.question_plan import RubricCriterion
from app.models.score import MAX_BAND, MIN_BAND
from app.modules import prompts
from app.modules.voice.nvidia.catalog import get_service

log = structlog.get_logger(__name__)

# A three-word "quote" matches half the transcript by accident. Below this the
# citation is not evidence, it is a coincidence.
MIN_QUOTE_CHARS = 12

# Long transcripts are truncated rather than rejected: the alternative is not
# scoring at all. Roughly 24k characters is well inside the model's window with
# the rubric and instructions alongside.
MAX_TRANSCRIPT_CHARS = 24_000

_WHITESPACE_RE = re.compile(r"\s+")


class Citation(BaseModel):
    quote: str = Field(min_length=1, max_length=2000)
    turn_ordinal: int | None = None


class CriterionVerdict(BaseModel):
    """What the model is asked for. ``score`` is nullable by design.

    Allowing null is what makes "the topic never came up" expressible. Without
    it, a model asked for a number always produces one, and a criterion the
    interview never reached gets scored on nothing.

    A null does NOT mean the candidate scored zero. Whether an unevidenced
    criterion counts against them is decided in ``aggregator`` from whether they
    participated at all -- not here, and deliberately not by the model. MEASURED:
    asked to flag "was this topic raised?", Nemotron answered yes for a
    criterion the transcript never mentions, which would have floor-scored a
    candidate on a question nobody put to them.
    """

    score: Decimal | None = None
    rationale: str | None = Field(default=None, max_length=4000)
    evidence: list[Citation] = Field(default_factory=list, max_length=8)

    @field_validator("score")
    @classmethod
    def _in_band(cls, v: Decimal | None) -> Decimal | None:
        if v is None:
            return None
        if not (MIN_BAND <= v <= MAX_BAND):
            raise ValueError(f"score must be between {MIN_BAND} and {MAX_BAND}, got {v}")
        # Quantised here rather than at the database boundary so the repair turn
        # sees the value that will actually be stored.
        return v.quantize(Decimal("0.01"))


@dataclass(slots=True)
class Graded:
    """A verified result, ready to persist."""

    score: Decimal | None
    rationale: str | None
    evidence: list[dict]


# --- Transcript rendering ---------------------------------------------------


def render_transcript(turns: list[InterviewTurn]) -> str:
    """The transcript as the model sees it, with ordinals it can cite.

    Ordinals are included so the model can point at a line rather than only at a
    string. They are a convenience for review, not the verification mechanism --
    a wrong ordinal is corrected from the quote match, never trusted over it.
    """
    lines = [f"#{t.ordinal} [{t.speaker.value}] {t.content}" for t in turns]
    rendered = "\n".join(lines)
    if len(rendered) > MAX_TRANSCRIPT_CHARS:
        # Trimmed from the front. The end of an interview is where the hard
        # questions and the strongest evidence live.
        rendered = "[...earlier turns omitted...]\n" + rendered[-MAX_TRANSCRIPT_CHARS:]
    return rendered


def _normalise(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip().lower()


# --- Evidence verification --------------------------------------------------


def verify_evidence(citations: list[Citation], turns: list[InterviewTurn]) -> list[dict]:
    """Keep only quotes that really appear in a candidate turn.

    Matching is whitespace- and case-insensitive because a model reproducing a
    line will normalise both, and rejecting on that would throw away genuine
    evidence. It is otherwise a strict substring test: a paraphrase does not
    match, which is exactly the intent.

    The ordinal is taken from the turn the quote was *found* in, not from what
    the model claimed. Those disagree often enough that trusting the claim would
    send a reviewer to the wrong moment in the recording.
    """
    candidate_turns = [t for t in turns if t.speaker is Speaker.CANDIDATE]
    haystacks = [(t, _normalise(t.content)) for t in candidate_turns]

    verified: list[dict] = []
    for citation in citations:
        quote = citation.quote.strip()
        if len(quote) < MIN_QUOTE_CHARS:
            continue
        needle = _normalise(quote)
        for turn, hay in haystacks:
            if needle in hay:
                verified.append(
                    {
                        "quote": quote,
                        "turn_ordinal": turn.ordinal,
                        "offset_ms": turn.started_offset_ms,
                    }
                )
                break
        else:
            log.info("scoring.evidence_unverified", quote=quote[:120])
    return verified


# --- Scoring ----------------------------------------------------------------


def _format_descriptors(descriptors: dict) -> str:
    if not descriptors:
        return "(none supplied -- score on the criterion description alone)"
    return "\n".join(f"  {band}: {text}" for band, text in sorted(descriptors.items()))


async def score_criterion(criterion: RubricCriterion, turns: list[InterviewTurn]) -> Graded:
    """Grade one criterion. Never raises: an ungraded criterion is a valid result.

    A model or network failure produces the same shape as an honest "not enough
    evidence", with the reason in the rationale. That is deliberate: the
    aggregator already knows how to exclude an ungraded criterion, so one flaky
    call costs that dimension rather than the whole report.
    """
    messages = prompts.render(
        "scorer",
        criterion_name=criterion.name,
        criterion_description=criterion.description or "(no description supplied)",
        descriptors=_format_descriptors(criterion.descriptors),
        transcript=render_transcript(turns),
    )

    try:
        verdict = await nim_client.complete_structured(messages, CriterionVerdict)
    except Exception as exc:  # noqa: BLE001 - one criterion must not sink the report
        log.warning("scoring.criterion_failed", criterion=criterion.name, error=str(exc)[:300])
        # Ungraded, never a floor score. Our infrastructure failing is not the
        # candidate answering badly, and must not cost them a mark.
        return Graded(score=None, rationale=f"Scoring failed: {str(exc)[:200]}", evidence=[])

    evidence = verify_evidence(verdict.evidence, turns)

    if verdict.score is not None and not evidence:
        # The model produced a number it could not support. Reported as ungraded
        # with the reason preserved, rather than silently kept.
        log.warning(
            "scoring.score_discarded_unsupported",
            criterion=criterion.name,
            claimed=str(verdict.score),
            cited=len(verdict.evidence),
        )
        return Graded(
            score=None,
            rationale=(
                "Discarded: the model proposed a score but cited no evidence that could be "
                "found in the transcript. "
                f"Its stated reasoning was: {(verdict.rationale or '').strip()[:1000]}"
            ),
            evidence=[],
        )

    return Graded(score=verdict.score, rationale=verdict.rationale, evidence=evidence)


async def score_all(
    criteria: list[RubricCriterion], turns: list[InterviewTurn]
) -> tuple[list[tuple[RubricCriterion, Graded]], str]:
    """Grade every criterion. Returns the results and the model that produced them.

    Sequential, not gathered. The criteria are few (3 to 6) and the shared NIM
    endpoint sheds load under concurrency with a 503 the client then has to back
    off from -- so firing them in parallel makes the whole job slower, not
    faster, on top of being ruder to a metered service.
    """
    results = [(criterion, await score_criterion(criterion, turns)) for criterion in criteria]
    graded = sum(1 for _, r in results if r.score is not None)
    log.info("scoring.criteria_scored", total=len(results), graded=graded)
    return results, get_service("llm").model
