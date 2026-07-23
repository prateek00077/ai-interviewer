"""Criterion scores -> one overall number and a recommendation band.

THE COVERAGE PROBLEM, which is the only interesting decision in this file.

The rubric's weights sum to 1.0, and that is enforced upstream. But criteria can
come back ungraded -- the interview ran short, a topic never came up, the model
cited nothing verifiable. A plain weighted sum over what remains then silently
understates the candidate: three criteria worth 0.6 in total, all scored 4.0,
would produce an overall of 2.4 out of 5. That number is not "a 2.4 candidate",
it is "a 4.0 candidate we only asked 60% of the questions to".

So the mean is taken over the graded weights only -- the weights are
renormalised across the subset that was actually assessed. This preserves the
rubric's relative priorities among the criteria that were covered and reports
the candidate at the level they actually performed at.

That is safe only while coverage is high, which is why MIN_COVERAGE exists.
Below it, too little of the rubric was reached for renormalisation to mean
anything, and the answer is INSUFFICIENT_EVIDENCE rather than a confident number
computed from one criterion out of six. Coverage is reported alongside the score
in every case, so a recruiter always sees how much of the rubric it rests on.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import structlog

from app.models.score import MAX_BAND, MIN_BAND, Recommendation

log = structlog.get_logger(__name__)

# Fraction of the rubric's total weight that must have been graded before an
# overall score is reported at all.
MIN_COVERAGE = Decimal("0.5")

# Band thresholds, applied to the 1-5 overall. Inclusive lower bounds, checked
# top down. BORDERLINE is a real answer, not a hedge: an interview that lands at
# 3.0 has genuinely not settled the question, and rounding it into HIRE or
# NO_HIRE would fabricate a decision the evidence does not support.
BANDS: tuple[tuple[Decimal, Recommendation], ...] = (
    (Decimal("4.5"), Recommendation.STRONG_HIRE),
    (Decimal("3.5"), Recommendation.HIRE),
    (Decimal("2.5"), Recommendation.BORDERLINE),
    (MIN_BAND, Recommendation.NO_HIRE),
)

_QUANTUM = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class Outcome:
    overall: Decimal | None
    recommendation: Recommendation
    coverage: Decimal
    graded_count: int
    total_count: int

    @property
    def is_assessed(self) -> bool:
        return self.overall is not None


def band_for(overall: Decimal) -> Recommendation:
    for threshold, recommendation in BANDS:
        if overall >= threshold:
            return recommendation
    return Recommendation.NO_HIRE


def aggregate(
    scored: list[tuple[Decimal, Decimal | None]], *, participated: bool = False
) -> Outcome:
    """``[(weight, score_or_None), ...]`` -> the overall outcome.

    ``participated`` is whether the candidate actually said anything. It decides
    the one genuinely ambiguous case, and the distinction matters to a real
    person:

    - Nothing graded and they never spoke -> INSUFFICIENT_EVIDENCE, no number.
      Their audio failed. Nobody assessed them, and filing that next to a poor
      performance would be a false record.
    - Nothing graded and they DID speak -> the floor of the band, NO_HIRE. They
      joined, were asked, and answered "I don't remember" and "can we move on".
      That is not missing data; it is an answer, and reporting it as
      "insufficient evidence" told a recruiter nothing about an interview that
      had in fact happened.

    Decided here from the transcript rather than by asking the model which case
    it is. MEASURED: asked to flag whether a topic had been raised, Nemotron
    said yes for a criterion the transcript never mentions -- which would have
    floor-scored a candidate on a question nobody put to them.

    Takes plain numbers rather than ORM rows so the arithmetic is testable
    without a database, and so the same function serves both the live scoring
    job and any later recomputation.
    """
    total_weight = sum((weight for weight, _ in scored), Decimal(0))
    graded = [(weight, score) for weight, score in scored if score is not None]
    graded_weight = sum((weight for weight, _ in graded), Decimal(0))

    if total_weight <= 0 or not graded:
        spoke_but_evidenced_nothing = participated and total_weight > 0
        return Outcome(
            overall=MIN_BAND if spoke_but_evidenced_nothing else None,
            recommendation=(
                Recommendation.NO_HIRE
                if spoke_but_evidenced_nothing
                else Recommendation.INSUFFICIENT_EVIDENCE
            ),
            coverage=Decimal(0),
            graded_count=0,
            total_count=len(scored),
        )

    coverage = (graded_weight / total_weight).quantize(_QUANTUM)
    if coverage < MIN_COVERAGE and not participated:
        log.warning(
            "scoring.coverage_too_low",
            coverage=str(coverage),
            graded=len(graded),
            total=len(scored),
        )
        return Outcome(
            overall=None,
            recommendation=Recommendation.INSUFFICIENT_EVIDENCE,
            coverage=coverage,
            graded_count=len(graded),
            total_count=len(scored),
        )

    # Renormalised over the graded subset; see the module docstring.
    weighted = sum((weight * score for weight, score in graded), Decimal(0))
    overall = (weighted / graded_weight).quantize(_QUANTUM, rounding=ROUND_HALF_UP)
    # The CHECK constraint is on the band, and floating rubric weights can push
    # a perfect set of 5s to 5.0000000001. Clamping here keeps a rounding
    # artefact from turning into an integrity error at the very last step.
    overall = min(max(overall, MIN_BAND), MAX_BAND)

    return Outcome(
        overall=overall,
        recommendation=band_for(overall),
        coverage=coverage,
        graded_count=len(graded),
        total_count=len(scored),
    )
