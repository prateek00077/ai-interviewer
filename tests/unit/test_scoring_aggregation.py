"""The arithmetic that turns criterion scores into a hire recommendation.

Pure functions, no database. Every test here exists because the failure it
pins is silent: a mis-weighted mean produces a plausible number, and nothing
downstream can tell it apart from a correct one.
"""

from decimal import Decimal

import pytest

from app.models.score import Recommendation
from app.modules.scoring.aggregator import MIN_COVERAGE, aggregate, band_for

D = Decimal


def _even(*scores) -> list[tuple[Decimal, Decimal | None]]:
    """Four criteria of equal weight, scored as given (None = ungraded)."""
    weight = D(1) / D(len(scores))
    return [(weight, None if s is None else D(str(s))) for s in scores]


# --- The weighted mean ------------------------------------------------------


def test_a_fully_graded_rubric_is_a_plain_weighted_mean():
    outcome = aggregate([(D("0.5"), D("4")), (D("0.3"), D("3")), (D("0.2"), D("5"))])
    # 0.5*4 + 0.3*3 + 0.2*5 = 3.9
    assert outcome.overall == D("3.90")
    assert outcome.coverage == D("1.00")


def test_weights_decide_the_outcome_not_the_criterion_count():
    """One heavily weighted criterion must be able to outvote two light ones."""
    outcome = aggregate([(D("0.8"), D("5")), (D("0.1"), D("1")), (D("0.1"), D("1"))])
    assert outcome.overall == D("4.20")
    assert outcome.recommendation is Recommendation.HIRE


# --- The coverage problem ---------------------------------------------------


def test_ungraded_criteria_do_not_drag_the_score_down():
    """THE failure this module exists to prevent.

    Three criteria worth 0.6 in total, all answered at 4.0. A plain weighted sum
    over the whole rubric gives 2.4 -- which reads as a mediocre candidate when
    what actually happened is that we never asked the remaining 40%.
    """
    outcome = aggregate(
        [(D("0.2"), D("4")), (D("0.2"), D("4")), (D("0.2"), D("4")), (D("0.4"), None)]
    )
    assert outcome.overall == D("4.00"), "ungraded weight leaked into the denominator"
    assert outcome.coverage == D("0.60")
    assert outcome.graded_count == 3
    assert outcome.total_count == 4


def test_renormalisation_keeps_relative_priorities_among_graded_criteria():
    """Rescaling must not flatten the surviving weights into an average."""
    outcome = aggregate([(D("0.6"), D("5")), (D("0.2"), D("1")), (D("0.2"), None)])
    # Graded weight 0.8; (0.6*5 + 0.2*1) / 0.8 = 4.0, not the flat mean of 3.0.
    assert outcome.overall == D("4.00")


def test_coverage_below_the_floor_refuses_to_report_a_number():
    """Renormalising one criterion out of six is not an assessment."""
    outcome = aggregate([(D("0.2"), D("5")), *[(D("0.16"), None)] * 5])
    assert outcome.overall is None
    assert outcome.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
    assert outcome.coverage < MIN_COVERAGE


def test_coverage_exactly_at_the_floor_is_reported():
    """The bound is inclusive; a rubric half-covered still yields a score."""
    outcome = aggregate([(D("0.5"), D("4")), (D("0.5"), None)])
    assert outcome.coverage == MIN_COVERAGE
    assert outcome.overall == D("4.00")


# --- Not assessed vs assessed badly -----------------------------------------


def test_nothing_graded_is_insufficient_evidence_not_no_hire():
    """A candidate whose audio failed has not been rejected. They have not been
    assessed, and the two must never share a heading."""
    outcome = aggregate(_even(None, None, None, None))
    assert outcome.overall is None
    assert outcome.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
    assert outcome.recommendation is not Recommendation.NO_HIRE


def test_an_empty_rubric_is_insufficient_evidence():
    outcome = aggregate([])
    assert outcome.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
    assert outcome.is_assessed is False


def test_a_genuinely_weak_interview_is_no_hire_not_insufficient_evidence():
    outcome = aggregate(_even(1, 1.5, 2, 1))
    assert outcome.recommendation is Recommendation.NO_HIRE
    assert outcome.is_assessed is True


# --- Bands ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overall", "expected"),
    [
        ("5.0", Recommendation.STRONG_HIRE),
        ("4.5", Recommendation.STRONG_HIRE),  # inclusive lower bound
        ("4.49", Recommendation.HIRE),
        ("3.5", Recommendation.HIRE),
        ("3.49", Recommendation.BORDERLINE),
        ("2.5", Recommendation.BORDERLINE),
        ("2.49", Recommendation.NO_HIRE),
        ("1.0", Recommendation.NO_HIRE),
    ],
)
def test_band_boundaries(overall, expected):
    assert band_for(Decimal(overall)) is expected


def test_a_middling_interview_is_borderline_rather_than_forced_either_way():
    """3.0 has genuinely not settled the question. Rounding it into HIRE or
    NO_HIRE would fabricate a decision the evidence does not support."""
    assert aggregate(_even(3, 3, 3, 3)).recommendation is Recommendation.BORDERLINE


# --- Rounding ---------------------------------------------------------------


def test_a_perfect_score_cannot_exceed_the_band_ceiling():
    """Renormalising by a repeating-decimal weight must not produce 5.0000001,
    which the CHECK constraint would reject at the very last step."""
    third = D(1) / D(3)
    outcome = aggregate([(third, D("5")), (third, D("5")), (third, D("5"))])
    assert outcome.overall == D("5.00")
    assert outcome.recommendation is Recommendation.STRONG_HIRE
