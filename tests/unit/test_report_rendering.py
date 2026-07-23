"""Rendering both reports, and what each PDF is allowed to contain.

The candidate PDF is extracted back to text and searched for anything
score-shaped. That check is the last line of defence: the four structural
guarantees in test_report_separation.py all hold at the type level, and this one
holds at the level of "what is actually on the page a person receives".
"""

import re
import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.models.interview import InterviewTurn, Speaker
from app.modules.reports import renderer
from app.modules.reports.candidate import CandidateView
from app.modules.reports.recruiter import CriterionView, RecruiterView

D = Decimal


def _turn(ordinal: int, speaker: Speaker, content: str) -> InterviewTurn:
    return InterviewTurn(
        ordinal=ordinal,
        speaker=speaker,
        content=content,
        started_offset_ms=ordinal * 1000,
        ended_offset_ms=ordinal * 1000 + 500,
        is_final=False,
    )


def _recruiter_view(**overrides) -> RecruiterView:
    defaults = dict(
        candidate_name="Ada Lovelace",
        candidate_email="ada@example.com",
        job_title="Staff Engineer",
        interview_id=uuid.uuid4(),
        status="COMPLETED",
        completed_at=datetime.now(UTC),
        overall=D("3.60"),
        recommendation="HIRE",
        rubric_coverage=0.6,
        criteria_graded=1,
        criteria_total=2,
        scored_by="nemotron",
        has_recording=True,
        criteria=[
            CriterionView(
                name="System Design",
                weight=D("0.6"),
                score=D("4.00"),
                rationale="Named the tradeoff and what it cost.",
                evidence=[
                    {
                        "quote": "We sharded on tenant id",
                        "turn_ordinal": 1,
                        "offset_ms": 65_000,
                    }
                ],
            ),
            CriterionView(
                name="Communication",
                weight=D("0.4"),
                score=None,
                rationale="Never covered.",
                evidence=[],
            ),
        ],
        proctoring_verdict="CLEAN",
        proctoring_reasons=["no events of concern"],
        frames_analysed=12,
        delivery_signals={"filler_count": 3, "median_pitch_hz": 148.0},
        turns=[_turn(0, Speaker.CANDIDATE, "We sharded on tenant id")],
    )
    return RecruiterView(**{**defaults, **overrides})


CANDIDATE_VIEW = CandidateView(
    candidate_name="Ada Lovelace",
    job_title="Staff Engineer",
    summary="You explained the sharding decision and what it cost you.",
    strengths=[
        {"title": "Tradeoff reasoning", "detail": "You named what sharding cost in reporting."}
    ],
    growth_areas=[
        {"title": "Quantify impact", "detail": "Put numbers on the migration next time."}
    ],
)


def _pdf_text(pdf: bytes) -> str:
    """Extract the page text, so the assertions are about what a reader sees.

    pypdf is already a dependency for resume parsing, so this costs nothing.
    """
    import io

    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(pdf)).pages)


# --- The recruiter report ---------------------------------------------------


def test_the_recruiter_report_shows_the_score_with_its_evidence():
    html = renderer.render_html(renderer.RECRUITER_TEMPLATE, _recruiter_view())
    assert "3.60" in html
    assert "HIRE" in html
    assert "We sharded on tenant id" in html, "a score arrived without its evidence"
    # The offset is what lets a reviewer go and listen.
    assert "1m5s" in html


def test_the_recruiter_report_states_the_coverage_next_to_the_overall():
    """"3.6 across the whole rubric" and "3.6 across the 60% we got to" are
    different findings, and the page has to say which one it is."""
    html = renderer.render_html(renderer.RECRUITER_TEMPLATE, _recruiter_view())
    assert "60% of rubric weight" in html
    assert "1 of 2 criteria" in html


def test_an_ungraded_criterion_is_shown_as_ungraded_not_as_zero():
    html = renderer.render_html(renderer.RECRUITER_TEMPLATE, _recruiter_view())
    assert "not graded" in html
    assert "0.00 / 5" not in html


def test_the_delivery_signals_are_labelled_as_not_scored():
    """A recruiter reading "long pauses" has to know nobody marked the
    candidate down for it."""
    html = renderer.render_html(renderer.RECRUITER_TEMPLATE, _recruiter_view())
    assert "None of these contributed to the score" in html


def test_an_unassessed_interview_says_so_rather_than_rendering_blank():
    """A recruiter looking at an empty page cannot tell "not assessed" from
    "the report is broken"."""
    html = renderer.render_html(
        renderer.RECRUITER_TEMPLATE,
        _recruiter_view(
            overall=None,
            recommendation="INSUFFICIENT_EVIDENCE",
            criteria=[],
            rubric_coverage=None,
        ),
    )
    assert "NOT ASSESSED" in html
    assert "not a judgement of the candidate" in html


def test_a_verdict_never_appears_without_its_reasons():
    html = renderer.render_html(
        renderer.RECRUITER_TEMPLATE,
        _recruiter_view(proctoring_verdict="FLAGGED", proctoring_reasons=["second speaker heard"]),
    )
    assert "FLAGGED" in html
    assert "second speaker heard" in html


# --- The candidate report ---------------------------------------------------


def test_the_candidate_report_renders_the_feedback():
    html = renderer.render_html(renderer.CANDIDATE_TEMPLATE, CANDIDATE_VIEW)
    assert "Tradeoff reasoning" in html
    assert "Quantify impact" in html
    assert "You explained the sharding decision" in html


def test_the_candidate_pdf_contains_nothing_score_shaped():
    """THE test. Extracted back to text and searched the way an angry candidate
    with a PDF reader would search it."""
    pdf = renderer._to_pdf(renderer.render_html(renderer.CANDIDATE_TEMPLATE, CANDIDATE_VIEW))
    text = _pdf_text(pdf).lower()

    # Without this the whole test passes vacuously the day pypdf stops
    # extracting text -- an all-clear from a search of an empty string.
    assert "tradeoff reasoning" in text, "PDF text extraction returned nothing to search"

    for word in ("score", "rating", "overall", "recommendation", "hire", "rubric", "verdict"):
        assert word not in text, f"the candidate PDF contains {word!r}"

    # Nothing shaped like "3.6 / 5", "4/5", or "62%".
    assert not re.search(r"\d\s*(/|out of)\s*5", text), "a mark reached the candidate PDF"
    assert not re.search(r"\d+\s*%", text), "a percentage reached the candidate PDF"


def test_the_candidate_pdf_says_something_when_there_is_no_feedback():
    """Someone who sat through an interview and receives a blank page assumes
    the worst."""
    empty = CandidateView(candidate_name="Ada", job_title="Staff Engineer", summary="")
    html = renderer.render_html(renderer.CANDIDATE_TEMPLATE, empty)
    assert "contact the hiring team" in html


# --- Escaping ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("template", "view"),
    [
        (renderer.CANDIDATE_TEMPLATE, CANDIDATE_VIEW),
        (renderer.RECRUITER_TEMPLATE, _recruiter_view()),
    ],
)
def test_untrusted_strings_are_escaped(template, view):
    """Every string on these pages is candidate-supplied, model-authored, or
    both, and WeasyPrint parses the result as HTML. A candidate whose name
    contains a tag would otherwise reshape the recruiter's report."""
    hostile = "</style><script>alert(1)</script>"
    if hasattr(view, "candidate_name"):
        object.__setattr__(view, "candidate_name", hostile)

    html = renderer.render_html(template, view)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# --- Keys -------------------------------------------------------------------


def test_report_keys_are_org_prefixed_and_audience_tagged():
    org, interview = uuid.uuid4(), uuid.uuid4()
    key = renderer.report_key(org, interview, "candidate")
    assert key.startswith(f"{org}/{interview}/candidate-")
    assert key.endswith(".pdf")


def test_each_render_gets_a_fresh_key():
    """So a presigned URL a recruiter is already holding keeps resolving to the
    version they were reading, instead of changing under them."""
    org, interview = uuid.uuid4(), uuid.uuid4()
    first = renderer.report_key(org, interview, "recruiter")
    second = renderer.report_key(org, interview, "recruiter")
    assert first != second
