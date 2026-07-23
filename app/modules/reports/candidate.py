"""Candidate report: feedback and gaps ONLY.

THE GUARANTEE IS THE SIGNATURE. ``generate`` takes a job title, a list of
topic names, and a transcript. It does not take a ``Score``, a
``CriterionScore``, an overall, or a recommendation -- so there is no value in
scope that could leak into the output, whatever the model decides to write.
That is a stronger property than "we remembered to strip it out", and it is
enforced by a test that inspects this module's parameters.

WHY BAND DESCRIPTORS ARE ALSO WITHHELD, though they are not scores: a
descriptor read back to a candidate ("does not name a specific tradeoff they
made") is the scoring rubric in disguise. Handing it over tells the next
applicant exactly what to say, which quietly destroys the assessment for
everyone who comes after.

WHY THIS RUNS A SEPARATE MODEL PASS rather than reusing the scorer's
rationales: those rationales are written for a recruiter deciding whether to
hire, and they cite band descriptors by name. Rewriting them for a candidate
would mean laundering the same text through a filter and hoping. Generating
fresh from the transcript costs one more call and cannot leak what it was
never shown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import structlog
from pydantic import BaseModel, Field

from app.integrations import nim_client
from app.modules import prompts

log = structlog.get_logger(__name__)

MAX_TRANSCRIPT_CHARS = 24_000

# What a candidate is told when the interview produced nothing to work from.
# Deliberately not silence: someone who sat through an interview and gets an
# empty page assumes the worst.
NO_TRANSCRIPT_SUMMARY = (
    "We were not able to capture enough of your interview to write useful "
    "feedback. This is a problem on our side, not a reflection of how you did. "
    "Please contact the hiring team."
)


class FeedbackItem(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    detail: str = Field(min_length=10, max_length=2000)


class Feedback(BaseModel):
    strengths: list[FeedbackItem] = Field(default_factory=list, max_length=4)
    growth_areas: list[FeedbackItem] = Field(default_factory=list, max_length=4)
    summary: str = Field(default="", max_length=2000)


@dataclass(slots=True)
class CandidateView:
    """Everything the candidate PDF renders. Note the absent fields."""

    candidate_name: str
    job_title: str
    summary: str
    strengths: list[dict] = field(default_factory=list)
    growth_areas: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def _render_topics(topic_names: list[str]) -> str:
    if not topic_names:
        return "(the interview was unstructured)"
    return "\n".join(f"- {name}" for name in topic_names)


def _render_transcript(turns: list) -> str:
    """Speaker-labelled prose. No ordinals -- a candidate cites nothing."""
    lines = [f"[{t.speaker.value}] {t.content}" for t in turns if t.content.strip()]
    rendered = "\n".join(lines)
    if len(rendered) > MAX_TRANSCRIPT_CHARS:
        rendered = rendered[-MAX_TRANSCRIPT_CHARS:]
    return rendered


async def generate(
    *,
    job_title: str,
    topic_names: list[str],
    turns: list,
) -> Feedback:
    """Feedback from the transcript alone.

    The parameter list is the security boundary; do not add a score-bearing
    argument here. If a future feature needs one, it needs a different function.

    Never raises: feedback is a courtesy the candidate is owed, and a model
    outage should produce an honest note rather than a failed Celery task that
    retries a language model every thirty seconds.
    """
    transcript = _render_transcript(turns)
    if not transcript.strip():
        return Feedback(summary=NO_TRANSCRIPT_SUMMARY)

    messages = prompts.render(
        "candidate_feedback",
        job_title=job_title,
        topics=_render_topics(topic_names),
        transcript=transcript,
    )
    try:
        feedback = await nim_client.complete_structured(messages, Feedback)
    except Exception as exc:  # noqa: BLE001
        log.warning("reports.candidate_feedback_failed", error=str(exc)[:300])
        return Feedback(summary=NO_TRANSCRIPT_SUMMARY)

    log.info(
        "reports.candidate_feedback_generated",
        strengths=len(feedback.strengths),
        growth_areas=len(feedback.growth_areas),
    )
    return feedback


def build_view(
    *, candidate_name: str, job_title: str, feedback: Feedback | None, stored: object = None
) -> CandidateView:
    """Assemble the render model from generated or previously stored feedback.

    ``stored`` is a ``CandidateReport`` row. It is typed loosely on purpose:
    this module must not import the score models even transitively, and keeping
    the annotation broad makes that hard to do by accident.
    """
    if feedback is not None:
        return CandidateView(
            candidate_name=candidate_name,
            job_title=job_title,
            summary=feedback.summary,
            strengths=[item.model_dump() for item in feedback.strengths],
            growth_areas=[item.model_dump() for item in feedback.growth_areas],
        )

    return CandidateView(
        candidate_name=candidate_name,
        job_title=job_title,
        summary=getattr(stored, "summary", "") or "",
        strengths=list(getattr(stored, "strengths", []) or []),
        growth_areas=list(getattr(stored, "growth_areas", []) or []),
    )
