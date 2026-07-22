"""Injects question plan + retrieved resume chunks into the LLM context.

THE SYSTEM PROMPT IS ASSEMBLED HERE AND NEVER LEAVES THE SERVER. The browser
gets audio and nothing else -- no prompt, no plan, no rubric. If the prompt were
built client-side, or even round-tripped through the client, a candidate could
read the questions before answering them and edit the instructions that govern
scoring. That is the whole reason the voice pipeline runs in this process rather
than in the browser talking to NVIDIA directly.

The rubric is deliberately NOT in the prompt. The interviewer's job is to elicit
evidence; scoring happens offline against the frozen rubric. Telling the live
model how answers will be weighted invites it to steer the candidate toward a
better score, which corrupts the evidence it is collecting.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.question_plan import QuestionPlan
from app.modules import prompts
from app.modules.interview import service as interview_service
from app.modules.jobs import service as jobs_service
from app.modules.question_plan import service as plan_service
from app.modules.resume import retriever
from app.modules.users import service as users_service

log = structlog.get_logger(__name__)

DEFAULT_DURATION_MINUTES = 30
RESUME_TOP_K = 6

NO_JOB = "(no job description was provided)"
NO_RESUME = "(no resume was provided)"


@dataclass(frozen=True, slots=True)
class InterviewContext:
    """Everything the live model is told, plus what the session needs to track."""

    messages: list[dict[str, str]]
    question_count: int
    plan_version: int | None
    candidate_name: str | None


def _render_questions(plan: QuestionPlan | None) -> str:
    """The plan as a numbered brief.

    Follow-up hints are included because they are the difference between a
    probe that lands and one the model improvises. Competency names are NOT --
    they are rubric vocabulary, and a model told which criterion a question
    feeds tends to say so out loud.
    """
    if plan is None or not plan.questions:
        return (
            "No plan was prepared. Interview from the job context above: ask about "
            "the candidate's relevant experience, probe for specifics, and keep to "
            "the time."
        )

    lines: list[str] = []
    for question in plan.questions:
        lines.append(f"{question.ordinal + 1}. {question.body}")
        for hint in question.follow_up_hints or []:
            lines.append(f"   - if the answer is thin: {hint}")
    return "\n".join(lines)


async def build(session: AsyncSession, interview_id: uuid.UUID) -> InterviewContext:
    """Assemble the live context for one interview.

    Every lookup here is optional. An interview with no job, no resume and no
    plan still produces a usable prompt, because refusing to start would strand
    a candidate who is already connected.
    """
    interview = await interview_service.get_interview(session, interview_id)

    candidate_name: str | None = None
    try:
        candidate = await users_service.get_candidate(session, interview.candidate_id)
        candidate_name = candidate.full_name
    except Exception:  # noqa: BLE001 - a name is a nicety, not a requirement
        log.warning("voice.candidate_name_unavailable", interview_id=str(interview_id))

    job_title = "this role"
    job_description = NO_JOB
    if interview.job_id is not None:
        job = await jobs_service.get_job(session, interview.job_id)
        job_title = job.title
        description = await jobs_service.get_active_description(session, job.id)
        if description is not None:
            job_description = description.content

    plan = await plan_service.get_for_interview(session, interview_id)

    resume_context = NO_RESUME
    resume = await retriever.latest_ready_resume(session, interview.candidate_id)
    if resume is not None:
        chunks = await retriever.search(
            session,
            resume_id=resume.id,
            query=f"{job_title}. {job_description}"[:2000],
            top_k=RESUME_TOP_K,
        )
        if chunks:
            resume_context = "\n\n".join(c.content for c in chunks)

    messages = prompts.render(
        "interviewer",
        job_title=job_title,
        job_description=job_description,
        resume_context=resume_context,
        questions=_render_questions(plan),
        duration_minutes=DEFAULT_DURATION_MINUTES,
    )

    log.info(
        "voice.context_built",
        interview_id=str(interview_id),
        questions=len(plan.questions) if plan else 0,
        has_resume=resume is not None,
        has_job=interview.job_id is not None,
    )
    return InterviewContext(
        messages=messages,
        question_count=len(plan.questions) if plan else 0,
        plan_version=plan.version if plan else None,
        candidate_name=candidate_name,
    )
