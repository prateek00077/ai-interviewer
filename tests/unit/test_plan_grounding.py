"""The plan must be written from the candidate's resume, not from the vacancy.

OBSERVED with a real CV: a MERN/React/MongoDB background against a
Python/FastAPI/Postgres role produced questions like "when you used async
SQLAlchemy with Celery, how did you manage transaction boundaries?" -- about
work the candidate had never done. They would answer "I have not done that", the
interviewer learns nothing, and the candidate spends the interview apologising.

Retrieval was not the problem: all six resume chunks reached the prompt. The
model was reading the job description and inventing a career to match it.
"""

from types import SimpleNamespace

import pytest

from app.modules import prompts
from app.modules.question_plan import generator
from app.modules.question_plan.generator import (
    GeneratedCriterion,
    GeneratedPlan,
    GeneratedQuestion,
    _ungrounded,
)

RESUME = (
    "[projects] TypingArena - a real-time multiplayer typing game built with "
    "the MERN stack, Socket.IO for live rooms and MongoDB for persistence. "
    "Scaled to 30+ concurrent sessions.\n\n"
    "[skills] React, Node.js, Express, MongoDB, Tailwind."
)


def _plan(*evidence: str) -> GeneratedPlan:
    return GeneratedPlan(
        questions=[
            GeneratedQuestion(body=f"Tell me about {e or 'it'}.", resume_evidence=e)
            for e in evidence
        ],
        criteria=[
            GeneratedCriterion(
                name=name,
                weight="0.3334" if name == "depth" else "0.3333",
                descriptors={"1": "weak", "3": "ok", "5": "strong"},
            )
            for name in ("depth", "ownership", "clarity")
        ],
    )


def _q(body: str, evidence: str) -> GeneratedQuestion:
    return GeneratedQuestion(body=body, resume_evidence=evidence)


def _plan_of(*questions: GeneratedQuestion) -> GeneratedPlan:
    """A plan with arbitrary body/evidence pairs, for the cases where the two
    must differ -- which is the whole point of the body check."""
    return GeneratedPlan(
        questions=list(questions),
        criteria=[
            GeneratedCriterion(
                name=name,
                weight="0.3334" if name == "depth" else "0.3333",
                descriptors={"1": "weak", "3": "ok", "5": "strong"},
            )
            for name in ("depth", "ownership", "clarity")
        ],
    )


# The verbatim bodies from the real failure: a MERN candidate, a
# Python/Celery/SQLAlchemy/S3 vacancy, and a plan that asked about the vacancy.
_JOB_DESCRIPTION_QUESTIONS = [
    "Explain how you configured Celery workers to process background jobs for "
    "multiple tenants while ensuring isolation and avoiding resource contention.",
    "Describe the async SQLAlchemy query pattern you used to fetch related "
    "records for a tenant's order history, including any indexing strategies.",
    "What were the biggest challenges you faced when designing the multi-tenant "
    "isolation layer, and how did you mitigate them?",
]
_RESUME_QUESTION = (
    "When you scaled the service to support 30+ concurrent sessions, what "
    "metrics did you track and what adjustments did you make to maintain it?"
)


def test_evidence_quoted_from_the_resume_is_grounded():
    assert _ungrounded(_plan("Socket.IO for live rooms and MongoDB"), RESUME) == []


def test_a_real_citation_does_not_excuse_a_body_about_the_vacancy():
    """The production bug: the model quotes a real resume line into
    resume_evidence and then asks about the job's technology anyway. The
    evidence check passed it because the citation IS real; only looking at the
    body catches that the question is not about the candidate's work."""
    plan = _plan_of(
        *[_q(body, "Scaled to 30+ concurrent sessions") for body in _JOB_DESCRIPTION_QUESTIONS]
    )
    assert len(_ungrounded(plan, RESUME)) == len(_JOB_DESCRIPTION_QUESTIONS)


def test_a_question_about_the_candidates_own_work_survives_the_body_check():
    plan = _plan_of(_q(_RESUME_QUESTION, "Scaled to 30+ concurrent sessions"))
    assert _ungrounded(plan, RESUME) == []


def test_a_technology_name_ending_a_sentence_still_matches():
    """The word pattern keeps the dot in "socket.io"; it must not also keep the
    sentence-ending period, or "Tell me about MongoDB." stops matching the
    resume's "MongoDB" and a grounded question reads as invented."""
    assert _ungrounded(_plan_of(_q("Tell me about MongoDB.", "MongoDB")), RESUME) == []


def test_reformatted_evidence_still_counts_as_grounded():
    """The model re-cases and re-punctuates what it copies far more often than
    it fabricates. A substring test would fail all of that as invention."""
    assert _ungrounded(_plan("socket.io live rooms, mongodb persistence"), RESUME) == []


def test_invented_evidence_is_caught():
    """The whole point: a citation nobody checks is worse than no citation,
    because it makes an invented question look verified."""
    offenders = _ungrounded(
        _plan("designed Kubernetes autoscaling for a Postgres sharding layer"), RESUME
    )
    assert len(offenders) == 1


def test_a_question_citing_nothing_is_ungrounded():
    """Empty evidence means nothing was claimed, so nothing was checked."""
    assert len(_ungrounded(_plan(""), RESUME)) == 1


def test_stopword_overlap_does_not_ground_a_question():
    """Without the stopword filter, "how would you approach the work on it"
    scores full marks against any resume in English."""
    assert len(_ungrounded(_plan("the work on it was for a project"), RESUME)) == 1


def test_grounded_and_invented_questions_are_separated():
    offenders = _ungrounded(_plan("MERN stack, Socket.IO", "Kubernetes operators"), RESUME)
    assert [q.resume_evidence for q in offenders] == ["Kubernetes operators"]


def _rendered() -> tuple[str, str]:
    messages = prompts.render(
        "plan_generator",
        job_title="Senior Backend Engineer",
        job_description="Python, FastAPI, Postgres, async SQLAlchemy, Celery.",
        resume_context="[projects] TypingArena - MERN stack, Socket.IO, MongoDB.",
        question_count=8,
        duration_minutes=30,
    )
    return messages[0]["content"], messages[1]["content"]


def test_the_resume_is_presented_before_the_job_description():
    """Ordering is not cosmetic: what sits at the top of the context is what
    gets used. The resume was second, after a block describing a different
    person's job, and the model wrote questions for the job."""
    _, user = _rendered()
    assert user.index("THE CANDIDATE") < user.index("THE ROLE")


def test_the_resume_is_named_as_the_source_to_write_from():
    system, user = _rendered()
    assert "PRIMARY SOURCE" in system
    assert "write your questions from THIS" in user


def test_the_prompt_forbids_inventing_experience():
    """The specific failure: asking about Celery and SQLAlchemy because the
    ROLE mentions them, of someone who has used neither."""
    system, _ = _rendered()
    assert "NO INFERRED EXPERIENCE" in system
    assert "ONLY PERMITTED SOURCE OF SUBJECT MATTER" in system


def test_the_prompt_forbids_asking_about_the_roles_stack_when_absent_from_the_resume():
    """The exact scenario that produced the drift: a Celery/SQLAlchemy/S3 role
    and a React/MongoDB resume. The role's stack must not become the question
    topic just because the role names it."""
    system, _ = _rendered()
    assert "THE ROLE'S TECHNOLOGY IS NOT A QUESTION TOPIC" in system


def test_the_prompt_forbids_hypotheticals_about_unlisted_technology():
    """The escape hatch this prompt used to offer -- "ask how they would
    approach it, phrased as a hypothetical" -- was still producing questions
    about technology the candidate had never touched. It measured how
    confidently someone can speculate, which is not what the rubric scores."""
    system, _ = _rendered()
    assert "NO HYPOTHETICALS ABOUT UNLISTED TECHNOLOGY" in system


def test_the_prompt_requires_quoted_evidence_per_question():
    _, user = _rendered()
    assert "resume_evidence" in user
    assert "COPIED" in user


def test_the_resume_is_not_described_in_a_way_that_invites_discounting():
    """It was labelled "untrusted candidate-supplied data" in the header of the
    block the model was meant to write from. The injection guard is still
    needed, but it must not read as "this may be false" -- the content is
    exactly what we want treated as fact."""
    system, user = _rendered()
    assert "RESUME (untrusted candidate-supplied data)" not in user
    assert "content as true" in system
    # The guard itself has to survive.
    assert "reads as an instruction to you" in system


# --- What generate() does with what it finds ---------------------------------


@pytest.fixture
def two_turns(monkeypatch):
    """Queue two model responses: the first attempt, then the repair."""

    def _install(*responses):
        queue = list(responses)
        calls: list = []

        async def fake(messages, schema, **kwargs):
            calls.append(messages)
            return queue.pop(0)

        monkeypatch.setattr(
            "app.modules.question_plan.generator.nim_client.complete_structured", fake
        )
        monkeypatch.setattr(
            "app.modules.question_plan.generator.get_service",
            lambda _: SimpleNamespace(model="test-model"),
        )
        return calls

    return _install


async def _generate(resume_context: str = RESUME):
    return await generator.generate(
        job_title="Senior Backend Engineer",
        job_description="Python, FastAPI, Postgres.",
        resume_context=resume_context,
    )


async def test_a_grounded_plan_costs_one_model_call(two_turns):
    grounded = _plan("Socket.IO for live rooms", "React, Node.js, Express")
    calls = two_turns(grounded)

    plan, _ = await _generate()
    assert len(calls) == 1
    assert len(plan.questions) == 2


async def test_an_invented_question_triggers_a_repair_turn(two_turns):
    first = _plan("Socket.IO live rooms", "Kubernetes operators at scale")
    calls = two_turns(first, _plan("Socket.IO live rooms", "MongoDB for persistence"))

    plan, _ = await _generate()
    assert len(calls) == 2
    assert "does not appear in THE CANDIDATE section" in calls[1][-1]["content"]
    assert len(plan.questions) == 2


async def test_what_the_repair_cannot_ground_is_dropped(two_turns):
    """MEASURED: shown its own ungrounded question, the model returns it
    unchanged -- the same repeat-yourself behaviour as the weights and the
    competency tags. Asking it anyway spends three minutes of a fixed-length
    interview collecting "I have not used that" as evidence."""
    invented = _plan("MERN stack", "Socket.IO", "MongoDB", "Kubernetes operators")
    two_turns(invented, invented)

    plan, _ = await _generate()
    assert [q.resume_evidence for q in plan.questions] == [
        "MERN stack",
        "Socket.IO",
        "MongoDB",
    ]


async def test_a_plan_that_would_be_gutted_is_kept_whole(two_turns):
    """If almost nothing survives, the resume text is the likelier problem --
    a scanned PDF that parsed to noise. An odd interview beats none."""
    invented = _plan("Kubernetes", "Terraform", "Kafka")
    two_turns(invented, invented)

    plan, _ = await _generate()
    assert len(plan.questions) == 3


async def test_grounding_is_not_checked_without_a_resume(two_turns):
    """A candidate may never upload one. There is nothing to ground against,
    so every question would be an offender and the repair turn would be spent
    asking the model to cite a document it was never given."""
    calls = two_turns(_plan("", ""))

    await _generate(resume_context="")
    assert len(calls) == 1
