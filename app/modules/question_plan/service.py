"""Persist, version, and apply recruiter edits to plans.

Two invariants everything here defends:

FROZEN IS PERMANENT. A plan is frozen when the interview starts, and from then
on the questions and weights are the record of what the candidate was actually
assessed against. If they could still move, a score would mean nothing and a
rejected candidate asking "what was I measured on" could not be given a
truthful answer. Every mutating function checks this first.

WEIGHTS SUM TO 1.0. The scorer takes a weighted mean. A recruiter editing one
weight without adjusting the others would silently skew every score produced
afterwards, so an edit that breaks the sum is rejected rather than normalised --
unlike model output, a human edit is deliberate and should be corrected by the
human who made it.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.exceptions import ConflictError, NotFoundError
from app.models.question_plan import (
    WEIGHT_TOLERANCE,
    PlanGenerationStatus,
    PlannedQuestion,
    PlanStatus,
    QuestionPlan,
    RubricCriterion,
)
from app.modules.question_plan.generator import GeneratedPlan

log = structlog.get_logger(__name__)


class PlanFrozenError(ConflictError):
    code = "plan_frozen"
    message = "This plan is frozen and can no longer be edited."


class PlanVersionConflictError(ConflictError):
    code = "plan_version_conflict"
    message = "The plan changed since you loaded it. Reload and reapply your edit."


# --- Reads ------------------------------------------------------------------


def _with_contents(stmt):
    """selectinload the children, and overwrite whatever is already cached.

    Two separate reasons, both load-bearing:

    - selectinload rather than lazy access, because these collections are read
      by the API and by the worker, and a lazy load outside SQLAlchemy's
      greenlet is a MissingGreenlet rather than a query.
    - populate_existing, because the edit paths clear children with a bulk
      DELETE, which does not touch the identity map. Without it a re-read after
      an edit returns the *cached* collection and the caller sees the rows it
      just deleted.
    """
    return stmt.options(
        selectinload(QuestionPlan.questions), selectinload(QuestionPlan.criteria)
    ).execution_options(populate_existing=True)


async def get_plan(session: AsyncSession, plan_id: uuid.UUID) -> QuestionPlan:
    """A plan with its questions and criteria freshly loaded."""
    stmt = _with_contents(select(QuestionPlan).where(QuestionPlan.id == plan_id))
    plan = (await session.execute(stmt)).scalar_one_or_none()
    if plan is None:
        raise NotFoundError("Question plan not found.", plan_id=str(plan_id))
    return plan


async def get_for_interview(
    session: AsyncSession, interview_id: uuid.UUID
) -> QuestionPlan | None:
    return (
        await session.execute(
            _with_contents(select(QuestionPlan).where(QuestionPlan.interview_id == interview_id))
        )
    ).scalar_one_or_none()


# --- Creation and generation ------------------------------------------------


async def ensure_plan(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    job_description_id: uuid.UUID | None = None,
    resume_id: uuid.UUID | None = None,
) -> QuestionPlan:
    """Get or create the PENDING shell for an interview.

    Creating the row before generation starts means the recruiter can see that a
    plan is coming, and a generation that fails leaves a FAILED row explaining
    why rather than nothing at all.
    """
    existing = await get_for_interview(session, interview_id)
    if existing is not None:
        return existing

    plan = QuestionPlan(
        org_id=org_id,
        interview_id=interview_id,
        job_description_id=job_description_id,
        resume_id=resume_id,
    )
    session.add(plan)
    await session.flush()
    log.info("plan_created", plan_id=str(plan.id), interview_id=str(interview_id))
    return plan


async def apply_generated(
    session: AsyncSession,
    *,
    plan: QuestionPlan,
    generated: GeneratedPlan,
    model_name: str,
) -> QuestionPlan:
    """Replace a plan's contents with a fresh generation.

    Replace, not append: regenerating is "try again", and merging two attempts
    would produce a rubric whose weights no longer sum to 1.0 and a question
    list with duplicates.
    """
    if not plan.is_editable:
        raise PlanFrozenError()

    # Read these before clearing: bulk DELETE leaves any loaded collection on
    # `plan` holding rows that no longer exist, and the obvious fix --
    # expire_all() -- expires `plan` itself, making the very next attribute read
    # a lazy load and therefore a MissingGreenlet.
    plan_id, org_id = plan.id, plan.org_id

    # Bulk DELETE rather than iterating plan.questions/plan.criteria. A plan that
    # was just created has those relationships unloaded, and touching them
    # lazy-loads -- which under asyncio is a MissingGreenlet, not a query.
    await _clear_contents(session, plan_id)

    for ordinal, criterion in enumerate(generated.criteria):
        session.add(
            RubricCriterion(
                org_id=org_id,
                plan_id=plan_id,
                ordinal=ordinal,
                name=criterion.name.strip(),
                description=criterion.description,
                weight=criterion.weight,
                descriptors=criterion.descriptors,
            )
        )
    for ordinal, question in enumerate(generated.questions):
        session.add(
            PlannedQuestion(
                org_id=org_id,
                plan_id=plan_id,
                ordinal=ordinal,
                body=question.body.strip(),
                competency=question.competency.strip() if question.competency else None,
                follow_up_hints=question.follow_up_hints,
                time_budget_secs=question.time_budget_secs,
            )
        )

    plan.generation_status = PlanGenerationStatus.READY
    plan.status = PlanStatus.DRAFT
    plan.generated_by = model_name
    plan.error = None
    plan.version += 1
    await session.flush()

    log.info(
        "plan_populated",
        plan_id=str(plan.id),
        questions=len(generated.questions),
        criteria=len(generated.criteria),
    )
    return plan


async def _clear_contents(session: AsyncSession, plan_id: uuid.UUID) -> None:
    """Drop a plan's questions and criteria without loading them.

    Callers must re-read through ``get_plan`` afterwards rather than trusting an
    already-loaded ``plan.questions``: a bulk DELETE does not update the
    identity map, so a stale collection would still list the deleted rows.
    """
    await session.execute(delete(PlannedQuestion).where(PlannedQuestion.plan_id == plan_id))
    await session.execute(delete(RubricCriterion).where(RubricCriterion.plan_id == plan_id))
    await session.flush()


async def mark_failed(session: AsyncSession, *, plan: QuestionPlan, error: str) -> None:
    plan.generation_status = PlanGenerationStatus.FAILED
    plan.error = error[:2000]
    await session.flush()


# --- Recruiter edits --------------------------------------------------------


def _guard(plan: QuestionPlan, expected_version: int | None) -> None:
    """Frozen check plus optimistic concurrency, in that order.

    Version is optional so a single-user flow does not have to thread it
    through, but a client that sends it gets protection from a concurrent edit.
    """
    if not plan.is_editable:
        raise PlanFrozenError()
    if expected_version is not None and expected_version != plan.version:
        raise PlanVersionConflictError(
            expected=expected_version, actual=plan.version
        )


async def replace_questions(
    session: AsyncSession,
    *,
    plan: QuestionPlan,
    questions: list[dict],
    expected_version: int | None = None,
) -> QuestionPlan:
    """Overwrite the question list wholesale.

    Wholesale rather than per-question PATCH because ordering is part of the
    edit: a recruiter reordering, deleting and adding in one pass is a single
    intent, and expressing it as a sequence of individual operations would leave
    the plan briefly inconsistent between them.
    """
    _guard(plan, expected_version)

    valid_names = {c.name for c in plan.criteria}
    for question in questions:
        competency = (question.get("competency") or "").strip()
        if competency and competency not in valid_names:
            raise ConflictError(
                f"Question references unknown criterion {competency!r}.",
                valid=sorted(valid_names),
            )

    await session.execute(delete(PlannedQuestion).where(PlannedQuestion.plan_id == plan.id))
    await session.flush()

    for ordinal, question in enumerate(questions):
        session.add(
            PlannedQuestion(
                org_id=plan.org_id,
                plan_id=plan.id,
                ordinal=ordinal,
                body=question["body"].strip(),
                competency=(question.get("competency") or "").strip() or None,
                follow_up_hints=question.get("follow_up_hints") or [],
                time_budget_secs=question.get("time_budget_secs") or 180,
            )
        )

    plan.version += 1
    await session.flush()
    return await get_plan(session, plan.id)


async def replace_criteria(
    session: AsyncSession,
    *,
    plan: QuestionPlan,
    criteria: list[dict],
    expected_version: int | None = None,
) -> QuestionPlan:
    """Overwrite the rubric. Weights must sum to 1.0 exactly."""
    _guard(plan, expected_version)

    total = sum((Decimal(str(c["weight"])) for c in criteria), Decimal(0))
    if abs(total - Decimal(1)) > WEIGHT_TOLERANCE:
        # Not normalised, unlike model output: a human edit is deliberate, and
        # silently rescaling it would change weights the recruiter chose.
        raise ConflictError(f"Criterion weights must sum to 1.0, got {total}.")

    names = [c["name"].strip() for c in criteria]
    if len(set(names)) != len(names):
        raise ConflictError("Criterion names must be unique.")

    # Questions point at criteria by name, so a rename or deletion would orphan
    # them. Clearing the link is honest -- the question survives, ungraded --
    # and the recruiter can retag it.
    surviving = set(names)
    for question in plan.questions:
        if question.competency and question.competency not in surviving:
            question.competency = None

    await session.execute(delete(RubricCriterion).where(RubricCriterion.plan_id == plan.id))
    await session.flush()

    for ordinal, criterion in enumerate(criteria):
        session.add(
            RubricCriterion(
                org_id=plan.org_id,
                plan_id=plan.id,
                ordinal=ordinal,
                name=criterion["name"].strip(),
                description=criterion.get("description"),
                weight=Decimal(str(criterion["weight"])),
                descriptors=criterion.get("descriptors") or {},
            )
        )

    plan.version += 1
    await session.flush()
    return await get_plan(session, plan.id)


# --- Lifecycle --------------------------------------------------------------


async def approve(
    session: AsyncSession, *, plan: QuestionPlan, expected_version: int | None = None
) -> QuestionPlan:
    """Recruiter sign-off. Still editable afterwards -- approval is a statement
    that the plan is good enough to interview with, not a lock."""
    _guard(plan, expected_version)
    if plan.generation_status is not PlanGenerationStatus.READY:
        raise ConflictError("Cannot approve a plan that has not been generated.")
    if not plan.questions or not plan.criteria:
        raise ConflictError("Cannot approve a plan with no questions or no rubric.")

    plan.status = PlanStatus.APPROVED
    plan.version += 1
    await session.flush()
    log.info("plan_approved", plan_id=str(plan.id))
    return plan


async def freeze(session: AsyncSession, *, plan: QuestionPlan) -> QuestionPlan:
    """Called when the interview starts. Irreversible, and idempotent.

    No version check: this is the system acting at session start, not a
    recruiter racing another recruiter.
    """
    if plan.status is PlanStatus.FROZEN:
        return plan
    if not plan.questions or not plan.criteria:
        raise ConflictError("Cannot start an interview against an empty plan.")

    plan.status = PlanStatus.FROZEN
    plan.version += 1
    await session.flush()
    log.info("plan_frozen", plan_id=str(plan.id), version=plan.version)
    return plan
