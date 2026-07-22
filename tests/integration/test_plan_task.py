"""The plan generation worker path, with the model stubbed.

This file exists because of a specific recurring bug. Every failure so far in
the worker has been the same shape: an ORM relationship that is loaded in the
API process (because some earlier query happened to populate it) but unloaded in
the worker, where touching it is a lazy load -- and a lazy load under asyncio is
a MissingGreenlet, not a query.

The unit tests cannot catch it and the live-model test does not exercise the
context gathering. So this drives the real task function against a real database
and asserts it completes.
"""

import uuid

import pytest

from app.models.interview import Interview
from app.models.job import Job, JobDescription
from app.models.question_plan import PlanGenerationStatus
from app.modules.question_plan import service as plan_service
from app.modules.question_plan.generator import GeneratedPlan
from app.workers.tasks import plan_tasks

pytestmark = pytest.mark.integration


GENERATED = GeneratedPlan.model_validate(
    {
        "criteria": [
            {
                "name": "depth",
                "weight": "0.6",
                "descriptors": {"1": "weak", "3": "ok", "5": "strong"},
            },
            {
                "name": "comms",
                "weight": "0.2",
                "descriptors": {"1": "weak", "3": "ok", "5": "strong"},
            },
            {
                "name": "ownership",
                "weight": "0.2",
                "descriptors": {"1": "weak", "3": "ok", "5": "strong"},
            },
        ],
        "questions": [{"body": "Walk me through the migration.", "competency": "depth"}],
    }
)


@pytest.fixture
def stub_model(monkeypatch):
    """Capture what the generator was asked, without spending an API call."""
    seen: dict = {}

    async def _fake(**kwargs):
        seen.update(kwargs)
        return GENERATED, "stub-model"

    monkeypatch.setattr(plan_tasks.generator, "generate", _fake)
    return seen


async def _seed(tenant_session, org, *, with_job: bool) -> uuid.UUID:
    """An interview, optionally attached to a job with an active description."""
    interview_id = uuid.uuid4()
    async with tenant_session(org.org_id, "user", org.user_id) as s:
        job_id = None
        if with_job:
            job = Job(org_id=org.org_id, title="Staff Backend Engineer")
            s.add(job)
            await s.flush()
            s.add(
                JobDescription(
                    org_id=org.org_id,
                    job_id=job.id,
                    content="Own the payments ledger and the Kafka event bus.",
                    is_active=True,
                )
            )
            job_id = job.id

        s.add(
            Interview(
                id=interview_id,
                org_id=org.org_id,
                candidate_id=org.candidate_id,
                job_id=job_id,
            )
        )
        await s.flush()

    async with tenant_session(org.org_id, "system", None) as s:
        await plan_service.ensure_plan(s, org_id=org.org_id, interview_id=interview_id)
    return interview_id


async def test_the_worker_generates_a_plan_end_to_end(tenant_session, org_a, stub_model):
    """The regression test for the lazy-relationship MissingGreenlet."""
    interview_id = await _seed(tenant_session, org_a, with_job=True)

    result = await plan_tasks._generate(
        org_a.org_id, interview_id, question_count=4, duration_minutes=20
    )
    assert result["questions"] == 1
    assert result["criteria"] == 3

    async with tenant_session(org_a.org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
        assert plan.generation_status is PlanGenerationStatus.READY
        assert plan.generated_by == "stub-model"
        assert [q.body for q in plan.questions] == ["Walk me through the migration."]
        assert {c.name for c in plan.criteria} == {"depth", "comms", "ownership"}


async def test_the_job_description_reaches_the_generator(tenant_session, org_a, stub_model):
    interview_id = await _seed(tenant_session, org_a, with_job=True)
    await plan_tasks._generate(
        org_a.org_id, interview_id, question_count=4, duration_minutes=20
    )

    assert stub_model["job_title"] == "Staff Backend Engineer"
    assert "payments ledger" in stub_model["job_description"]


async def test_the_provenance_link_is_recorded(tenant_session, org_a, stub_model):
    """Which description version the plan was derived from is worth keeping."""
    interview_id = await _seed(tenant_session, org_a, with_job=True)
    await plan_tasks._generate(
        org_a.org_id, interview_id, question_count=4, duration_minutes=20
    )

    async with tenant_session(org_a.org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
        assert plan.job_description_id is not None


async def test_an_interview_with_no_job_still_gets_a_plan(tenant_session, org_a, stub_model):
    """A recruiter who skipped the job setup gets a generic plan, not a failure."""
    interview_id = await _seed(tenant_session, org_a, with_job=False)

    result = await plan_tasks._generate(
        org_a.org_id, interview_id, question_count=4, duration_minutes=20
    )
    assert result["questions"] == 1
    assert stub_model["job_description"].startswith("(no job description")
    # No resume uploaded either.
    assert stub_model["resume_context"] == ""


async def test_a_second_run_does_not_regenerate(tenant_session, org_a, stub_model):
    """Celery is at-least-once; a duplicate delivery must not spend a generation."""
    interview_id = await _seed(tenant_session, org_a, with_job=True)
    await plan_tasks._generate(
        org_a.org_id, interview_id, question_count=4, duration_minutes=20
    )

    stub_model.clear()
    result = await plan_tasks._generate(
        org_a.org_id, interview_id, question_count=4, duration_minutes=20
    )
    assert result["skipped"] == "already generated"
    assert stub_model == {}, "the model was called again on a duplicate delivery"


async def test_a_model_failure_marks_the_plan_and_re_raises(
    tenant_session, org_a, monkeypatch
):
    """FAILED with a reason beats a plan that is silently never populated, and
    the raise is what lets Celery retry a transient model outage."""
    interview_id = await _seed(tenant_session, org_a, with_job=True)

    async def _boom(**_):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(plan_tasks.generator, "generate", _boom)

    with pytest.raises(RuntimeError, match="model exploded"):
        await plan_tasks._generate(
            org_a.org_id, interview_id, question_count=4, duration_minutes=20
        )

    async with tenant_session(org_a.org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
        assert plan.generation_status is PlanGenerationStatus.FAILED
        assert "model exploded" in plan.error


async def test_a_frozen_plan_is_never_regenerated(tenant_session, org_a, stub_model):
    interview_id = await _seed(tenant_session, org_a, with_job=True)
    async with tenant_session(org_a.org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
        await plan_service.apply_generated(
            s, plan=plan, generated=GENERATED, model_name="original"
        )
        await plan_service.freeze(s, plan=await plan_service.get_plan(s, plan.id))

    result = await plan_tasks._generate(
        org_a.org_id, interview_id, question_count=4, duration_minutes=20
    )
    assert result["skipped"] in ("frozen", "already generated")

    async with tenant_session(org_a.org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
        assert plan.generated_by == "original"
