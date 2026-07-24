"""When a plan generation may be skipped, and when it must not be.

De-duplication and regeneration pull in opposite directions, and getting the
balance wrong is silent either way: skip too eagerly and the interview runs on
stale questions, skip too rarely and two generations race to write the same
rows.

OBSERVED: the invite starts a generation, the candidate uploads their CV thirty
seconds later, the resume task correctly triggers a replan -- and the guard
dropped it as a duplicate. The interview then asked questions written from the
job description alone, which is precisely what ingesting a resume exists to
avoid.
"""

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.models.question_plan import PlanGenerationStatus, PlanStatus
from app.modules.question_plan import service as plan_service
from app.workers.tasks import plan_tasks


class _FakePlan:
    def __init__(self, gen: PlanGenerationStatus, status=PlanStatus.DRAFT, age_secs=1):
        self.id = uuid.uuid4()
        self.generation_status = gen
        self.status = status
        self.updated_at = datetime.now(UTC) - timedelta(seconds=age_secs)
        self.error = None

    @property
    def is_editable(self) -> bool:
        return self.status is not PlanStatus.FROZEN


@pytest.fixture
def plan_lookup(monkeypatch):
    """Stub the session, the plan read, and the resume-readiness lookups; only
    the guard logic is under test.

    ``holder`` controls the scenario: ``plan`` is the row the task reads;
    ``ready_resume`` and ``incoming`` drive the resume-wait guard. By default no
    resume is ready and none is on its way, so generation proceeds -- the tests
    that care about waiting set ``incoming``.
    """
    holder: dict = {}

    class _Session:
        async def flush(self): ...

    class _Ctx:
        async def __aenter__(self): return _Session()
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(plan_tasks, "tenant_session", lambda *a, **k: _Ctx())

    async def _get(_session, _interview_id):
        return holder.get("plan")

    monkeypatch.setattr(plan_tasks.plan_service, "get_for_interview", _get)

    async def _interview(_session, interview_id):
        return SimpleNamespace(id=interview_id, candidate_id=uuid.uuid4())

    monkeypatch.setattr(plan_tasks.interview_service, "get_interview", _interview)

    async def _ready(_session, _candidate_id):
        return holder.get("ready_resume")

    monkeypatch.setattr(plan_tasks.retriever, "latest_ready_resume", _ready)

    async def _incoming(_session, _candidate_id):
        return holder.get("incoming", False)

    monkeypatch.setattr(plan_tasks.retriever, "has_incoming_resume", _incoming)
    return holder


async def _run(force: bool, wait_for_resume: bool = True) -> dict:
    return await plan_tasks._generate(
        uuid.uuid4(),
        uuid.uuid4(),
        question_count=8,
        duration_minutes=30,
        force=force,
        wait_for_resume=wait_for_resume,
    )


async def test_a_duplicate_delivery_is_skipped(plan_lookup):
    """Two clicks on Generate, or a redelivered task. Nothing new to say."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.GENERATING)
    assert (await _run(force=False))["skipped"] == "already generating"


async def test_a_forced_replan_waits_instead_of_being_dropped(plan_lookup):
    """THE BUG. A replan carrying a resume the first pass never saw must not be
    discarded because that first pass happens to still be running."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.GENERATING)
    result = await _run(force=True)
    assert "skipped" not in result
    assert result["retry_after"] > 0


async def test_a_ready_plan_is_not_regenerated_by_a_duplicate(plan_lookup):
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.READY)
    assert (await _run(force=False))["skipped"] == "already generated"


async def test_a_ready_plan_IS_regenerated_when_forced(plan_lookup, monkeypatch):
    """A resume landing after generation finished is the common case -- the
    upload takes longer than the model does."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.READY)

    reached = {}

    async def _gather(_session, _plan, _interview_id):
        reached["yes"] = True
        raise RuntimeError("stop here; the guard let us through")

    monkeypatch.setattr(plan_tasks, "_gather_context", _gather)
    with pytest.raises(RuntimeError):
        await _run(force=True)
    assert reached.get("yes"), "a forced regeneration was skipped as already generated"


@pytest.mark.parametrize("force", [True, False])
async def test_a_frozen_plan_is_never_touched(plan_lookup, force):
    """An interview in flight is being conducted against these questions.
    Not even a forced replan may move them."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.READY, status=PlanStatus.FROZEN)
    assert (await _run(force=force))["skipped"] == "frozen"


async def test_a_stale_generation_is_restarted(plan_lookup, monkeypatch):
    """A worker killed mid-generation must not lock the plan forever."""
    plan_lookup["plan"] = _FakePlan(
        PlanGenerationStatus.GENERATING,
        age_secs=plan_tasks.GENERATION_STALE_AFTER_SECS + 60,
    )

    async def _gather(_session, _plan, _interview_id):
        raise RuntimeError("stop here; the guard let us through")

    monkeypatch.setattr(plan_tasks, "_gather_context", _gather)
    with pytest.raises(RuntimeError):
        await _run(force=False)


# --- Waiting for a resume that is still landing -----------------------------


async def test_generation_waits_while_the_resume_is_still_parsing(plan_lookup):
    """THE "random questions first" bug. A generation that runs while the CV is
    still parsing writes questions from the job description alone. It must wait
    instead, so the first plan the recruiter sees is already about the candidate."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.PENDING)
    plan_lookup["ready_resume"] = None
    plan_lookup["incoming"] = True

    result = await _run(force=True)
    assert "skipped" not in result
    assert result["retry_after"] > 0


async def test_generation_proceeds_once_the_resume_is_ready(plan_lookup, monkeypatch):
    """A ready resume is the whole point of waiting -- once it exists, generate."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.PENDING)
    plan_lookup["ready_resume"] = SimpleNamespace(id=uuid.uuid4())
    plan_lookup["incoming"] = True  # a newer upload in flight must not block it

    async def _gather(_session, _plan, _interview_id):
        raise RuntimeError("stop here; the guard let us through")

    monkeypatch.setattr(plan_tasks, "_gather_context", _gather)
    with pytest.raises(RuntimeError):
        await _run(force=True)


async def test_the_final_attempt_generates_without_waiting(plan_lookup, monkeypatch):
    """The wait is bounded. When the retries are spent (wait_for_resume False), a
    plan from the job description alone beats leaving the recruiter with none."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.PENDING)
    plan_lookup["ready_resume"] = None
    plan_lookup["incoming"] = True

    async def _gather(_session, _plan, _interview_id):
        raise RuntimeError("stop here; the guard let us through")

    monkeypatch.setattr(plan_tasks, "_gather_context", _gather)
    with pytest.raises(RuntimeError):
        await _run(force=True, wait_for_resume=False)


async def test_no_incoming_resume_does_not_wait(plan_lookup, monkeypatch):
    """The invite-time path: no CV uploaded yet, nothing to wait for. Generate a
    plan from the job description now; the resume task replans when it lands."""
    plan_lookup["plan"] = _FakePlan(PlanGenerationStatus.PENDING)
    plan_lookup["incoming"] = False

    async def _gather(_session, _plan, _interview_id):
        raise RuntimeError("stop here; the guard let us through")

    monkeypatch.setattr(plan_tasks, "_gather_context", _gather)
    with pytest.raises(RuntimeError):
        await _run(force=True)


# --- The API-side gate: a click must not stack a second generation ----------


def test_generation_in_flight_is_true_only_for_a_fresh_generating_plan():
    fresh = _FakePlan(PlanGenerationStatus.GENERATING, age_secs=1)
    assert plan_service.generation_in_flight(fresh) is True


def test_generation_in_flight_is_false_once_the_plan_is_ready():
    assert plan_service.generation_in_flight(_FakePlan(PlanGenerationStatus.READY)) is False


def test_generation_in_flight_is_false_for_a_stale_generation():
    """Past the stale window the worker is presumed dead, so a Generate click is
    allowed to restart it rather than being told to keep polling forever."""
    stale = _FakePlan(
        PlanGenerationStatus.GENERATING,
        age_secs=plan_service.GENERATION_STALE_AFTER_SECS + 60,
    )
    assert plan_service.generation_in_flight(stale) is False


def test_the_resume_task_forces_its_replan():
    """Belt and braces: the call site has to pass force, or none of the above
    matters."""
    import inspect

    source = inspect.getsource(plan_tasks_caller())
    assert "force=True" in source


def plan_tasks_caller():
    from app.workers.tasks import resume_tasks

    return resume_tasks._regenerate_plans_without_this_resume
