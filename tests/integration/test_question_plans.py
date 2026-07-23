"""Question plan review, editing and freezing, over HTTP.

Generation itself is stubbed: the model's behaviour is pinned by the unit tests
and by a separate nim-marked test. What matters here is everything around it --
who may read a plan, what a recruiter may change, and that a frozen plan is
genuinely immutable, because a score means nothing if the rubric could still
move after the interview.
"""

import uuid
from decimal import Decimal

import pytest

from app.db.session import tenant_session
from app.models.question_plan import PlanStatus
from app.modules.question_plan import service as plan_service
from app.modules.question_plan.generator import GeneratedPlan

pytestmark = pytest.mark.integration


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


def _criterion(name: str, weight: str) -> dict:
    return {
        "name": name,
        "description": f"measures {name}",
        "weight": weight,
        "descriptors": {"1": "weak", "3": "adequate", "5": "strong"},
    }


GENERATED = GeneratedPlan.model_validate(
    {
        "criteria": [
            _criterion("technical_depth", "0.5"),
            _criterion("communication", "0.3"),
            _criterion("ownership", "0.2"),
        ],
        "questions": [
            {"body": "Walk me through the Kafka migration.", "competency": "technical_depth"},
            {
                "body": "How did you explain that tradeoff to the team?",
                "competency": "communication",
            },
        ],
    }
)


@pytest.fixture
async def interview(api_client, registered_org):
    """An invited interview, created through the public invite flow."""
    response = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={"candidate_email": f"cand-{uuid.uuid4().hex[:8]}@example.com"},
    )
    assert response.status_code == 201, response.text
    return response.json()


@pytest.fixture
async def plan(registered_org, interview):
    """A populated DRAFT plan, written directly -- generation is not under test."""
    org_id = uuid.UUID(registered_org["org_id"])
    async with tenant_session(org_id, "system", None) as s:
        row = await plan_service.ensure_plan(
            s, org_id=org_id, interview_id=uuid.UUID(interview["interview_id"])
        )
        await plan_service.apply_generated(
            s, plan=row, generated=GENERATED, model_name="test-model"
        )
        return row.id


def _url(interview: dict, suffix: str = "") -> str:
    return f"/api/v1/interviews/{interview['interview_id']}/plan{suffix}"


# --- Reading ----------------------------------------------------------------


async def test_a_recruiter_reads_the_plan_with_questions_and_rubric(
    api_client, registered_org, interview, plan
):
    response = await api_client.get(_url(interview), headers=_auth(registered_org))
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["status"] == "DRAFT"
    assert body["generation_status"] == "READY"
    assert body["generated_by"] == "test-model"
    assert [q["competency"] for q in body["questions"]] == [
        "technical_depth",
        "communication",
    ]
    assert sum(Decimal(c["weight"]) for c in body["criteria"]) == Decimal("1")


async def test_a_candidate_token_cannot_read_the_plan(api_client, registered_org, interview, plan):
    """It is the answer key: questions and weights before the interview starts."""
    redeemed = await api_client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": interview["invite_token"]}
    )
    candidate = {"Authorization": f"Bearer {redeemed.json()['interview_token']}"}

    assert (await api_client.get(_url(interview), headers=candidate)).status_code == 403


async def test_another_org_cannot_read_the_plan(api_client, registered_org, interview, plan):
    slug = f"rival-{uuid.uuid4().hex[:10]}"
    rival = (
        await api_client.post(
            "/api/v1/auth/register-org",
            json={
                "org_name": "Rival",
                "slug": slug,
                "admin_email": f"admin@{slug}.example.com",
                "admin_password": "correct-horse-battery-staple",
            },
        )
    ).json()
    assert (await api_client.get(_url(interview), headers=_auth(rival))).status_code == 404


async def test_an_interview_with_no_plan_is_a_404(api_client, registered_org):
    """Scheduled directly rather than invited.

    Inviting now opens the plan row and enqueues generation, so an invited
    interview always has a plan -- the 404 only survives for an interview
    scheduled ahead of time and not yet invited, which is exactly when a
    recruiter would hit it.
    """
    candidate = await api_client.post(
        "/api/v1/candidates",
        headers=_auth(registered_org),
        json={"email": f"noplan-{uuid.uuid4().hex[:8]}@example.com", "full_name": "No Plan"},
    )
    assert candidate.status_code == 201, candidate.text

    scheduled = await api_client.post(
        "/api/v1/interviews",
        headers=_auth(registered_org),
        json={"candidate_id": candidate.json()["id"]},
    )
    assert scheduled.status_code == 201, scheduled.text

    response = await api_client.get(
        f"/api/v1/interviews/{scheduled.json()['id']}/plan", headers=_auth(registered_org)
    )
    assert response.status_code == 404


async def test_inviting_a_candidate_opens_a_plan(api_client, registered_org, interview):
    """The other half: a plan exists from the moment the invite is created, so a
    recruiter sees "generating" rather than a 404 while the model runs."""
    response = await api_client.get(_url(interview), headers=_auth(registered_org))
    assert response.status_code == 200, response.text
    assert response.json()["generation_status"] in ("PENDING", "GENERATING", "READY", "FAILED")


# --- Editing ----------------------------------------------------------------


async def test_replacing_questions_bumps_the_version(
    api_client, registered_org, interview, plan
):
    before = (await api_client.get(_url(interview), headers=_auth(registered_org))).json()

    response = await api_client.put(
        _url(interview, "/questions"),
        headers=_auth(registered_org),
        json={
            "questions": [
                {"body": "A replacement question about systems.", "competency": "ownership"}
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["questions"]) == 1
    assert body["version"] == before["version"] + 1


async def test_a_question_cannot_name_a_criterion_that_does_not_exist(
    api_client, registered_org, interview, plan
):
    response = await api_client.put(
        _url(interview, "/questions"),
        headers=_auth(registered_org),
        json={"questions": [{"body": "A question about nothing real.", "competency": "invented"}]},
    )
    assert response.status_code == 409


async def test_a_rubric_edit_whose_weights_do_not_sum_to_one_is_rejected(
    api_client, registered_org, interview, plan
):
    """Not normalised, unlike model output: a human edit is deliberate, and
    rescaling it would change weights the recruiter chose."""
    response = await api_client.put(
        _url(interview, "/criteria"),
        headers=_auth(registered_org),
        json={"criteria": [_criterion("only_thing", "0.7"), _criterion("other", "0.2")]},
    )
    assert response.status_code == 409
    assert "sum to 1.0" in response.json()["error"]["message"]


async def test_renaming_a_criterion_unlinks_rather_than_orphans_its_questions(
    api_client, registered_org, interview, plan
):
    """The question survives, ungraded, and can be retagged."""
    response = await api_client.put(
        _url(interview, "/criteria"),
        headers=_auth(registered_org),
        json={
            "criteria": [
                _criterion("renamed_depth", "0.5"),
                _criterion("communication", "0.3"),
                _criterion("ownership", "0.2"),
            ]
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()

    competencies = {q["competency"] for q in body["questions"]}
    assert None in competencies, "the question pointing at the old name was not unlinked"
    assert "communication" in competencies, "an unaffected question lost its link"
    # And no question still points at the vanished name.
    assert "technical_depth" not in competencies


async def test_a_stale_expected_version_is_a_conflict(
    api_client, registered_org, interview, plan
):
    """Two recruiters editing the same plan must not silently overwrite."""
    response = await api_client.put(
        _url(interview, "/questions"),
        headers=_auth(registered_org),
        json={
            "questions": [{"body": "An edit made against a stale read."}],
            "expected_version": 999,
        },
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "plan_version_conflict"


# --- Lifecycle --------------------------------------------------------------


async def test_approve_moves_the_plan_to_approved_and_leaves_it_editable(
    api_client, registered_org, interview, plan
):
    approved = await api_client.post(
        _url(interview, "/approve"), headers=_auth(registered_org), json={}
    )
    assert approved.status_code == 200, approved.text
    assert approved.json()["status"] == "APPROVED"

    # Approval says "good enough to interview with", not "locked".
    edit = await api_client.put(
        _url(interview, "/questions"),
        headers=_auth(registered_org),
        json={"questions": [{"body": "A late but legitimate edit to the plan."}]},
    )
    assert edit.status_code == 200


async def test_a_frozen_plan_rejects_every_mutation(
    api_client, registered_org, interview, plan
):
    """The interview has started. What the candidate was assessed against is now
    the record, and a score would mean nothing if it could still move."""
    org_id = uuid.UUID(registered_org["org_id"])
    async with tenant_session(org_id, "system", None) as s:
        await plan_service.freeze(s, plan=await plan_service.get_plan(s, plan))

    headers = _auth(registered_org)
    questions = await api_client.put(
        _url(interview, "/questions"),
        headers=headers,
        json={"questions": [{"body": "An edit after the interview started."}]},
    )
    criteria = await api_client.put(
        _url(interview, "/criteria"),
        headers=headers,
        json={"criteria": [_criterion("rewritten", "1.0")]},
    )
    approve = await api_client.post(_url(interview, "/approve"), headers=headers, json={})
    regenerate = await api_client.post(
        _url(interview, "/generate"), headers=headers, json={}
    )

    for response in (questions, criteria, approve, regenerate):
        assert response.status_code == 409, response.text
        assert response.json()["error"]["code"] == "plan_frozen"

    # And it really is unchanged.
    current = (await api_client.get(_url(interview), headers=headers)).json()
    assert current["status"] == "FROZEN"
    assert len(current["questions"]) == 2


async def test_freezing_is_idempotent(registered_org, interview, plan):
    org_id = uuid.UUID(registered_org["org_id"])
    async with tenant_session(org_id, "system", None) as s:
        first = await plan_service.freeze(s, plan=await plan_service.get_plan(s, plan))
        version = first.version
        again = await plan_service.freeze(s, plan=await plan_service.get_plan(s, plan))
    assert again.status is PlanStatus.FROZEN
    assert again.version == version, "a repeat freeze bumped the version"


# --- Generation dispatch ----------------------------------------------------


async def test_generate_creates_a_pending_shell_and_queues_the_worker(
    api_client, registered_org, interview, monkeypatch
):
    """202 with a row to poll, rather than a job id to correlate."""
    queued: list[tuple] = []
    monkeypatch.setattr(
        "app.api.v1.question_plans.generate_plan.delay",
        lambda *args: queued.append(args),
    )

    response = await api_client.post(
        _url(interview, "/generate"),
        headers=_auth(registered_org),
        json={"question_count": 6, "duration_minutes": 30},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["generation_status"] == "PENDING"
    assert body["questions"] == []

    # The trailing True is ``force``: a recruiter clicking Generate on a plan
    # that already exists means "do it again", not "skip because there is one".
    assert queued == [
        (registered_org["org_id"], interview["interview_id"], 6, 30, True)
    ]


async def test_generate_is_idempotent_about_the_plan_row(
    api_client, registered_org, interview, monkeypatch
):
    """Regenerating replaces contents; it must not create a second plan."""
    monkeypatch.setattr("app.api.v1.question_plans.generate_plan.delay", lambda *args: None)

    first = await api_client.post(
        _url(interview, "/generate"), headers=_auth(registered_org), json={}
    )
    second = await api_client.post(
        _url(interview, "/generate"), headers=_auth(registered_org), json={}
    )
    assert first.json()["id"] == second.json()["id"]


@pytest.fixture
def queued(monkeypatch) -> list:
    """Capture the dispatch instead of reaching the broker."""
    from app.workers.tasks import plan_tasks

    calls: list = []
    monkeypatch.setattr(plan_tasks.generate_plan, "delay", lambda *a, **k: calls.append((a, k)))
    return calls


async def test_generation_accepts_a_request_with_no_body(
    api_client, registered_org, interview, queued
):
    """A bare POST is the obvious call, and every field has a default.

    Requiring a body meant the test console -- and anything else that just
    posted to the URL -- got 422 "body: Field required", while the plan appeared
    anyway from the invite-time generation. That combination reads as a flaky
    endpoint rather than a strict one.
    """
    response = await api_client.post(
        f"{_url(interview)}/generate", headers=_auth(registered_org)
    )
    assert response.status_code == 202, response.text


async def test_generation_still_honours_an_explicit_body(
    api_client, registered_org, interview, queued
):
    response = await api_client.post(
        f"{_url(interview)}/generate",
        headers=_auth(registered_org),
        json={"question_count": 5, "duration_minutes": 20},
    )
    assert response.status_code == 202, response.text
    assert queued and queued[-1][0][2:] == (5, 20, True), queued
