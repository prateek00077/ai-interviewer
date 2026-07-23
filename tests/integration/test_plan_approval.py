"""What recruiter approval means, and what it does not.

An interview starts whether or not the plan was approved. That is deliberate:
the candidate is already on the call with their microphone open, and refusing to
begin punishes the one person who did nothing wrong. See
``interview/service.start``.

The defect was not that it starts. It was that approval left no trace: ``freeze``
overwrites ``status`` with FROZEN, so afterwards nothing anywhere recorded
whether a human had ever looked at the questions -- and ``PlanStatus.APPROVED``
was written by one line and read by none.
"""

import uuid

import pytest

from app.db.session import tenant_session
from app.models.question_plan import PlanStatus
from app.modules.interview import service as interview_service
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
            _criterion("depth", "0.5"),
            _criterion("comms", "0.3"),
            _criterion("ownership", "0.2"),
        ],
        "questions": [{"body": "Walk me through the migration.", "competency": "depth"}],
    }
)


@pytest.fixture
async def interview(api_client, registered_org):
    response = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={"candidate_email": f"cand-{uuid.uuid4().hex[:8]}@example.com"},
    )
    return response.json()


async def _populate(org_id: uuid.UUID, interview_id: uuid.UUID) -> uuid.UUID:
    async with tenant_session(org_id, "system", None) as s:
        plan = await plan_service.ensure_plan(s, org_id=org_id, interview_id=interview_id)
        await plan_service.apply_generated(
            s, plan=plan, generated=GENERATED, model_name="test-model"
        )
        return plan.id


# --- Approval leaves a trace ------------------------------------------------


async def test_an_unapproved_plan_has_no_approved_at(registered_org, interview):
    org_id = uuid.UUID(registered_org["org_id"])
    await _populate(org_id, uuid.UUID(interview["interview_id"]))

    async with tenant_session(org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, uuid.UUID(interview["interview_id"]))
    assert plan.approved_at is None


async def test_approving_records_when(api_client, registered_org, interview):
    org_id = uuid.UUID(registered_org["org_id"])
    await _populate(org_id, uuid.UUID(interview["interview_id"]))

    response = await api_client.post(
        f"/api/v1/interviews/{interview['interview_id']}/plan/approve",
        headers=_auth(registered_org),
        json={},
    )
    assert response.status_code == 200, response.text
    assert response.json()["approved_at"] is not None


async def test_the_approval_survives_freezing(registered_org, interview):
    """The whole point. ``freeze`` overwrites ``status``, so a timestamp is the
    only thing that can still answer "did anyone review this?" afterwards."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(interview["interview_id"])
    await _populate(org_id, interview_id)

    async with tenant_session(org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
        await plan_service.approve(s, plan=plan)

    async with tenant_session(org_id, "system", None) as s:
        await interview_service.start(s, interview_id)

    async with tenant_session(org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
    assert plan.status is PlanStatus.FROZEN, "the interview did not freeze the plan"
    assert plan.approved_at is not None, "freezing erased the fact that it was reviewed"


# --- Starting is still allowed ----------------------------------------------


async def test_an_interview_starts_without_approval(registered_org, interview):
    """Deliberate. The candidate is already on the call; refusing to begin
    punishes the one person who did nothing wrong."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(interview["interview_id"])
    await _populate(org_id, interview_id)

    async with tenant_session(org_id, "system", None) as s:
        started = await interview_service.start(s, interview_id)
    assert started.status.value == "IN_PROGRESS"

    async with tenant_session(org_id, "system", None) as s:
        plan = await plan_service.get_for_interview(s, interview_id)
    assert plan.status is PlanStatus.FROZEN
    assert plan.approved_at is None, "an unreviewed plan must not look reviewed"


async def test_the_report_warns_when_nobody_reviewed_the_plan():
    """A recruiter reading a score should not have to go looking for this."""
    from decimal import Decimal

    from app.modules.reports import renderer
    from app.modules.reports.recruiter import CriterionView, RecruiterView

    view = RecruiterView(
        candidate_name="Ada",
        candidate_email="ada@example.com",
        job_title="Staff Engineer",
        interview_id=uuid.uuid4(),
        status="COMPLETED",
        overall=Decimal("3.60"),
        recommendation="HIRE",
        plan_approved_at=None,
        criteria=[
            CriterionView(
                name="depth",
                weight=Decimal("1.0"),
                score=Decimal("4"),
                rationale="r",
                evidence=[],
            )
        ],
    )
    html = renderer.render_html(renderer.RECRUITER_TEMPLATE, view)
    assert "No recruiter reviewed this plan" in html
    # Before the criteria, so the caveat reaches the reader before the numbers do.
    assert html.index("No recruiter reviewed") < html.index("<h2>Criteria")


async def test_the_report_stays_quiet_when_it_was_reviewed():
    from datetime import UTC, datetime
    from decimal import Decimal

    from app.modules.reports import renderer
    from app.modules.reports.recruiter import RecruiterView

    view = RecruiterView(
        candidate_name="Ada",
        candidate_email="ada@example.com",
        job_title="Staff Engineer",
        interview_id=uuid.uuid4(),
        status="COMPLETED",
        overall=Decimal("3.60"),
        recommendation="HIRE",
        plan_approved_at=datetime.now(UTC),
    )
    html = renderer.render_html(renderer.RECRUITER_TEMPLATE, view)
    assert "No recruiter reviewed this plan" not in html
