"""Scoring persistence and the recruiter-facing score API.

Two things are under test here that the unit tests cannot reach: that a score
survives a round trip through Postgres with its Decimals and JSONB intact, and
that the candidate cannot see any of it -- enforced by RLS, not by the route.
"""

import uuid
from decimal import Decimal

import pytest

from app.db.session import tenant_session
from app.models.question_plan import PlanStatus, RubricCriterion
from app.models.score import Recommendation, ScoringStatus
from app.modules.interview import service as interview_service
from app.modules.interview import transcript
from app.modules.scoring import aggregator
from app.modules.scoring import service as scoring_service
from app.modules.scoring.rubric_scorer import Graded

pytestmark = pytest.mark.integration

D = Decimal


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


@pytest.fixture
async def invited(api_client, registered_org):
    """A job, an interview against it, and a redeemed candidate token."""
    job = await api_client.post(
        "/api/v1/jobs", headers=_auth(registered_org), json={"title": "Staff Engineer"}
    )
    job_id = job.json()["id"]

    invite = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={
            "candidate_email": f"cand-{uuid.uuid4().hex[:8]}@example.com",
            "job_id": job_id,
        },
    )
    body = invite.json()
    redeemed = await api_client.post(
        "/api/v1/auth/invites/redeem", json={"invite_token": body["invite_token"]}
    )
    return {
        **body,
        "job_id": job_id,
        "org_id": registered_org["org_id"],
        "headers": {"Authorization": f"Bearer {redeemed.json()['interview_token']}"},
    }


async def _seed_rubric(org_id: uuid.UUID, interview_id: uuid.UUID) -> list[uuid.UUID]:
    """A frozen two-criterion rubric, as an interview in progress would have.

    ``ensure_plan`` rather than a bare INSERT: creating an invite now opens the
    plan row and enqueues generation, so an interview always has exactly one
    plan by the time a test gets here. Inserting a second is a unique violation
    on ``uq_question_plans_interview_id``.
    """
    from app.modules.question_plan import service as plan_service

    async with tenant_session(org_id, "system", None) as session:
        plan = await plan_service.ensure_plan(
            session, org_id=org_id, interview_id=interview_id
        )
        plan.status = PlanStatus.FROZEN
        await session.flush()
        for ordinal, (name, weight) in enumerate(
            [("System Design", D("0.6")), ("Communication", D("0.4"))]
        ):
            session.add(
                RubricCriterion(
                    org_id=org_id,
                    plan_id=plan.id,
                    ordinal=ordinal,
                    name=name,
                    weight=weight,
                    descriptors={"1": "weak", "3": "adequate", "5": "strong"},
                )
            )
        await session.flush()
        return [plan.id]


async def _store_result(
    org_id: uuid.UUID, interview_id: uuid.UUID, scores: list[Decimal | None]
) -> None:
    """Run the real store path with model output stubbed out."""
    async with tenant_session(org_id, "system", None) as session:
        from app.modules.question_plan import service as plan_service

        plan = await plan_service.get_for_interview(session, interview_id)
        assert plan is not None
        results = [
            (
                criterion,
                Graded(
                    score=value,
                    rationale=f"reasoning for {criterion.name}",
                    evidence=(
                        [{"quote": "we sharded on tenant id", "turn_ordinal": 1, "offset_ms": 6000}]
                        if value is not None
                        else []
                    ),
                ),
            )
            for criterion, value in zip(plan.criteria, scores, strict=True)
        ]
        outcome = aggregator.aggregate(scoring_service.weights_and_scores(results))
        score = await scoring_service.ensure_score(
            session, org_id=org_id, interview_id=interview_id, plan_id=plan.id
        )
        await scoring_service.store(
            session,
            score=score,
            results=results,
            outcome=outcome,
            signals={"filler_count": 3},
            model_name="test-model",
        )


# --- Reading ----------------------------------------------------------------


async def test_an_unscored_interview_is_a_404_not_an_empty_score(
    api_client, registered_org, invited
):
    response = await api_client.get(
        f"/api/v1/interviews/{invited['interview_id']}/score", headers=_auth(registered_org)
    )
    assert response.status_code == 404


async def test_a_pending_score_is_visible_before_the_model_has_run(
    api_client, registered_org, invited
):
    """A recruiter opening the page seconds after the call ends should see
    "in progress", not a 404 that reads like scoring never started."""
    org_id = uuid.UUID(registered_org["org_id"])
    async with tenant_session(org_id, "system", None) as session:
        await scoring_service.ensure_score(
            session, org_id=org_id, interview_id=uuid.UUID(invited["interview_id"])
        )

    body = (
        await api_client.get(
            f"/api/v1/interviews/{invited['interview_id']}/score", headers=_auth(registered_org)
        )
    ).json()
    assert body["status"] == ScoringStatus.PENDING.value
    assert body["overall"] is None
    assert body["criteria"] == []


async def test_the_score_carries_its_criteria_and_their_evidence(
    api_client, registered_org, invited
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [D("4"), D("3")])

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/score", headers=_auth(registered_org)
    )
    assert response.status_code == 200, response.text
    body = response.json()

    # 0.6*4 + 0.4*3 = 3.6
    assert Decimal(body["overall"]) == D("3.60")
    assert body["recommendation"] == Recommendation.HIRE.value
    assert body["status"] == ScoringStatus.READY.value
    assert body["scored_by"] == "test-model"
    assert body["scored_at"] is not None

    assert [c["name"] for c in body["criteria"]] == ["System Design", "Communication"]
    evidence = body["criteria"][0]["evidence"]
    assert evidence and evidence[0]["offset_ms"] == 6000, "evidence lost its timestamp"


async def test_the_signals_travel_with_the_score_but_not_inside_it(
    api_client, registered_org, invited
):
    """Coverage and delivery signals are reported beside the number, never
    folded into it."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [D("4"), D("4")])

    body = (
        await api_client.get(
            f"/api/v1/interviews/{interview_id}/score", headers=_auth(registered_org)
        )
    ).json()
    signals = body["confidence_signals"]
    assert signals["filler_count"] == 3
    assert signals["rubric_coverage"] == 1.0
    assert signals["criteria_graded"] == 2
    # The signals did not move the number: both criteria scored 4.
    assert Decimal(body["overall"]) == D("4.00")


async def test_an_ungraded_criterion_is_reported_as_such_not_as_a_low_score(
    api_client, registered_org, invited
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [D("4"), None])

    body = (
        await api_client.get(
            f"/api/v1/interviews/{interview_id}/score", headers=_auth(registered_org)
        )
    ).json()
    ungraded = body["criteria"][1]
    assert ungraded["score"] is None
    assert ungraded["evidence"] == []
    # The graded 0.6 was renormalised rather than averaged against a zero.
    assert Decimal(body["overall"]) == D("4.00")
    assert body["confidence_signals"]["rubric_coverage"] == 0.6


# --- Idempotency ------------------------------------------------------------


async def test_rescoring_replaces_the_criteria_rather_than_duplicating_them(
    api_client, registered_org, invited
):
    """The scoring task is at-least-once. A redelivery must overwrite."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)

    await _store_result(org_id, interview_id, [D("4"), D("3")])
    first = (
        await api_client.get(
            f"/api/v1/interviews/{interview_id}/score", headers=_auth(registered_org)
        )
    ).json()

    await _store_result(org_id, interview_id, [D("2"), D("2")])
    second = (
        await api_client.get(
            f"/api/v1/interviews/{interview_id}/score", headers=_auth(registered_org)
        )
    ).json()

    assert second["id"] == first["id"], "a second score row was created"
    assert len(second["criteria"]) == 2, "criterion rows accumulated across runs"
    assert Decimal(second["overall"]) == D("2.00")
    assert second["recommendation"] == Recommendation.NO_HIRE.value


# --- Access control ---------------------------------------------------------


async def test_a_candidate_cannot_read_their_own_score(api_client, registered_org, invited):
    """Not a UX choice. The candidate report is feedback and gaps; a number, a
    band or a recommendation must never reach the person being assessed."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [D("4"), D("3")])

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/score", headers=invited["headers"]
    )
    assert response.status_code == 403


async def test_rls_hides_scores_from_a_candidate_session_entirely(registered_org, invited):
    """The route check is the first barrier; this is the one that holds when a
    future endpoint forgets it."""
    from sqlalchemy import select

    from app.models.score import Score

    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [D("4"), D("3")])

    candidate_id = None
    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)
        candidate_id = interview.candidate_id
        assert (await session.execute(select(Score))).scalars().all(), "seed failed"

    async with tenant_session(org_id, "candidate", candidate_id) as session:
        rows = (await session.execute(select(Score))).scalars().all()
    assert rows == [], "a candidate session could read score rows"


async def test_another_org_cannot_read_the_score(api_client, registered_org, invited):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [D("4"), D("3")])

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
    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/score", headers=_auth(rival)
    )
    assert response.status_code == 404


# --- Rescore ----------------------------------------------------------------


async def test_a_live_interview_cannot_be_rescored(api_client, registered_org, invited):
    """Scoring a conversation that is still happening produces a number about
    half an answer."""
    response = await api_client.post(
        f"/api/v1/interviews/{invited['interview_id']}/score/rescore",
        headers=_auth(registered_org),
    )
    assert response.status_code == 409


async def test_rescoring_a_finished_interview_queues_a_run_and_keeps_the_old_score(
    api_client, registered_org, invited, monkeypatch
):
    """The happy path, with the broker stubbed.

    Two things are pinned. The response has to serialise AFTER the explicit
    commit that makes the row visible to the worker -- with the wrong session
    settings that read is a lazy refresh outside the greenlet, which fails as a
    500 rather than as anything legible. And the previous score must survive:
    a recruiter mid-review should not watch the report empty out.
    """
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [D("4"), D("3")])

    await api_client.post(
        f"/api/v1/interviews/{interview_id}/terminate", headers=_auth(registered_org), json={}
    )

    queued: list[tuple] = []
    from app.workers.tasks import scoring_tasks

    monkeypatch.setattr(
        scoring_tasks.score_interview, "delay", lambda *args: queued.append(args)
    )

    response = await api_client.post(
        f"/api/v1/interviews/{interview_id}/score/rescore", headers=_auth(registered_org)
    )
    assert response.status_code == 202, response.text
    assert queued == [(str(org_id), str(interview_id))]

    body = response.json()
    assert Decimal(body["overall"]) == D("3.60"), "the previous score was cleared on request"
    assert len(body["criteria"]) == 2


async def test_a_candidate_cannot_trigger_a_rescore(api_client, invited):
    response = await api_client.post(
        f"/api/v1/interviews/{invited['interview_id']}/score/rescore",
        headers=invited["headers"],
    )
    assert response.status_code == 403


# --- Empty transcripts ------------------------------------------------------


async def test_an_interview_with_no_transcript_is_insufficient_evidence(
    registered_org, invited
):
    """Distinct from NO_HIRE. Nobody assessed this candidate."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_rubric(org_id, interview_id)
    await _store_result(org_id, interview_id, [None, None])

    async with tenant_session(org_id, "system", None) as session:
        score = await scoring_service.require_for_interview(session, interview_id)
        assert score.overall is None
        assert score.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
        assert score.status is ScoringStatus.READY, "an honest gap is not a failed job"


async def test_the_transcript_the_scorer_reads_is_the_stored_one(registered_org, invited):
    """Guards the ordering in the pipeline: the scorer verifies quotes against
    this text, so it has to be the corrected version."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])

    async with tenant_session(org_id, "system", None) as session:
        await transcript.record_turn(
            session,
            org_id=org_id,
            interview_id=interview_id,
            ordinal=0,
            speaker="CANDIDATE",
            content="we sharded on tenant id",
            started_offset_ms=1000,
            ended_offset_ms=6000,
        )

    async with tenant_session(org_id, "system", None) as session:
        rendered = await transcript.as_text(session, interview_id)
    assert "we sharded on tenant id" in rendered
