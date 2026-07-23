"""The two report endpoints, and the wall between their audiences.

The unit tests prove the candidate report has no field that could carry a
score. These prove the other half: that a candidate session cannot reach the
recruiter report at all, which is a Postgres policy rather than a route check.
"""

import uuid

import pytest

from app.db.session import tenant_session
from app.models.report import CandidateReport, RecruiterReport, ReportStatus
from app.modules.interview import service as interview_service
from app.modules.reports import service as reports_service

pytestmark = pytest.mark.integration


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


@pytest.fixture
async def invited(api_client, registered_org):
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


async def _seed_reports(org_id: uuid.UUID, interview_id: uuid.UUID) -> None:
    """Both reports, READY, as the pipeline would leave them."""
    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)

        recruiter = await reports_service.ensure_recruiter_report(
            session, org_id=org_id, interview_id=interview_id
        )
        await reports_service.mark_ready(
            session, recruiter, s3_key=f"{org_id}/{interview_id}/recruiter-x.pdf"
        )

        candidate = await reports_service.ensure_candidate_report(
            session,
            org_id=org_id,
            interview_id=interview_id,
            candidate_id=interview.candidate_id,
        )
        candidate.summary = "You explained the sharding decision well."
        candidate.strengths = [{"title": "Tradeoffs", "detail": "You named what it cost."}]
        candidate.growth_areas = [{"title": "Numbers", "detail": "Quantify the impact."}]
        await reports_service.mark_ready(
            session, candidate, s3_key=f"{org_id}/{interview_id}/candidate-x.pdf"
        )


# --- Recruiter --------------------------------------------------------------


async def test_an_ungenerated_report_is_a_404(api_client, registered_org, invited):
    response = await api_client.get(
        f"/api/v1/interviews/{invited['interview_id']}/report", headers=_auth(registered_org)
    )
    assert response.status_code == 404


async def test_a_recruiter_reads_the_report_status(api_client, registered_org, invited):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

    body = (
        await api_client.get(
            f"/api/v1/interviews/{interview_id}/report", headers=_auth(registered_org)
        )
    ).json()
    assert body["status"] == ReportStatus.READY.value
    assert body["generated_at"] is not None


async def test_the_download_returns_a_link_and_never_the_key(
    api_client, registered_org, invited
):
    """A bucket name is guessable and a key is structured, so returning both is
    most of the way to handing out the object."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/report/download", headers=_auth(registered_org)
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["download_url"].startswith("http")
    assert body["expires_in"] > 0
    assert "s3_key" not in body
    assert "recruiter-x.pdf" not in str(body.get("expires_in"))


async def test_a_pending_report_cannot_be_downloaded(api_client, registered_org, invited):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    async with tenant_session(org_id, "system", None) as session:
        await reports_service.ensure_recruiter_report(
            session, org_id=org_id, interview_id=interview_id
        )

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/report/download", headers=_auth(registered_org)
    )
    assert response.status_code == 404


async def test_another_org_cannot_read_the_report(api_client, registered_org, invited):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

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
        f"/api/v1/interviews/{interview_id}/report", headers=_auth(rival)
    )
    assert response.status_code == 404


async def test_a_live_interview_cannot_be_regenerated(api_client, registered_org, invited):
    response = await api_client.post(
        f"/api/v1/interviews/{invited['interview_id']}/report/regenerate",
        headers=_auth(registered_org),
    )
    assert response.status_code == 409


async def test_regenerating_queues_both_audiences(
    api_client, registered_org, invited, monkeypatch
):
    """Both, not just the recruiter's. Letting them drift to different vintages
    is how a candidate receives feedback that contradicts the report the
    recruiter is reading."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)
    await api_client.post(
        f"/api/v1/interviews/{interview_id}/terminate", headers=_auth(registered_org), json={}
    )

    queued: list[str] = []
    from app.workers.tasks import report_tasks

    monkeypatch.setattr(
        report_tasks.render_recruiter_report, "delay", lambda *a: queued.append("recruiter")
    )
    monkeypatch.setattr(
        report_tasks.render_candidate_report, "delay", lambda *a: queued.append("candidate")
    )

    response = await api_client.post(
        f"/api/v1/interviews/{interview_id}/report/regenerate", headers=_auth(registered_org)
    )
    assert response.status_code == 202, response.text
    assert sorted(queued) == ["candidate", "recruiter"]


# --- Candidate --------------------------------------------------------------


async def test_a_candidate_reads_their_own_feedback(api_client, registered_org, invited):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

    response = await api_client.get("/api/v1/reports/me", headers=invited["headers"])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["summary"].startswith("You explained")
    assert body["strengths"][0]["title"] == "Tradeoffs"
    assert body["growth_areas"][0]["title"] == "Numbers"


async def test_the_candidate_response_carries_no_score_field(
    api_client, registered_org, invited
):
    """The serialized body, checked the way a candidate opening devtools would."""
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

    raw = (await api_client.get("/api/v1/reports/me", headers=invited["headers"])).text.lower()
    for word in ("score", "overall", "recommendation", "verdict", "rubric", "weight"):
        assert word not in raw, f"the candidate response leaks {word!r}"


async def test_a_recruiter_token_cannot_use_the_candidate_route(
    api_client, registered_org, invited
):
    response = await api_client.get("/api/v1/reports/me", headers=_auth(registered_org))
    assert response.status_code == 403


async def test_a_candidate_cannot_reach_the_recruiter_report(
    api_client, registered_org, invited
):
    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

    response = await api_client.get(
        f"/api/v1/interviews/{interview_id}/report", headers=invited["headers"]
    )
    assert response.status_code == 403


# --- The wall, at the database ----------------------------------------------


async def test_rls_hides_the_recruiter_report_from_a_candidate_session(
    registered_org, invited
):
    """The route check is the first barrier. This is the one that holds when a
    future endpoint forgets it."""
    from sqlalchemy import select

    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)
        candidate_id = interview.candidate_id
        assert (await session.execute(select(RecruiterReport))).scalars().all(), "seed failed"

    async with tenant_session(org_id, "candidate", candidate_id) as session:
        assert (await session.execute(select(RecruiterReport))).scalars().all() == []
        # ...but their own feedback is readable.
        own = (await session.execute(select(CandidateReport))).scalars().all()
        assert len(own) == 1


async def test_a_candidate_cannot_read_another_candidates_feedback(
    api_client, registered_org, invited
):
    """Same org, different person. The policy narrows on candidate_id."""
    from sqlalchemy import select

    org_id = uuid.UUID(registered_org["org_id"])
    await _seed_reports(org_id, uuid.UUID(invited["interview_id"]))

    other = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={"candidate_email": f"other-{uuid.uuid4().hex[:8]}@example.com"},
    )
    other_candidate_id = uuid.UUID(other.json()["candidate_id"])

    async with tenant_session(org_id, "candidate", other_candidate_id) as session:
        rows = (await session.execute(select(CandidateReport))).scalars().all()
    assert rows == [], "a candidate read someone else's feedback"


async def test_a_candidate_cannot_write_their_own_feedback(registered_org, invited):
    """Read-own, not write-own. A candidate editing their feedback before a
    recruiter reads it would be a small disaster."""
    from sqlalchemy.exc import DBAPIError

    org_id = uuid.UUID(registered_org["org_id"])
    interview_id = uuid.UUID(invited["interview_id"])
    await _seed_reports(org_id, interview_id)

    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)
        candidate_id = interview.candidate_id

    with pytest.raises(DBAPIError):
        async with tenant_session(org_id, "candidate", candidate_id) as session:
            session.add(
                CandidateReport(
                    org_id=org_id,
                    interview_id=interview_id,
                    candidate_id=candidate_id,
                    summary="I did great",
                )
            )
            await session.flush()
