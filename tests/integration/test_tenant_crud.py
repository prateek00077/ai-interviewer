"""Users, candidates and jobs over HTTP, against a live database.

The point of this file is not that CRUD works -- it is that the tenant boundary
holds through the whole HTTP stack. Every resource is created by org A and then
attacked from org B's token, and the expected answer is always 404: a 403 would
confirm the row exists, which is itself a leak.
"""

import uuid

import pytest

pytestmark = pytest.mark.integration


# --- Helpers ----------------------------------------------------------------


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


async def _register(api_client) -> dict:
    """A second tenant, so cross-org attacks have somewhere to come from."""
    slug = f"rival-{uuid.uuid4().hex[:10]}"
    response = await api_client.post(
        "/api/v1/auth/register-org",
        json={
            "org_name": "Rival Corp",
            "slug": slug,
            "admin_email": f"admin@{slug}.example.com",
            "admin_password": "correct-horse-battery-staple",
            "admin_full_name": "Rhea Rival",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _create_candidate(api_client, org: dict, email: str | None = None) -> dict:
    response = await api_client.post(
        "/api/v1/candidates",
        headers=_auth(org),
        json={"email": email or f"cand-{uuid.uuid4().hex[:8]}@example.com", "full_name": "Cass"},
    )
    assert response.status_code == 201, response.text
    return response.json()


async def _create_job(api_client, org: dict, title: str = "Staff Engineer") -> dict:
    response = await api_client.post(
        "/api/v1/jobs",
        headers=_auth(org),
        json={"title": title, "department": "Platform", "employment_type": "FULL_TIME"},
    )
    assert response.status_code == 201, response.text
    return response.json()


# --- Users ------------------------------------------------------------------


async def test_admin_sees_itself_in_the_roster(api_client, registered_org):
    response = await api_client.get("/api/v1/users", headers=_auth(registered_org))
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["items"][0]["id"] == registered_org["user_id"]
    # The omission that matters: no password hash reaches a response.
    assert "hashed_password" not in body["items"][0]


async def test_user_list_is_scoped_to_the_callers_org(api_client, registered_org):
    rival = await _register(api_client)
    response = await api_client.get("/api/v1/users", headers=_auth(rival))
    ids = {u["id"] for u in response.json()["items"]}
    assert registered_org["user_id"] not in ids


async def test_reading_another_orgs_user_is_a_404_not_a_403(api_client, registered_org):
    rival = await _register(api_client)
    response = await api_client.get(
        f"/api/v1/users/{registered_org['user_id']}", headers=_auth(rival)
    )
    assert response.status_code == 404


async def test_admin_can_add_a_recruiter_who_can_then_log_in(api_client, registered_org):
    email = f"recruiter-{uuid.uuid4().hex[:8]}@example.com"
    created = await api_client.post(
        "/api/v1/users",
        headers=_auth(registered_org),
        json={
            "email": email,
            "password": "another-correct-horse-staple",
            "full_name": "Rob Recruiter",
            "role": "RECRUITER",
        },
    )
    assert created.status_code == 201, created.text
    assert created.json()["role"] == "RECRUITER"
    assert "password" not in created.json()
    assert "hashed_password" not in created.json()

    login = await api_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "another-correct-horse-staple"},
    )
    assert login.status_code == 200, login.text


async def test_a_recruiter_cannot_add_or_edit_users(api_client, registered_org):
    email = f"recruiter-{uuid.uuid4().hex[:8]}@example.com"
    await api_client.post(
        "/api/v1/users",
        headers=_auth(registered_org),
        json={"email": email, "password": "another-correct-horse-staple", "role": "RECRUITER"},
    )
    login = await api_client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "another-correct-horse-staple"},
    )
    recruiter_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    # Reading the roster is fine -- a recruiter needs to see their team.
    assert (await api_client.get("/api/v1/users", headers=recruiter_headers)).status_code == 200
    # Changing it is not.
    assert (
        await api_client.post(
            "/api/v1/users",
            headers=recruiter_headers,
            json={"email": "x@example.com", "password": "yet-another-long-password"},
        )
    ).status_code == 403
    assert (
        await api_client.patch(
            f"/api/v1/users/{registered_org['user_id']}",
            headers=recruiter_headers,
            json={"is_active": False},
        )
    ).status_code == 403


async def test_the_last_admin_cannot_be_demoted_by_another_admin(api_client, registered_org):
    """Self-demotion is blocked elsewhere; this is the two-admin path."""
    email = f"admin2-{uuid.uuid4().hex[:8]}@example.com"
    second = await api_client.post(
        "/api/v1/users",
        headers=_auth(registered_org),
        json={"email": email, "password": "another-correct-horse-staple", "role": "ADMIN"},
    )
    assert second.status_code == 201
    login = await api_client.post(
        "/api/v1/auth/login", json={"email": email, "password": "another-correct-horse-staple"}
    )
    second_headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    # Two admins: demoting one is allowed.
    demoted = await api_client.patch(
        f"/api/v1/users/{registered_org['user_id']}",
        headers=second_headers,
        json={"role": "RECRUITER"},
    )
    assert demoted.status_code == 200, demoted.text

    # One admin left, and they are the caller, so the self-guard catches it first.
    assert (
        await api_client.patch(
            f"/api/v1/users/{second.json()['id']}",
            headers=second_headers,
            json={"role": "RECRUITER"},
        )
    ).status_code == 403


async def test_an_email_already_registered_elsewhere_is_refused_vaguely(
    api_client, registered_org
):
    """Global email uniqueness, without confirming which tenant holds it."""
    rival = await _register(api_client)
    response = await api_client.post(
        "/api/v1/users",
        headers=_auth(rival),
        json={
            "email": registered_org["admin_email"],
            "password": "another-correct-horse-staple",
        },
    )
    assert response.status_code == 409
    body = response.json()["error"]["message"].lower()
    assert "another organization" not in body and "already registered" not in body


async def test_admin_cannot_demote_itself(api_client, registered_org):
    """Otherwise a tenant can lock itself out of its own admin surface."""
    response = await api_client.patch(
        f"/api/v1/users/{registered_org['user_id']}",
        headers=_auth(registered_org),
        json={"role": "RECRUITER"},
    )
    assert response.status_code == 403


async def test_admin_cannot_deactivate_itself(api_client, registered_org):
    response = await api_client.delete(
        f"/api/v1/users/{registered_org['user_id']}", headers=_auth(registered_org)
    )
    assert response.status_code == 403


async def test_full_name_can_be_cleared_but_omitting_it_changes_nothing(
    api_client, registered_org
):
    """The distinction model_fields_set exists to preserve."""
    url = f"/api/v1/users/{registered_org['user_id']}"

    omitted = await api_client.patch(url, headers=_auth(registered_org), json={})
    assert omitted.json()["full_name"] == "Ada Admin"

    cleared = await api_client.patch(
        url, headers=_auth(registered_org), json={"full_name": None}
    )
    assert cleared.json()["full_name"] is None


# --- Candidates -------------------------------------------------------------


async def test_create_and_read_a_candidate(api_client, registered_org):
    created = await _create_candidate(api_client, registered_org, "cass@example.com")
    assert created["org_id"] == registered_org["org_id"]

    fetched = await api_client.get(
        f"/api/v1/candidates/{created['id']}", headers=_auth(registered_org)
    )
    assert fetched.status_code == 200
    assert fetched.json()["email"] == "cass@example.com"


async def test_duplicate_candidate_email_in_one_org_conflicts(api_client, registered_org):
    email = f"dup-{uuid.uuid4().hex[:8]}@example.com"
    await _create_candidate(api_client, registered_org, email)

    response = await api_client.post(
        "/api/v1/candidates", headers=_auth(registered_org), json={"email": email}
    )
    assert response.status_code == 409


async def test_the_same_candidate_email_may_exist_in_two_orgs(api_client, registered_org):
    """Uniqueness is per tenant: interviewing at two companies is two rows."""
    rival = await _register(api_client)
    email = f"shared-{uuid.uuid4().hex[:8]}@example.com"

    a = await _create_candidate(api_client, registered_org, email)
    b = await _create_candidate(api_client, rival, email)
    assert a["id"] != b["id"]
    assert a["org_id"] != b["org_id"]


async def test_another_org_cannot_read_update_or_delete_a_candidate(
    api_client, registered_org
):
    candidate = await _create_candidate(api_client, registered_org)
    rival = await _register(api_client)
    url = f"/api/v1/candidates/{candidate['id']}"

    assert (await api_client.get(url, headers=_auth(rival))).status_code == 404
    assert (
        await api_client.patch(url, headers=_auth(rival), json={"full_name": "Owned"})
    ).status_code == 404
    assert (await api_client.delete(url, headers=_auth(rival))).status_code == 404

    # And the row is untouched.
    survivor = await api_client.get(url, headers=_auth(registered_org))
    assert survivor.json()["full_name"] == "Cass"


async def test_candidate_search_matches_email_and_name(api_client, registered_org):
    await _create_candidate(api_client, registered_org, "findme@example.com")
    response = await api_client.get(
        "/api/v1/candidates", headers=_auth(registered_org), params={"search": "findme"}
    )
    assert response.json()["total"] == 1


# --- Jobs -------------------------------------------------------------------


async def test_create_job_and_add_description_versions(api_client, registered_org):
    job = await _create_job(api_client, registered_org)
    assert job["status"] == "DRAFT"

    first = await api_client.post(
        f"/api/v1/jobs/{job['id']}/descriptions",
        headers=_auth(registered_org),
        json={"content": "We are hiring a staff engineer to own the platform."},
    )
    assert first.status_code == 201
    assert first.json()["version"] == 1
    assert first.json()["is_active"] is True

    second = await api_client.post(
        f"/api/v1/jobs/{job['id']}/descriptions",
        headers=_auth(registered_org),
        json={"content": "Revised: we are hiring a staff engineer for the platform team."},
    )
    assert second.json()["version"] == 2

    listing = await api_client.get(
        f"/api/v1/jobs/{job['id']}/descriptions", headers=_auth(registered_org)
    )
    versions = [d["version"] for d in listing.json()]
    assert versions == [2, 1], "descriptions must come back newest first"

    # Exactly one active, and it is the newest.
    active = [d for d in listing.json() if d["is_active"]]
    assert len(active) == 1
    assert active[0]["version"] == 2


async def test_activating_an_older_description_moves_the_flag(api_client, registered_org):
    job = await _create_job(api_client, registered_org)
    v1 = (
        await api_client.post(
            f"/api/v1/jobs/{job['id']}/descriptions",
            headers=_auth(registered_org),
            json={"content": "The original description text for this role."},
        )
    ).json()
    await api_client.post(
        f"/api/v1/jobs/{job['id']}/descriptions",
        headers=_auth(registered_org),
        json={"content": "The replacement description text for this role."},
    )

    rolled_back = await api_client.post(
        f"/api/v1/jobs/{job['id']}/descriptions/{v1['id']}/activate",
        headers=_auth(registered_org),
    )
    assert rolled_back.status_code == 200
    assert rolled_back.json()["is_active"] is True

    listing = (
        await api_client.get(
            f"/api/v1/jobs/{job['id']}/descriptions", headers=_auth(registered_org)
        )
    ).json()
    assert [d["version"] for d in listing if d["is_active"]] == [1]


async def test_job_status_filter(api_client, registered_org):
    await _create_job(api_client, registered_org, "Draft Role")
    opened = await _create_job(api_client, registered_org, "Open Role")
    await api_client.patch(
        f"/api/v1/jobs/{opened['id']}", headers=_auth(registered_org), json={"status": "OPEN"}
    )

    response = await api_client.get(
        "/api/v1/jobs", headers=_auth(registered_org), params={"status": "OPEN"}
    )
    assert [j["title"] for j in response.json()["items"]] == ["Open Role"]


async def test_another_org_cannot_reach_a_job_or_its_descriptions(api_client, registered_org):
    job = await _create_job(api_client, registered_org)
    rival = await _register(api_client)

    assert (
        await api_client.get(f"/api/v1/jobs/{job['id']}", headers=_auth(rival))
    ).status_code == 404
    assert (
        await api_client.get(f"/api/v1/jobs/{job['id']}/descriptions", headers=_auth(rival))
    ).status_code == 404
    # Writing a description into another org's job must not silently succeed.
    assert (
        await api_client.post(
            f"/api/v1/jobs/{job['id']}/descriptions",
            headers=_auth(rival),
            json={"content": "Injected description from a rival tenant."},
        )
    ).status_code == 404


async def test_a_rivals_job_does_not_appear_in_a_listing(api_client, registered_org):
    await _create_job(api_client, registered_org, "Secret Role")
    rival = await _register(api_client)

    response = await api_client.get("/api/v1/jobs", headers=_auth(rival))
    assert response.json()["total"] == 0


# --- Candidate tokens -------------------------------------------------------


async def test_a_candidate_token_is_rejected_by_every_recruiter_router(
    api_client, registered_org
):
    """An interview token must not become a foothold in the recruiter API."""
    invite = await api_client.post(
        "/api/v1/auth/invites",
        headers=_auth(registered_org),
        json={"candidate_email": f"invitee-{uuid.uuid4().hex[:8]}@example.com"},
    )
    assert invite.status_code == 201, invite.text
    redeemed = await api_client.post(
        "/api/v1/auth/invites/redeem",
        json={"invite_token": invite.json()["invite_token"]},
    )
    assert redeemed.status_code == 200, redeemed.text
    candidate_headers = {"Authorization": f"Bearer {redeemed.json()['interview_token']}"}

    for path in ("/api/v1/users", "/api/v1/candidates", "/api/v1/jobs"):
        response = await api_client.get(path, headers=candidate_headers)
        assert response.status_code == 403, f"{path} accepted a candidate token"


async def test_unauthenticated_requests_are_rejected(api_client):
    for path in ("/api/v1/users", "/api/v1/candidates", "/api/v1/jobs"):
        assert (await api_client.get(path)).status_code == 401
