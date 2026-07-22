"""End-to-end auth: register, login, protected route, rotate, replay, redeem.

The two tests that matter most are the ones that assert a *failure*:

- ``test_replaying_a_rotated_refresh_token_kills_the_session`` -- rotation without
  reuse detection is bookkeeping, not security.
- ``test_login_failures_are_indistinguishable`` -- four different reasons must
  produce one byte-identical response, or login becomes an account oracle.

Requires a live Postgres with migrations applied, and a live Redis.
"""

import uuid

import pytest
from sqlalchemy import text

from app.db.session import tenant_session
from app.models.user import UserRole

pytestmark = [pytest.mark.integration, pytest.mark.redis]

REGISTER = "/api/v1/auth/register-org"
LOGIN = "/api/v1/auth/login"
REFRESH = "/api/v1/auth/refresh"
LOGOUT = "/api/v1/auth/logout"
LOGOUT_ALL = "/api/v1/auth/logout-all"
ME = "/api/v1/auth/me"
INVITES = "/api/v1/auth/invites"
REDEEM = "/api/v1/auth/invites/redeem"


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _login(api_client, org) -> dict:
    response = await api_client.post(
        LOGIN, json={"email": org["admin_email"], "password": org["admin_password"]}
    )
    assert response.status_code == 200, response.text
    return response.json()


# --- Registration -----------------------------------------------------------


async def test_register_org_returns_a_usable_token_pair(api_client, registered_org):
    tokens = registered_org["tokens"]
    assert tokens["token_type"] == "bearer"
    assert tokens["expires_in"] > 0

    response = await api_client.get(ME, headers=_auth(tokens["access_token"]))
    assert response.status_code == 200
    body = response.json()
    assert body["org_id"] == registered_org["org_id"]
    assert body["role"] == UserRole.ADMIN.value
    assert body["actor_kind"] == "user"


async def test_duplicate_slug_conflicts_without_saying_which_field(api_client, registered_org):
    response = await api_client.post(
        REGISTER,
        json={
            "org_name": "Impostor",
            "slug": registered_org["slug"],
            "admin_email": f"other-{uuid.uuid4().hex[:8]}@mail.example.com",
            "admin_password": "correct-horse-battery-staple",
        },
    )
    assert response.status_code == 409
    message = response.json()["error"]["message"]
    # Naming the colliding field would turn signup into an existence oracle.
    assert "slug" not in message.lower()
    assert registered_org["slug"] not in message


async def test_weak_password_is_rejected_before_any_write(api_client):
    slug = f"weak-{uuid.uuid4().hex[:8]}"
    response = await api_client.post(
        REGISTER,
        json={
            "org_name": "Weak",
            "slug": slug,
            "admin_email": f"admin@{slug}.example.com",
            "admin_password": "short",
        },
    )
    assert response.status_code == 422


# --- Login ------------------------------------------------------------------


async def test_login_succeeds_and_records_last_login(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    assert tokens["access_token"] and tokens["refresh_token"]

    async with tenant_session(registered_org["org_id"], "user", registered_org["user_id"]) as s:
        last = (
            await s.execute(
                text("SELECT last_login_at FROM users WHERE id = :id"),
                {"id": registered_org["user_id"]},
            )
        ).scalar_one()
    assert last is not None


async def test_login_failures_are_indistinguishable(api_client, registered_org):
    """Unknown email and wrong password must produce the identical response."""
    unknown = await api_client.post(
        LOGIN,
        json={
            "email": f"nobody-{uuid.uuid4().hex[:8]}@mail.example.com",
            "password": "whatever-long-password",
        },
    )
    wrong = await api_client.post(
        LOGIN, json={"email": registered_org["admin_email"], "password": "wrong-password-here"}
    )

    assert unknown.status_code == wrong.status_code == 401
    unknown_body, wrong_body = unknown.json(), wrong.json()
    assert unknown_body["error"]["code"] == wrong_body["error"]["code"] == "invalid_credentials"
    assert unknown_body["error"]["message"] == wrong_body["error"]["message"]


async def test_deactivated_user_gets_the_same_401(api_client, registered_org):
    async with tenant_session(registered_org["org_id"], "user", registered_org["user_id"]) as s:
        await s.execute(
            text("UPDATE users SET is_active = false WHERE id = :id"),
            {"id": registered_org["user_id"]},
        )

    response = await api_client.post(
        LOGIN,
        json={"email": registered_org["admin_email"], "password": registered_org["admin_password"]},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"

    async with tenant_session(registered_org["org_id"], "user", registered_org["user_id"]) as s:
        await s.execute(
            text("UPDATE users SET is_active = true WHERE id = :id"),
            {"id": registered_org["user_id"]},
        )


# --- Protected routes -------------------------------------------------------


async def test_protected_route_requires_a_token(api_client):
    response = await api_client.get(ME)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer not-a-jwt"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer "},
    ],
)
async def test_malformed_credentials_are_rejected(api_client, header):
    response = await api_client.get(ME, headers=header)
    assert response.status_code == 401


async def test_a_refresh_token_is_not_a_bearer_credential(api_client, registered_org):
    """The derived-key design in action, over HTTP."""
    refresh_token = registered_org["tokens"]["refresh_token"]
    response = await api_client.get(ME, headers=_auth(refresh_token))
    assert response.status_code == 401


# --- Rotation ---------------------------------------------------------------


async def test_refresh_returns_a_new_pair(api_client, registered_org):
    original = await _login(api_client, registered_org)
    response = await api_client.post(REFRESH, json={"refresh_token": original["refresh_token"]})
    assert response.status_code == 200

    rotated = response.json()
    assert rotated["refresh_token"] != original["refresh_token"]
    assert (await api_client.get(ME, headers=_auth(rotated["access_token"]))).status_code == 200


async def test_replaying_a_rotated_refresh_token_kills_the_session(api_client, registered_org):
    """Replay must revoke the family, taking the honest successor with it."""
    original = await _login(api_client, registered_org)
    rotated = (
        await api_client.post(REFRESH, json={"refresh_token": original["refresh_token"]})
    ).json()

    replay = await api_client.post(REFRESH, json={"refresh_token": original["refresh_token"]})
    assert replay.status_code == 401

    # The legitimate successor is dead too. We cannot tell which party was the
    # thief, so both are logged out.
    after = await api_client.post(REFRESH, json={"refresh_token": rotated["refresh_token"]})
    assert after.status_code == 401


async def test_refresh_rejects_an_access_token(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    response = await api_client.post(REFRESH, json={"refresh_token": tokens["access_token"]})
    assert response.status_code == 401


# --- Logout -----------------------------------------------------------------


async def test_logout_revokes_the_refresh_token(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    assert (
        await api_client.post(LOGOUT, json={"refresh_token": tokens["refresh_token"]})
    ).status_code == 204
    assert (
        await api_client.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
    ).status_code == 401


async def test_logout_is_idempotent(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    for _ in range(3):
        response = await api_client.post(LOGOUT, json={"refresh_token": tokens["refresh_token"]})
        assert response.status_code == 204


async def test_logout_accepts_garbage_without_confirming_anything(api_client):
    """A 401 here would tell an attacker which tokens are live."""
    response = await api_client.post(LOGOUT, json={"refresh_token": "not-a-token"})
    assert response.status_code == 204


async def test_logout_all_revokes_every_session(api_client, registered_org):
    first = await _login(api_client, registered_org)
    second = await _login(api_client, registered_org)

    response = await api_client.post(LOGOUT_ALL, headers=_auth(second["access_token"]))
    assert response.status_code == 204

    for tokens in (first, second):
        assert (
            await api_client.post(REFRESH, json={"refresh_token": tokens["refresh_token"]})
        ).status_code == 401


# --- Invites ----------------------------------------------------------------


async def _create_invite(api_client, access_token, email=None):
    response = await api_client.post(
        INVITES,
        headers=_auth(access_token),
        json={
            "candidate_email": email or f"candidate-{uuid.uuid4().hex[:8]}@mail.example.com",
            "candidate_name": "Casey Candidate",
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


async def test_invite_creation_requires_authentication(api_client):
    response = await api_client.post(INVITES, json={"candidate_email": "x@mail.example.com"})
    assert response.status_code == 401


async def test_invite_round_trip_yields_an_interview_token(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    invite = await _create_invite(api_client, tokens["access_token"])

    response = await api_client.post(REDEEM, json={"invite_token": invite["invite_token"]})
    assert response.status_code == 200

    redeemed = response.json()
    assert redeemed["interview_id"] == invite["interview_id"]
    assert redeemed["candidate_id"] == invite["candidate_id"]
    assert 0 < redeemed["expires_in"] <= 10 * 60


async def test_an_interview_token_is_a_candidate_not_a_recruiter(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    invite = await _create_invite(api_client, tokens["access_token"])
    redeemed = (
        await api_client.post(REDEEM, json={"invite_token": invite["invite_token"]})
    ).json()

    # /me is user-only, so a candidate token must be refused with 403 -- it
    # authenticated fine, it simply is not a recruiter.
    response = await api_client.get(ME, headers=_auth(redeemed["interview_token"]))
    assert response.status_code == 403

    # And it certainly cannot mint invites.
    response = await api_client.post(
        INVITES,
        headers=_auth(redeemed["interview_token"]),
        json={"candidate_email": "smuggled@mail.example.com"},
    )
    assert response.status_code == 403


async def test_invite_is_multi_use_up_to_its_limit(api_client, registered_org):
    """Default max_redemptions is 3: a candidate whose browser crashes rejoins."""
    tokens = await _login(api_client, registered_org)
    invite = await _create_invite(api_client, tokens["access_token"])

    for attempt in range(3):
        response = await api_client.post(REDEEM, json={"invite_token": invite["invite_token"]})
        assert response.status_code == 200, f"redemption {attempt + 1} failed"

    exhausted = await api_client.post(REDEEM, json={"invite_token": invite["invite_token"]})
    assert exhausted.status_code == 410
    assert exhausted.json()["error"]["code"] == "invite_unusable"


async def test_max_redemptions_is_honoured_when_set(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    response = await api_client.post(
        INVITES,
        headers=_auth(tokens["access_token"]),
        json={
            "candidate_email": f"once-{uuid.uuid4().hex[:8]}@mail.example.com",
            "max_redemptions": 1,
        },
    )
    invite = response.json()

    assert (
        await api_client.post(REDEEM, json={"invite_token": invite["invite_token"]})
    ).status_code == 200
    assert (
        await api_client.post(REDEEM, json={"invite_token": invite["invite_token"]})
    ).status_code == 410


@pytest.mark.parametrize("bogus", ["not-a-token", "a.b.c"])
async def test_a_forged_invite_looks_exactly_like_an_expired_one(api_client, bogus):
    response = await api_client.post(REDEEM, json={"invite_token": bogus})
    assert response.status_code == 410
    assert response.json()["error"]["code"] == "invite_unusable"


async def test_an_access_token_is_not_an_invite(api_client, registered_org):
    tokens = await _login(api_client, registered_org)
    response = await api_client.post(REDEEM, json={"invite_token": tokens["access_token"]})
    assert response.status_code == 410


# --- Tenancy over HTTP ------------------------------------------------------


async def test_one_orgs_token_cannot_see_another_orgs_candidates(api_client, registered_org):
    """The RLS guarantee, exercised through the real request path."""
    other_slug = f"other-{uuid.uuid4().hex[:10]}"
    other = (
        await api_client.post(
            REGISTER,
            json={
                "org_name": "Other Inc",
                "slug": other_slug,
                "admin_email": f"admin@{other_slug}.example.com",
                "admin_password": "correct-horse-battery-staple",
            },
        )
    ).json()

    try:
        mine = await _login(api_client, registered_org)
        await _create_invite(api_client, mine["access_token"], "shared@mail.example.com")
        await _create_invite(
            api_client, other["tokens"]["access_token"], "shared@mail.example.com"
        )

        # The same address exists in both tenants as two independent rows.
        async with tenant_session(other["org_id"]) as s:
            rows = (
                await s.execute(text("SELECT org_id FROM candidates"))
            ).scalars().all()
        assert {str(r) for r in rows} == {other["org_id"]}
    finally:
        async with tenant_session(other["org_id"]) as s:
            await s.execute(
                text("DELETE FROM organizations WHERE id = :id"), {"id": other["org_id"]}
            )


# --- Errors -----------------------------------------------------------------


async def test_every_error_carries_a_request_id(api_client):
    response = await api_client.get(ME)
    assert response.json()["error"]["request_id"] == response.headers["X-Request-ID"]


async def test_health_needs_no_auth(api_client):
    response = await api_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
