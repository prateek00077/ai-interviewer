"""Proves a replayed refresh token kills its family, even under a race.

Rotation is only a security control if replay is *detected*, and detection is
only sound if check-and-consume is atomic. The concurrency test is therefore the
important one here: run against a non-atomic implementation, two simultaneous
uses of the same token both return OK and the theft goes unnoticed.

Requires a live Redis (``docker compose up redis``).
"""

import asyncio
import uuid

import pytest
import pytest_asyncio
from redis.asyncio import Redis

from app.core.config import settings
from app.modules.auth.tokens import (
    ConsumeOutcome,
    RefreshTokenStore,
    fam_dead_key,
    rt_key,
)

pytestmark = [pytest.mark.redis, pytest.mark.integration]


@pytest_asyncio.fixture(loop_scope="session")
async def store():
    redis = Redis.from_url(settings.redis_url)
    # Every test gets a namespace of its own via fresh uuids, so a flush is not
    # needed and a developer's local Redis survives the suite.
    try:
        await redis.ping()
    except Exception:  # pragma: no cover - environment guard
        pytest.skip("no Redis at REDIS_URL")
    try:
        yield RefreshTokenStore(redis)
    finally:
        await redis.aclose()


@pytest_asyncio.fixture(loop_scope="session")
async def issued(store):
    """One registered, active refresh token."""
    user_id, org_id = uuid.uuid4(), uuid.uuid4()
    family_id, jti = store.new_family(), str(uuid.uuid4())
    await store.register(jti=jti, user_id=user_id, org_id=org_id, family_id=family_id)
    return {"user_id": user_id, "org_id": org_id, "family_id": family_id, "jti": jti}


async def _rotate(store, issued, jti: str) -> str:
    """Consume ``jti`` and register its successor in the same family."""
    result = await store.consume(
        jti=jti, user_id=issued["user_id"], family_id=issued["family_id"]
    )
    assert result.ok
    successor = str(uuid.uuid4())
    await store.register(
        jti=successor,
        user_id=issued["user_id"],
        org_id=issued["org_id"],
        family_id=issued["family_id"],
    )
    return successor


# --- The happy path ---------------------------------------------------------


async def test_first_use_succeeds(store, issued):
    result = await store.consume(
        jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
    )
    assert result.outcome is ConsumeOutcome.OK


async def test_a_chain_of_rotations_all_succeed(store, issued):
    jti = issued["jti"]
    for _ in range(5):
        jti = await _rotate(store, issued, jti)
    assert not await store.is_family_dead(issued["family_id"])


# --- Reuse detection --------------------------------------------------------


async def test_replaying_a_consumed_token_is_reuse(store, issued):
    await _rotate(store, issued, issued["jti"])

    replay = await store.consume(
        jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
    )
    assert replay.outcome is ConsumeOutcome.REUSE


async def test_reuse_kills_the_whole_family_including_the_valid_successor(store, issued):
    """The victim is logged out too. We cannot tell which party was the thief."""
    successor = await _rotate(store, issued, issued["jti"])

    await store.consume(
        jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
    )

    assert await store.is_family_dead(issued["family_id"])
    later = await store.consume(
        jti=successor, user_id=issued["user_id"], family_id=issued["family_id"]
    )
    assert later.outcome is ConsumeOutcome.DEAD


async def test_reuse_deletes_member_records(store, issued):
    successor = await _rotate(store, issued, issued["jti"])
    await store.consume(
        jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
    )
    assert await store._redis.exists(rt_key(successor)) == 0


async def test_concurrent_use_of_one_token_yields_exactly_one_ok(store, issued):
    """The test the Lua script exists for.

    Without atomicity both coroutines read ``status == active`` before either
    writes, both are handed a fresh token, and the stolen credential is never
    noticed.
    """
    first, second = await asyncio.gather(
        store.consume(
            jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
        ),
        store.consume(
            jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
        ),
    )
    outcomes = sorted(r.outcome.value for r in (first, second))
    assert outcomes == ["OK", "REUSE"], f"expected one winner and one replay, got {outcomes}"


# --- Fail-closed paths ------------------------------------------------------


async def test_unknown_jti_is_rejected_and_tombstones_the_family(store):
    """A signed token with no record fails closed rather than open."""
    user_id, family_id = uuid.uuid4(), str(uuid.uuid4())
    result = await store.consume(jti=str(uuid.uuid4()), user_id=user_id, family_id=family_id)
    assert result.outcome is ConsumeOutcome.UNKNOWN
    assert await store.is_family_dead(family_id)


async def test_claims_disagreeing_with_the_record_are_hostile(store, issued):
    """Right jti, wrong user: someone is splicing claims across records."""
    result = await store.consume(
        jti=issued["jti"], user_id=uuid.uuid4(), family_id=issued["family_id"]
    )
    assert result.outcome is ConsumeOutcome.MISMATCH
    assert await store.is_family_dead(issued["family_id"])


async def test_family_mismatch_is_hostile(store, issued):
    result = await store.consume(
        jti=issued["jti"], user_id=issued["user_id"], family_id=str(uuid.uuid4())
    )
    assert result.outcome is ConsumeOutcome.MISMATCH


# --- Revocation -------------------------------------------------------------


async def test_logout_revokes_the_family(store, issued):
    await store.revoke_family(issued["family_id"])
    result = await store.consume(
        jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
    )
    assert result.outcome is ConsumeOutcome.DEAD


async def test_logout_is_idempotent(store, issued):
    await store.revoke_family(issued["family_id"])
    await store.revoke_family(issued["family_id"])  # must not raise
    assert await store.is_family_dead(issued["family_id"])


async def test_logout_all_kills_every_family_for_the_user(store):
    user_id, org_id = uuid.uuid4(), uuid.uuid4()
    families = []
    for _ in range(3):
        family_id, jti = store.new_family(), str(uuid.uuid4())
        await store.register(jti=jti, user_id=user_id, org_id=org_id, family_id=family_id)
        families.append((family_id, jti))

    assert await store.revoke_all_for_user(user_id) == 3

    for family_id, jti in families:
        result = await store.consume(jti=jti, user_id=user_id, family_id=family_id)
        assert result.outcome is ConsumeOutcome.DEAD


async def test_logout_all_leaves_other_users_alone(store, issued):
    await store.revoke_all_for_user(uuid.uuid4())
    result = await store.consume(
        jti=issued["jti"], user_id=issued["user_id"], family_id=issued["family_id"]
    )
    assert result.outcome is ConsumeOutcome.OK


# --- Expiry -----------------------------------------------------------------


async def test_records_outlive_the_jwt_they_describe(store, issued):
    """If the record expired first, a late replay would read as UNKNOWN, not REUSE."""
    ttl = await store._redis.ttl(rt_key(issued["jti"]))
    assert ttl > settings.refresh_token_ttl_days * 86_400


async def test_tombstones_are_not_immortal(store, issued):
    await store.revoke_family(issued["family_id"])
    assert await store._redis.ttl(fam_dead_key(issued["family_id"])) > 0
