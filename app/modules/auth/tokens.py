"""Refresh-token family store: rotation with reuse detection, backed by Redis.

Rotation alone is not a security property. It becomes one only if *replaying* a
consumed token is detectable, and detection is only sound if check-and-consume is
atomic: two concurrent replays that both read ``active`` would both be issued a
new token, which is precisely the theft the scheme exists to catch. So the whole
decision happens inside one Lua script, which Redis runs to completion without
interleaving.

Key schema
----------
``rt:{jti}``          hash  -- uid, org, fam, status(active|used)
``rtfam:{fam}``       set   -- every jti ever minted in this family
``rtfam:dead:{fam}``  key   -- tombstone; presence means the family is revoked
``rtuser:{uid}``      set   -- every family belonging to a user (for logout-all)

Every key's TTL is the refresh TTL plus a margin. The *consumed* record has to
outlive the JWT's own ``exp``, otherwise a replay arriving near expiry finds no
record, and "unknown" and "reused" become indistinguishable.

On reuse the family is tombstoned and every member jti deleted, which logs out
the attacker *and* the legitimate user. That is the intended outcome: one of the
two holds a stolen token and we cannot tell which.
"""

import uuid
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import structlog
from redis.asyncio import Redis

from app.core.config import settings

log = structlog.get_logger(__name__)

# The consumed record must outlive the JWT it describes, or reuse is invisible.
_TTL_MARGIN_SECONDS = 60


def _record_ttl() -> int:
    return settings.refresh_token_ttl_days * 86_400 + _TTL_MARGIN_SECONDS


def rt_key(jti: str) -> str:
    return f"rt:{jti}"


def fam_key(family_id: str) -> str:
    return f"rtfam:{family_id}"


def fam_dead_key(family_id: str) -> str:
    return f"rtfam:dead:{family_id}"


def user_key(user_id: uuid.UUID | str) -> str:
    return f"rtuser:{user_id}"


class ConsumeOutcome(StrEnum):
    OK = "OK"
    REUSE = "REUSE"
    UNKNOWN = "UNKNOWN"
    DEAD = "DEAD"
    MISMATCH = "MISMATCH"


@dataclass(frozen=True, slots=True)
class ConsumeResult:
    outcome: ConsumeOutcome

    @property
    def ok(self) -> bool:
        return self.outcome is ConsumeOutcome.OK


# KEYS[1] rt:{jti}   KEYS[2] rtfam:{fam}   KEYS[3] rtfam:dead:{fam}
# ARGV[1] uid        ARGV[2] fam           ARGV[3] tombstone ttl
#
# Ordering matters: the tombstone is checked before the record, so a family
# killed by an earlier replay stays dead even if some member record survives.
_CONSUME_LUA = """
if redis.call('EXISTS', KEYS[3]) == 1 then
  return 'DEAD'
end

if redis.call('EXISTS', KEYS[1]) == 0 then
  -- Signature was valid but we have no record: either the family was already
  -- purged, or this is a replay of something we forgot. Fail closed.
  redis.call('SET', KEYS[3], '1', 'EX', tonumber(ARGV[3]))
  return 'UNKNOWN'
end

local uid = redis.call('HGET', KEYS[1], 'uid')
local fam = redis.call('HGET', KEYS[1], 'fam')
if uid ~= ARGV[1] or fam ~= ARGV[2] then
  -- The JWT's claims disagree with the stored record. Treat as hostile.
  redis.call('SET', KEYS[3], '1', 'EX', tonumber(ARGV[3]))
  return 'MISMATCH'
end

if redis.call('HGET', KEYS[1], 'status') ~= 'active' then
  -- Replay of an already-rotated token. Kill the whole family.
  redis.call('SET', KEYS[3], '1', 'EX', tonumber(ARGV[3]))
  local members = redis.call('SMEMBERS', KEYS[2])
  for i = 1, #members do
    redis.call('DEL', 'rt:' .. members[i])
  end
  redis.call('DEL', KEYS[2])
  return 'REUSE'
end

redis.call('HSET', KEYS[1], 'status', 'used')
return 'OK'
"""


class RefreshTokenStore:
    """All Redis state for refresh rotation. One instance per app, reused."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        # register_script sends EVALSHA and falls back to EVAL on NOSCRIPT, so a
        # Redis restart mid-flight does not break rotation.
        self._consume = redis.register_script(_CONSUME_LUA)

    # --- issue ---------------------------------------------------------------

    async def register(
        self, *, jti: str, user_id: uuid.UUID, org_id: uuid.UUID, family_id: str
    ) -> None:
        """Record a freshly minted refresh token as the family's active member."""
        ttl = _record_ttl()
        pipe = self._redis.pipeline(transaction=True)
        pipe.hset(
            rt_key(jti),
            mapping={
                "uid": str(user_id),
                "org": str(org_id),
                "fam": family_id,
                "status": "active",
            },
        )
        pipe.expire(rt_key(jti), ttl)
        pipe.sadd(fam_key(family_id), jti)
        pipe.expire(fam_key(family_id), ttl)
        pipe.sadd(user_key(user_id), family_id)
        pipe.expire(user_key(user_id), ttl)
        await pipe.execute()

    def new_family(self) -> str:
        return str(uuid.uuid4())

    # --- rotate --------------------------------------------------------------

    async def consume(
        self, *, jti: str, user_id: uuid.UUID, family_id: str
    ) -> ConsumeResult:
        """Atomically mark a token used, or report why it cannot be used.

        The caller must treat anything other than OK as a 401 and must not
        distinguish the reasons to the client.
        """
        raw: Any = await self._consume(
            keys=[rt_key(jti), fam_key(family_id), fam_dead_key(family_id)],
            args=[str(user_id), family_id, _record_ttl()],
        )
        outcome = ConsumeOutcome(raw.decode() if isinstance(raw, bytes) else raw)

        if outcome is not ConsumeOutcome.OK:
            # A security event, not an application error: log it loudly, since a
            # REUSE is the only signal that a refresh token has been stolen.
            log.warning(
                "security.refresh_rejected",
                outcome=outcome.value,
                family_id=family_id,
                user_id=str(user_id),
            )
        return ConsumeResult(outcome)

    # --- revoke --------------------------------------------------------------

    async def _members(self, key: str) -> list[str]:
        """Set members as str.

        redis-py shares one class between the sync and async clients, so its stubs
        type every command as ``Awaitable[T] | T``. The cast is where that union
        stops, rather than at each call site.
        """
        raw = cast(Awaitable[set[Any]], self._redis.smembers(key))
        return [m.decode() if isinstance(m, bytes) else m for m in await raw]

    async def revoke_family(self, family_id: str) -> None:
        """Logout. Idempotent -- tombstoning an already-dead family is a no-op."""
        members = await self._members(fam_key(family_id))
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(fam_dead_key(family_id), "1", ex=_record_ttl())
        for member in members:
            pipe.delete(rt_key(member))
        pipe.delete(fam_key(family_id))
        await pipe.execute()

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> int:
        """Logout everywhere. Returns the number of families killed."""
        decoded = await self._members(user_key(user_id))
        for family_id in decoded:
            await self.revoke_family(family_id)
        await self._redis.delete(user_key(user_id))
        log.info("auth.logout_all", user_id=str(user_id), families=len(decoded))
        return len(decoded)

    async def is_family_dead(self, family_id: str) -> bool:
        return bool(await self._redis.exists(fam_dead_key(family_id)))
