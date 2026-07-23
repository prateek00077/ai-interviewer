"""The auth rate limiter, and the environment-dependent defaults around it.

Nothing covered this before -- ``tests/conftest.py`` referenced a
``test_rate_limiting.py`` that does not exist, and worked around the limiter by
clearing its keys. So the limiter was live enough to make manual testing
impossible (five registrations an hour) and untested enough that nobody noticed.
"""

import pytest

from app.core.config import Settings
from app.core.exceptions import RateLimitedError

PROD = {"jwt_secret": "x" * 40, "nvidia_api_key": "key"}


# --- Environment-dependent defaults -----------------------------------------


def test_production_keeps_the_strict_limits():
    """The number that matters. A limit that quietly evaporates on the way to
    production is worse than no limit: you ship believing you have one."""
    settings = Settings(environment="production", **PROD)
    assert settings.register_max_attempts == 5
    assert settings.login_max_attempts == 10


def test_staging_keeps_the_strict_limits_too():
    settings = Settings(environment="staging")
    assert settings.register_max_attempts == 5


def test_development_is_relaxed_enough_to_test_through():
    """Five registrations an hour makes the product untestable on a laptop --
    every walkthrough run needs a fresh tenant."""
    settings = Settings(environment="development")
    assert settings.register_max_attempts >= 100
    assert settings.login_max_attempts >= 100


def test_development_is_relaxed_but_not_disabled():
    """Raised, not removed. A limiter that does not exist in development is one
    nobody notices is broken until production, and the 429 path should still be
    reachable by hammering."""
    settings = Settings(environment="development")
    assert settings.register_max_attempts < 10_000
    assert settings.register_attempt_window_seconds > 0


@pytest.mark.parametrize("field", ["register_max_attempts", "login_max_attempts"])
def test_an_explicit_value_always_wins(field):
    """Including someone setting it low on purpose to exercise the 429 path."""
    settings = Settings(environment="development", **{field: 2})
    assert getattr(settings, field) == 2


def test_an_explicit_value_wins_in_production_too():
    settings = Settings(environment="production", register_max_attempts=50, **PROD)
    assert settings.register_max_attempts == 50


# --- The limiter itself ------------------------------------------------------


class _FakeRedis:
    """Counts INCRs. Enough to drive the fixed-window logic."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def pipeline(self, transaction: bool = True):  # noqa: FBT001, FBT002, ARG002
        return _FakePipeline(self)

    async def ttl(self, key: str) -> int:
        return 42


class _FakePipeline:
    def __init__(self, redis: _FakeRedis) -> None:
        self._redis = redis
        self._key: str | None = None

    def incr(self, key: str) -> None:
        self._key = key
        self._redis.counts[key] = self._redis.counts.get(key, 0) + 1

    def expire(self, key: str, seconds: int) -> None:
        pass

    async def execute(self) -> list:
        return [self._redis.counts[self._key], True]


async def test_the_limiter_allows_up_to_the_limit_then_raises():
    from app.core.dependencies import rate_limit

    redis = _FakeRedis()
    for _ in range(3):
        await rate_limit(redis, bucket="b", identifier="1.2.3.4", limit=3, window_seconds=60)

    with pytest.raises(RateLimitedError):
        await rate_limit(redis, bucket="b", identifier="1.2.3.4", limit=3, window_seconds=60)


async def test_the_limit_is_per_identifier():
    """One noisy address must not lock out everyone else."""
    from app.core.dependencies import rate_limit

    redis = _FakeRedis()
    for _ in range(3):
        await rate_limit(redis, bucket="b", identifier="noisy", limit=3, window_seconds=60)

    # A different caller still gets its full allowance.
    await rate_limit(redis, bucket="b", identifier="quiet", limit=3, window_seconds=60)


async def test_buckets_do_not_share_a_counter():
    """Failing to log in must not consume the registration allowance."""
    from app.core.dependencies import rate_limit

    redis = _FakeRedis()
    for _ in range(3):
        await rate_limit(redis, bucket="login", identifier="ip", limit=3, window_seconds=60)

    await rate_limit(redis, bucket="register", identifier="ip", limit=3, window_seconds=60)


async def test_the_error_carries_a_retry_after():
    """A 429 with no retry_after tells a client to guess."""
    from app.core.dependencies import rate_limit

    redis = _FakeRedis()
    await rate_limit(redis, bucket="b", identifier="ip", limit=1, window_seconds=60)

    with pytest.raises(RateLimitedError) as caught:
        await rate_limit(redis, bucket="b", identifier="ip", limit=1, window_seconds=60)
    # An attribute rather than context: the exception handler reads it to set
    # the Retry-After header, where a client will actually look for it.
    assert caught.value.retry_after >= 1
