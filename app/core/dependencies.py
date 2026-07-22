"""Shared FastAPI dependencies: get_current_user, get_current_org, require_role.

The chain is deliberately one-way:

    Bearer token -> Principal -> org-scoped DB session

``org_id`` is read only from the token's signed claims. It is never accepted from
a header, path, query string, or body -- doing so would make tenant isolation a
matter of the client's honesty, when the whole point of the RLS design is that it
is not.

ACCEPTED TRADEOFF: access tokens are stateless, so one stays valid for up to its
full TTL after logout. If that window becomes unacceptable, add a Redis
``denylist:{jti}`` check inside ``_principal_from_access_token`` -- that is the
single place it needs to go.
"""

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.exceptions import AuthenticationError, PermissionDeniedError, RateLimitedError
from app.core.logging import bind_principal
from app.core.security import (
    AccessClaims,
    ActorKind,
    InterviewClaims,
    TokenType,
    decode_token,
)
from app.db.session import tenant_session, unscoped_session
from app.models.user import UserRole

# auto_error=False so a missing header raises our own uniform 401 body rather
# than Starlette's, which has a different shape.
_bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True, slots=True)
class Principal:
    """Who is calling, as established by a signed token and nothing else."""

    org_id: uuid.UUID
    actor_kind: ActorKind
    actor_id: uuid.UUID
    role: str
    jti: str
    # Populated for candidates only.
    interview_id: uuid.UUID | None = None

    @property
    def is_user(self) -> bool:
        return self.actor_kind is ActorKind.USER

    @property
    def is_candidate(self) -> bool:
        return self.actor_kind is ActorKind.CANDIDATE


# --- Infrastructure ---------------------------------------------------------


def get_redis(request: Request) -> Redis:
    """The connection pool created at startup. One per process."""
    return request.app.state.redis


def get_token_store(request: Request):
    from app.modules.auth.tokens import RefreshTokenStore  # local: avoids a cycle

    store: RefreshTokenStore = request.app.state.token_store
    return store


async def get_unscoped_db() -> AsyncIterator[AsyncSession]:
    """A session with no org context. Under RLS it reads no tenant rows.

    Only the login path needs this, and it reaches the database solely through
    ``app.lookup_user_for_auth``.
    """
    async with unscoped_session() as session:
        yield session


# --- Authentication ---------------------------------------------------------


def _bearer_token(credentials: HTTPAuthorizationCredentials | None) -> str:
    if credentials is None or not credentials.credentials:
        raise AuthenticationError()
    if credentials.scheme.lower() != "bearer":
        raise AuthenticationError()
    return credentials.credentials


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    """Accept either an access token or an interview token.

    Both are decoded with their own derived key, so a candidate's interview token
    cannot be presented as a recruiter's access token: it fails at signature
    verification, not at a claim comparison.
    """
    raw = _bearer_token(credentials)

    try:
        claims = AccessClaims.parse(decode_token(raw, TokenType.ACCESS))
        principal = Principal(
            org_id=claims.org_id,
            actor_kind=ActorKind.USER,
            actor_id=claims.user_id,
            role=claims.role,
            jti=claims.jti,
        )
    except AuthenticationError:
        # Not an access token. The only other type a request may carry is an
        # interview token; invite and refresh tokens are never bearer credentials.
        interview = InterviewClaims.parse(decode_token(raw, TokenType.INTERVIEW))
        principal = Principal(
            org_id=interview.org_id,
            actor_kind=ActorKind.CANDIDATE,
            actor_id=interview.candidate_id,
            role="CANDIDATE",
            jti=interview.jti,
            interview_id=interview.interview_id,
        )

    bind_principal(
        org_id=str(principal.org_id),
        actor_kind=principal.actor_kind.value,
        actor_id=str(principal.actor_id),
    )
    return principal


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


async def get_current_user(principal: CurrentPrincipal) -> Principal:
    """Reject candidate tokens on recruiter-facing routes."""
    if not principal.is_user:
        raise PermissionDeniedError()
    return principal


async def get_current_candidate(principal: CurrentPrincipal) -> Principal:
    if not principal.is_candidate:
        raise PermissionDeniedError()
    return principal


async def get_current_org(principal: CurrentPrincipal) -> uuid.UUID:
    return principal.org_id


# --- Scoped database session ------------------------------------------------


async def get_db(principal: CurrentPrincipal) -> AsyncIterator[AsyncSession]:
    """The org-scoped session every authenticated route should depend on.

    Because the GUCs come from ``principal``, a route that forgets to filter by
    org is still safe -- the policy filters for it.
    """
    async with tenant_session(
        principal.org_id, principal.actor_kind.value, principal.actor_id
    ) as session:
        yield session


ScopedSession = Annotated[AsyncSession, Depends(get_db)]


# --- Authorization ----------------------------------------------------------


def require_role(*roles: UserRole) -> Callable[[Principal], Principal]:
    """Restrict a route to specific user roles.

    ACCEPTED TRADEOFF: the role comes from the access token, so a demotion takes
    effect at the next refresh rather than instantly. The refresh path re-reads
    the role from the database precisely to bound that window.
    """
    allowed = {r.value for r in roles}

    def _check(principal: Annotated[Principal, Depends(get_current_user)]) -> Principal:
        if principal.role not in allowed:
            raise PermissionDeniedError(required=sorted(allowed), actual=principal.role)
        return principal

    return _check


# --- Rate limiting ----------------------------------------------------------


def client_ip(request: Request) -> str:
    """Best-effort client address.

    X-Forwarded-For is only trustworthy behind a proxy that overwrites it. Until
    one is in front of this service, the socket address is the honest answer.
    """
    return request.client.host if request.client else "unknown"


async def rate_limit(
    redis: Redis, *, bucket: str, identifier: str, limit: int, window_seconds: int
) -> None:
    """Fixed-window counter. Raises 429 once the limit is exceeded.

    INCR-then-EXPIRE is safe here because the key only ever grows within a
    window: a lost EXPIRE would leave a stuck counter, so the TTL is set on
    every increment rather than only the first.
    """
    key = f"rl:{bucket}:{identifier}"
    pipe = redis.pipeline(transaction=True)
    pipe.incr(key)
    pipe.expire(key, window_seconds)
    count, _ = await pipe.execute()

    if count > limit:
        ttl = await redis.ttl(key)
        raise RateLimitedError(
            retry_after=max(ttl, 1), bucket=bucket, identifier=identifier, count=count
        )


async def login_rate_limit(request: Request) -> None:
    await rate_limit(
        get_redis(request),
        bucket="login",
        identifier=client_ip(request),
        limit=settings.login_max_attempts,
        window_seconds=settings.login_attempt_window_seconds,
    )


async def register_rate_limit(request: Request) -> None:
    await rate_limit(
        get_redis(request),
        bucket="register",
        identifier=client_ip(request),
        limit=settings.register_max_attempts,
        window_seconds=settings.register_attempt_window_seconds,
    )
