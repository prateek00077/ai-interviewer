"""Login and token issuance.

Every function here takes its database session from the caller rather than
opening one, so an endpoint composes several operations in one transaction and a
failure anywhere rolls the whole thing back.

Two rules this module exists to enforce:

1. **Login is uniform.** Unknown email, wrong password, deactivated user and
   deactivated org all raise the same ``InvalidCredentialsError``, and the
   unknown-email branch still burns a KDF so response time does not disclose
   whether an address is registered.
2. **Nothing outside login runs without an org.** The org-less lookup goes
   through one narrow SECURITY DEFINER function; every other statement here runs
   on an org-scoped session and is filtered by RLS.
"""

import uuid
from dataclasses import dataclass

import structlog
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, InvalidCredentialsError, InvalidTokenError
from app.core.security import (
    RefreshClaims,
    TokenType,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_and_upgrade,
    verify_password_dummy,
)
from app.db.session import tenant_session
from app.models.org import Organization
from app.models.user import User, UserRole
from app.modules.auth.tokens import RefreshTokenStore

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TokenPair:
    access_token: str
    refresh_token: str
    expires_in: int
    token_type: str = "bearer"


@dataclass(frozen=True, slots=True)
class AuthenticatedUser:
    user_id: uuid.UUID
    org_id: uuid.UUID
    role: str


# The one deliberate RLS bypass. See app/db/rls.py for why it is shaped this way.
_LOOKUP_SQL = text(
    "SELECT id, org_id, hashed_password, role, is_active, org_active "
    "FROM app.lookup_user_for_auth(:email)"
)

_UPGRADE_HASH_SQL = text("UPDATE users SET hashed_password = :hash WHERE id = :id")

_TOUCH_LOGIN_SQL = text("UPDATE users SET last_login_at = now() WHERE id = :id")


# --- Registration -----------------------------------------------------------


async def register_org(
    *,
    org_name: str,
    slug: str,
    admin_email: str,
    admin_password: str,
    admin_full_name: str | None = None,
) -> AuthenticatedUser:
    """Create a tenant and its first admin.

    The org id is generated in Python *before* the INSERT so the session can be
    opened with it already in context. The organizations policy matches
    ``id = app.current_org()``, so WITH CHECK passes and no privileged bypass is
    needed to bootstrap a tenant -- if this ever requires one, the policies are
    wrong.
    """
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()

    try:
        async with tenant_session(org_id, "user", user_id) as session:
            session.add(Organization(id=org_id, name=org_name, slug=slug))
            await session.flush()
            session.add(
                User(
                    id=user_id,
                    org_id=org_id,
                    email=admin_email,
                    hashed_password=hash_password(admin_password),
                    full_name=admin_full_name,
                    role=UserRole.ADMIN,
                )
            )
    except IntegrityError as exc:
        # Slug and email both carry unique constraints. Which one collided is not
        # disclosed: it would turn signup into an account-existence oracle.
        log.info("auth.register_conflict", slug=slug, exc_info=exc)
        raise ConflictError("That organization name or email is already taken.") from exc

    log.info("auth.org_registered", org_id=str(org_id), slug=slug)
    return AuthenticatedUser(user_id=user_id, org_id=org_id, role=UserRole.ADMIN.value)


# --- Login ------------------------------------------------------------------


async def authenticate(session: AsyncSession, *, email: str, password: str) -> AuthenticatedUser:
    """Resolve an email+password to a principal, or raise a uniform 401.

    ``session`` must be *unscoped*: at this point there is no org to scope to.
    Only ``app.lookup_user_for_auth`` is reachable from it under RLS.
    """
    row = (await session.execute(_LOOKUP_SQL, {"email": email})).one_or_none()

    if row is None:
        # Keep both branches KDF-bound so timing does not reveal registration.
        verify_password_dummy()
        raise InvalidCredentialsError(reason="unknown_email")

    matched, upgraded = verify_and_upgrade(password, row.hashed_password)
    if not matched:
        raise InvalidCredentialsError(reason="bad_password", user_id=str(row.id))
    if not row.is_active:
        raise InvalidCredentialsError(reason="user_inactive", user_id=str(row.id))
    if not row.org_active:
        raise InvalidCredentialsError(reason="org_inactive", org_id=str(row.org_id))

    # Post-authentication writes need the org context the token has not been
    # issued for yet, so they run on their own scoped session.
    async with tenant_session(row.org_id, "user", row.id) as scoped:
        if upgraded is not None:
            await scoped.execute(_UPGRADE_HASH_SQL, {"hash": upgraded, "id": row.id})
        await scoped.execute(_TOUCH_LOGIN_SQL, {"id": row.id})

    log.info("auth.login", user_id=str(row.id), org_id=str(row.org_id))
    return AuthenticatedUser(user_id=row.id, org_id=row.org_id, role=row.role)


async def issue_token_pair(
    store: RefreshTokenStore, principal: AuthenticatedUser, *, family_id: str | None = None
) -> TokenPair:
    """Mint an access/refresh pair, registering the refresh token's family."""
    family = family_id or store.new_family()
    access, expires_in = create_access_token(
        user_id=principal.user_id, org_id=principal.org_id, role=principal.role
    )
    refresh, jti = create_refresh_token(
        user_id=principal.user_id, org_id=principal.org_id, family_id=family
    )
    await store.register(
        jti=jti, user_id=principal.user_id, org_id=principal.org_id, family_id=family
    )
    return TokenPair(access_token=access, refresh_token=refresh, expires_in=expires_in)


# --- Rotation ---------------------------------------------------------------


async def rotate_refresh(store: RefreshTokenStore, raw_refresh_token: str) -> TokenPair:
    """Trade a refresh token for a new pair, or fail closed.

    The role is re-read from the database on every rotation rather than carried
    in the refresh token, so a demotion or deactivation takes effect within one
    access-token lifetime instead of persisting for the full refresh TTL.
    """
    claims = RefreshClaims.parse(decode_token(raw_refresh_token, TokenType.REFRESH))

    result = await store.consume(
        jti=claims.jti, user_id=claims.user_id, family_id=claims.family_id
    )
    if not result.ok:
        # The store has already tombstoned the family and logged the reason. The
        # client learns only that the token is unusable.
        raise InvalidTokenError()

    async with tenant_session(claims.org_id, "user", claims.user_id) as session:
        user = await session.get(User, claims.user_id)
        if user is None or not user.is_active:
            # Deactivated between rotations. Kill the family rather than leaving
            # a valid refresh token in the deactivated user's hands.
            await store.revoke_family(claims.family_id)
            raise InvalidTokenError()
        role = user.role.value

    return await issue_token_pair(
        store,
        AuthenticatedUser(user_id=claims.user_id, org_id=claims.org_id, role=role),
        family_id=claims.family_id,
    )


# --- Logout -----------------------------------------------------------------


async def logout(store: RefreshTokenStore, raw_refresh_token: str) -> None:
    """Revoke one session. Idempotent, and never reports failure.

    A logout that 401s on an already-dead token teaches an attacker which tokens
    are live, and gives an honest client no way to finish signing out.
    """
    try:
        claims = RefreshClaims.parse(decode_token(raw_refresh_token, TokenType.REFRESH))
    except InvalidTokenError:
        return
    await store.revoke_family(claims.family_id)


async def logout_all(store: RefreshTokenStore, user_id: uuid.UUID) -> int:
    """Revoke every session for a user. Requires a valid access token."""
    return await store.revoke_all_for_user(user_id)
