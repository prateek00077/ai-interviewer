"""Recruiter and candidate management.

Every function takes an already org-scoped session. Nothing here filters on
``org_id`` by hand: RLS is the tenant boundary, and duplicating it in Python
would create a second place for it to be wrong. A row belonging to another org is
therefore simply not found, which is why cross-tenant access surfaces as 404
rather than 403 -- the caller is not told the row exists.
"""

import uuid
from typing import Any

import structlog
from sqlalchemy import Select, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError, PermissionDeniedError
from app.core.security import hash_password
from app.models.user import Candidate, User, UserRole

log = structlog.get_logger(__name__)


async def paginate(
    session: AsyncSession, stmt: Select[Any], *, limit: int, offset: int
) -> tuple[list[Any], int]:
    """Run a query twice: once for the page, once for the unpaginated total.

    ``order_by`` is applied by the caller and must be deterministic -- without a
    unique tiebreaker, OFFSET pagination can show one row on two pages and skip
    another entirely.
    """
    total = await session.scalar(select(func.count()).select_from(stmt.order_by(None).subquery()))
    rows = (await session.execute(stmt.limit(limit).offset(offset))).scalars().all()
    return list(rows), int(total or 0)


# --- Users ------------------------------------------------------------------


async def list_users(
    session: AsyncSession, *, limit: int, offset: int, include_inactive: bool = True
) -> tuple[list[User], int]:
    stmt = select(User)
    if not include_inactive:
        stmt = stmt.where(User.is_active.is_(True))
    return await paginate(
        session, stmt.order_by(User.created_at.desc(), User.id), limit=limit, offset=offset
    )


async def create_user(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    email: str,
    password: str,
    full_name: str | None = None,
    role: UserRole = UserRole.RECRUITER,
) -> User:
    """Add a teammate with a password the admin sets.

    An emailed invite-and-set-your-own-password flow is better and lands with the
    email integration. Until then this is the honest version: the admin knows the
    initial credential, which is why the account is created active and the
    password is never echoed back in any response.
    """
    user = User(
        org_id=org_id,
        email=email,
        hashed_password=hash_password(password),
        full_name=full_name,
        role=role,
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError as exc:
        # uq_users_email is global, not per-org: login resolves an address with no
        # org context, so one address must map to exactly one user everywhere.
        # The message stays vague -- confirming an address is registered elsewhere
        # would turn this endpoint into an account-enumeration oracle.
        raise ConflictError("That email address is not available.") from exc

    log.info("user_created", user_id=str(user.id), role=role.value)
    return user


async def get_user(session: AsyncSession, user_id: uuid.UUID) -> User:
    user = await session.get(User, user_id)
    if user is None:
        raise NotFoundError("User not found.", user_id=str(user_id))
    return user


async def update_user(
    session: AsyncSession,
    *,
    user_id: uuid.UUID,
    acting_user_id: uuid.UUID,
    full_name: str | None = None,
    role: UserRole | None = None,
    is_active: bool | None = None,
    fields_set: set[str] | None = None,
) -> User:
    """Apply an admin edit.

    ``fields_set`` is the set of keys the client actually sent, so clearing
    ``full_name`` to null stays distinguishable from omitting it.
    """
    fields = fields_set if fields_set is not None else set()
    user = await get_user(session, user_id)

    # An admin who demotes or deactivates themselves can lock the tenant out of
    # its own admin surface, and the recovery path is a support ticket. Refuse.
    if user.id == acting_user_id:
        if "role" in fields and role is not None and role is not UserRole.ADMIN:
            raise PermissionDeniedError("You cannot remove your own admin role.")
        if "is_active" in fields and is_active is False:
            raise PermissionDeniedError("You cannot deactivate your own account.")

    if "full_name" in fields:
        user.full_name = full_name
    if "role" in fields and role is not None:
        await _guard_last_admin(session, user, new_role=role)
        user.role = role
    if "is_active" in fields and is_active is not None:
        if is_active is False:
            await _guard_last_admin(session, user, deactivating=True)
        user.is_active = is_active

    await session.flush()
    log.info("user_updated", user_id=str(user.id), fields=sorted(fields))
    return user


async def deactivate_user(
    session: AsyncSession, *, user_id: uuid.UUID, acting_user_id: uuid.UUID
) -> User:
    """Soft delete.

    A hard DELETE would cascade away the invites and jobs this person created,
    rewriting the record of who did what. Deactivation also takes effect at the
    next token refresh, since the refresh path re-reads the user.
    """
    return await update_user(
        session,
        user_id=user_id,
        acting_user_id=acting_user_id,
        is_active=False,
        fields_set={"is_active"},
    )


async def _guard_last_admin(
    session: AsyncSession,
    user: User,
    *,
    new_role: UserRole | None = None,
    deactivating: bool = False,
) -> None:
    """An org with no active admin cannot invite, promote or configure anything."""
    losing_admin = user.role is UserRole.ADMIN and (
        deactivating or (new_role is not None and new_role is not UserRole.ADMIN)
    )
    if not losing_admin or not user.is_active:
        return

    remaining = await session.scalar(
        select(func.count())
        .select_from(User)
        .where(User.role == UserRole.ADMIN, User.is_active.is_(True), User.id != user.id)
    )
    if not remaining:
        raise ConflictError("An organization must keep at least one active admin.")


# --- Candidates -------------------------------------------------------------


async def list_candidates(
    session: AsyncSession, *, limit: int, offset: int, search: str | None = None
) -> tuple[list[Candidate], int]:
    stmt = select(Candidate)
    if search:
        # citext already makes the email side case-insensitive; full_name needs ilike.
        pattern = f"%{search}%"
        stmt = stmt.where(Candidate.email.ilike(pattern) | Candidate.full_name.ilike(pattern))
    return await paginate(
        session,
        stmt.order_by(Candidate.created_at.desc(), Candidate.id),
        limit=limit,
        offset=offset,
    )


async def get_candidate(session: AsyncSession, candidate_id: uuid.UUID) -> Candidate:
    candidate = await session.get(Candidate, candidate_id)
    if candidate is None:
        raise NotFoundError("Candidate not found.", candidate_id=str(candidate_id))
    return candidate


async def create_candidate(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    email: str,
    full_name: str | None = None,
    phone: str | None = None,
    external_ref: str | None = None,
) -> Candidate:
    candidate = Candidate(
        org_id=org_id,
        email=email,
        full_name=full_name,
        phone=phone,
        external_ref=external_ref,
    )
    session.add(candidate)
    try:
        await session.flush()
    except IntegrityError as exc:
        # Let the unique index decide rather than a prior SELECT: two concurrent
        # imports of the same candidate would both pass a check-then-insert.
        raise ConflictError("A candidate with that email already exists.", email=email) from exc
    return candidate


async def update_candidate(
    session: AsyncSession,
    *,
    candidate_id: uuid.UUID,
    full_name: str | None = None,
    phone: str | None = None,
    external_ref: str | None = None,
    fields_set: set[str] | None = None,
) -> Candidate:
    fields = fields_set if fields_set is not None else set()
    candidate = await get_candidate(session, candidate_id)

    if "full_name" in fields:
        candidate.full_name = full_name
    if "phone" in fields:
        candidate.phone = phone
    if "external_ref" in fields:
        candidate.external_ref = external_ref

    await session.flush()
    return candidate


async def delete_candidate(session: AsyncSession, candidate_id: uuid.UUID) -> None:
    """Hard delete, cascading to their interviews.

    Unlike a recruiter, a candidate row is personal data an org may be obliged to
    erase on request, so the cascade is the point rather than a side effect.
    """
    candidate = await get_candidate(session, candidate_id)
    await session.delete(candidate)
    await session.flush()
    log.info("candidate_deleted", candidate_id=str(candidate_id))
