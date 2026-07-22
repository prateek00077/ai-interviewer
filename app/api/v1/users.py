"""Recruiter CRUD and org membership.

Reading the roster is open to any authenticated user -- a recruiter needs to see
who else is on their team. Changing it is admin-only. ``require_role`` already
rejects candidate tokens, so no route here has to think about them.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import Principal, ScopedSession, get_current_user, require_role
from app.models.user import UserRole
from app.modules.users import service as users_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, Page
from app.schemas.user import UserCreate, UserRead, UserUpdate

router = APIRouter(prefix="/users", tags=["users"])

CurrentUser = Annotated[Principal, Depends(get_current_user)]
AdminOnly = Annotated[Principal, Depends(require_role(UserRole.ADMIN))]


@router.get("", response_model=Page[UserRead], summary="List recruiters in the org")
async def list_users(
    db: ScopedSession,
    _: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_inactive: bool = True,
) -> Page[UserRead]:
    items, total = await users_service.list_users(
        db, limit=limit, offset=offset, include_inactive=include_inactive
    )
    return Page[UserRead](
        items=[UserRead.model_validate(u) for u in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a recruiter to the org",
)
async def create_user(
    payload: UserCreate, db: ScopedSession, principal: AdminOnly
) -> UserRead:
    # org_id comes from the admin's token: there is no way to plant a user in
    # another tenant, and RLS would reject the INSERT even if there were.
    user = await users_service.create_user(
        db,
        org_id=principal.org_id,
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        role=payload.role,
    )
    return UserRead.model_validate(user)


@router.get("/{user_id}", response_model=UserRead, summary="Get one recruiter")
async def get_user(user_id: uuid.UUID, db: ScopedSession, _: CurrentUser) -> UserRead:
    # Another org's user id is simply not visible under RLS, so this 404s rather
    # than 403s -- the caller learns nothing about whether the row exists.
    return UserRead.model_validate(await users_service.get_user(db, user_id))


@router.patch("/{user_id}", response_model=UserRead, summary="Update a recruiter")
async def update_user(
    user_id: uuid.UUID, payload: UserUpdate, db: ScopedSession, principal: AdminOnly
) -> UserRead:
    user = await users_service.update_user(
        db,
        user_id=user_id,
        acting_user_id=principal.actor_id,
        full_name=payload.full_name,
        role=payload.role,
        is_active=payload.is_active,
        # What the client actually sent, so null and omitted stay distinct.
        fields_set=payload.model_fields_set,
    )
    return UserRead.model_validate(user)


@router.delete(
    "/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Deactivate a recruiter",
)
async def deactivate_user(user_id: uuid.UUID, db: ScopedSession, principal: AdminOnly) -> None:
    # Deactivation, not deletion: the invites and jobs this person created must
    # keep pointing at a real row.
    await users_service.deactivate_user(db, user_id=user_id, acting_user_id=principal.actor_id)
