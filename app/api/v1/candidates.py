"""Candidate CRUD and resume upload.

Resume upload is candidate-facing and lands in a later slice; everything here is
recruiter-facing. Both roles may read and write candidates, because sourcing is
the recruiter's day job -- admin-only would make the product unusable.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import Principal, ScopedSession, require_role
from app.models.user import UserRole
from app.modules.users import service as users_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, Page
from app.schemas.user import CandidateCreate, CandidateRead, CandidateUpdate

router = APIRouter(prefix="/candidates", tags=["candidates"])

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]


@router.get("", response_model=Page[CandidateRead], summary="List candidates")
async def list_candidates(
    db: ScopedSession,
    _: Recruiter,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
    search: Annotated[str | None, Query(max_length=200)] = None,
) -> Page[CandidateRead]:
    items, total = await users_service.list_candidates(
        db, limit=limit, offset=offset, search=search
    )
    return Page[CandidateRead](
        items=[CandidateRead.model_validate(c) for c in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "",
    response_model=CandidateRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a candidate",
)
async def create_candidate(
    payload: CandidateCreate, db: ScopedSession, principal: Recruiter
) -> CandidateRead:
    # org_id comes from the token, never from the body.
    candidate = await users_service.create_candidate(
        db,
        org_id=principal.org_id,
        email=payload.email,
        full_name=payload.full_name,
        phone=payload.phone,
        external_ref=payload.external_ref,
    )
    return CandidateRead.model_validate(candidate)


@router.get("/{candidate_id}", response_model=CandidateRead, summary="Get one candidate")
async def get_candidate(
    candidate_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> CandidateRead:
    return CandidateRead.model_validate(await users_service.get_candidate(db, candidate_id))


@router.patch("/{candidate_id}", response_model=CandidateRead, summary="Update a candidate")
async def update_candidate(
    candidate_id: uuid.UUID, payload: CandidateUpdate, db: ScopedSession, _: Recruiter
) -> CandidateRead:
    candidate = await users_service.update_candidate(
        db,
        candidate_id=candidate_id,
        full_name=payload.full_name,
        phone=payload.phone,
        external_ref=payload.external_ref,
        fields_set=payload.model_fields_set,
    )
    return CandidateRead.model_validate(candidate)


@router.delete(
    "/{candidate_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a candidate and their interviews",
)
async def delete_candidate(candidate_id: uuid.UUID, db: ScopedSession, _: Recruiter) -> None:
    # A real delete, cascading: this is the erasure path for personal data.
    await users_service.delete_candidate(db, candidate_id)
