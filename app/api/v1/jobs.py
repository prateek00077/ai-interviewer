"""Job and job-description CRUD.

Descriptions are nested under their job rather than exposed as a top-level
resource: a description has no meaning apart from the job it versions, and
nesting keeps the "does this description belong to this job" check on the path
instead of in every handler.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.deps import Principal, ScopedSession, require_role
from app.models.job import JobStatus
from app.models.user import UserRole
from app.modules.jobs import service as jobs_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, Page
from app.schemas.job import (
    JobCreate,
    JobDescriptionCreate,
    JobDescriptionRead,
    JobRead,
    JobUpdate,
)

router = APIRouter(prefix="/jobs", tags=["jobs"])

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]


# --- Jobs -------------------------------------------------------------------


@router.get("", response_model=Page[JobRead], summary="List jobs")
async def list_jobs(
    db: ScopedSession,
    _: Recruiter,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
    offset: Annotated[int, Query(ge=0)] = 0,
    status_filter: Annotated[JobStatus | None, Query(alias="status")] = None,
) -> Page[JobRead]:
    items, total = await jobs_service.list_jobs(
        db, limit=limit, offset=offset, status=status_filter
    )
    return Page[JobRead](
        items=[JobRead.model_validate(j) for j in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "", response_model=JobRead, status_code=status.HTTP_201_CREATED, summary="Create a job"
)
async def create_job(payload: JobCreate, db: ScopedSession, principal: Recruiter) -> JobRead:
    job = await jobs_service.create_job(
        db,
        org_id=principal.org_id,
        created_by_user_id=principal.actor_id,
        title=payload.title,
        department=payload.department,
        location=payload.location,
        employment_type=payload.employment_type,
        status=payload.status,
    )
    return JobRead.model_validate(job)


@router.get("/{job_id}", response_model=JobRead, summary="Get one job")
async def get_job(job_id: uuid.UUID, db: ScopedSession, _: Recruiter) -> JobRead:
    return JobRead.model_validate(await jobs_service.get_job(db, job_id))


@router.patch("/{job_id}", response_model=JobRead, summary="Update a job")
async def update_job(
    job_id: uuid.UUID, payload: JobUpdate, db: ScopedSession, _: Recruiter
) -> JobRead:
    job = await jobs_service.update_job(
        db,
        job_id=job_id,
        title=payload.title,
        department=payload.department,
        location=payload.location,
        employment_type=payload.employment_type,
        status=payload.status,
        fields_set=payload.model_fields_set,
    )
    return JobRead.model_validate(job)


@router.delete(
    "/{job_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a job"
)
async def delete_job(job_id: uuid.UUID, db: ScopedSession, _: Recruiter) -> None:
    await jobs_service.delete_job(db, job_id)


# --- Descriptions -----------------------------------------------------------


@router.get(
    "/{job_id}/descriptions",
    response_model=list[JobDescriptionRead],
    summary="List description versions, newest first",
)
async def list_descriptions(
    job_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> list[JobDescriptionRead]:
    rows = await jobs_service.list_descriptions(db, job_id)
    return [JobDescriptionRead.model_validate(d) for d in rows]


@router.post(
    "/{job_id}/descriptions",
    response_model=JobDescriptionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Add a description version",
)
async def add_description(
    job_id: uuid.UUID,
    payload: JobDescriptionCreate,
    db: ScopedSession,
    principal: Recruiter,
) -> JobDescriptionRead:
    # Append-only: this never edits the previous version in place, so a question
    # plan generated from an older version still matches the text it read.
    description = await jobs_service.add_description(
        db,
        org_id=principal.org_id,
        job_id=job_id,
        created_by_user_id=principal.actor_id,
        content=payload.content,
        activate=payload.activate,
    )
    return JobDescriptionRead.model_validate(description)


@router.post(
    "/{job_id}/descriptions/{description_id}/activate",
    response_model=JobDescriptionRead,
    summary="Roll back to an earlier description version",
)
async def activate_description(
    job_id: uuid.UUID, description_id: uuid.UUID, db: ScopedSession, _: Recruiter
) -> JobDescriptionRead:
    description = await jobs_service.activate_description(
        db, job_id=job_id, description_id=description_id
    )
    return JobDescriptionRead.model_validate(description)
