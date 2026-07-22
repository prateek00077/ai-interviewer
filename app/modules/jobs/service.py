"""Job lifecycle.

Descriptions are append-only. ``add_description`` never updates the previous row;
it writes a new version and moves the ``is_active`` flag. A question plan
generated last week therefore still points at the exact text it was derived from,
even after the recruiter rewrites the posting.
"""

import uuid

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError, NotFoundError
from app.models.job import EmploymentType, Job, JobDescription, JobStatus
from app.modules.users.service import paginate

log = structlog.get_logger(__name__)


# --- Jobs -------------------------------------------------------------------


async def list_jobs(
    session: AsyncSession, *, limit: int, offset: int, status: JobStatus | None = None
) -> tuple[list[Job], int]:
    stmt = select(Job)
    if status is not None:
        stmt = stmt.where(Job.status == status)
    return await paginate(
        session, stmt.order_by(Job.created_at.desc(), Job.id), limit=limit, offset=offset
    )


async def get_job(session: AsyncSession, job_id: uuid.UUID) -> Job:
    job = await session.get(Job, job_id)
    if job is None:
        raise NotFoundError("Job not found.", job_id=str(job_id))
    return job


async def create_job(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    created_by_user_id: uuid.UUID,
    title: str,
    department: str | None = None,
    location: str | None = None,
    employment_type: EmploymentType = EmploymentType.FULL_TIME,
    status: JobStatus = JobStatus.DRAFT,
) -> Job:
    job = Job(
        org_id=org_id,
        created_by_user_id=created_by_user_id,
        title=title,
        department=department,
        location=location,
        employment_type=employment_type,
        status=status,
    )
    session.add(job)
    await session.flush()
    log.info("job_created", job_id=str(job.id), title=title)
    return job


async def update_job(
    session: AsyncSession,
    *,
    job_id: uuid.UUID,
    title: str | None = None,
    department: str | None = None,
    location: str | None = None,
    employment_type: EmploymentType | None = None,
    status: JobStatus | None = None,
    fields_set: set[str] | None = None,
) -> Job:
    fields = fields_set if fields_set is not None else set()
    job = await get_job(session, job_id)

    if "title" in fields and title is not None:
        job.title = title
    if "department" in fields:
        job.department = department
    if "location" in fields:
        job.location = location
    if "employment_type" in fields and employment_type is not None:
        job.employment_type = employment_type
    if "status" in fields and status is not None:
        job.status = status

    await session.flush()
    return job


async def delete_job(session: AsyncSession, job_id: uuid.UUID) -> None:
    """Cascades to the job's descriptions.

    Interviews carry ``job_id`` as a plain column with no foreign key, so a
    completed interview and its report survive the job being deleted -- the
    scored record of a conversation should not disappear because a role closed.
    """
    job = await get_job(session, job_id)
    await session.delete(job)
    await session.flush()
    log.info("job_deleted", job_id=str(job_id))


# --- Descriptions -----------------------------------------------------------


async def list_descriptions(session: AsyncSession, job_id: uuid.UUID) -> list[JobDescription]:
    await get_job(session, job_id)  # 404 for another org's job, not an empty list
    rows = (
        await session.execute(
            select(JobDescription)
            .where(JobDescription.job_id == job_id)
            .order_by(JobDescription.version.desc())
        )
    ).scalars()
    return list(rows)


async def get_active_description(
    session: AsyncSession, job_id: uuid.UUID
) -> JobDescription | None:
    """The version a new question plan should be generated from."""
    return (
        await session.execute(
            select(JobDescription).where(
                JobDescription.job_id == job_id, JobDescription.is_active.is_(True)
            )
        )
    ).scalar_one_or_none()


async def add_description(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    job_id: uuid.UUID,
    created_by_user_id: uuid.UUID,
    content: str,
    activate: bool = True,
) -> JobDescription:
    """Append a new version, optionally making it the active one."""
    await get_job(session, job_id)

    next_version = int(
        await session.scalar(
            select(func.coalesce(func.max(JobDescription.version), 0) + 1).where(
                JobDescription.job_id == job_id
            )
        )
        or 1
    )

    if activate:
        await _clear_active(session, job_id)

    description = JobDescription(
        org_id=org_id,
        job_id=job_id,
        created_by_user_id=created_by_user_id,
        version=next_version,
        content=content,
        is_active=activate,
    )
    session.add(description)
    try:
        await session.flush()
    except IntegrityError as exc:
        # Either uq(job_id, version) or the partial unique "one active per job".
        # Both mean a concurrent writer got there first; the client should re-read
        # and retry rather than have us silently overwrite their work.
        raise ConflictError(
            "The job description changed concurrently. Reload and try again.",
            job_id=str(job_id),
        ) from exc

    log.info("job_description_added", job_id=str(job_id), version=next_version, active=activate)
    return description


async def activate_description(
    session: AsyncSession, *, job_id: uuid.UUID, description_id: uuid.UUID
) -> JobDescription:
    """Roll back to an earlier version."""
    description = await session.get(JobDescription, description_id)
    if description is None or description.job_id != job_id:
        raise NotFoundError("Job description not found.", description_id=str(description_id))

    if not description.is_active:
        await _clear_active(session, job_id)
        description.is_active = True
        await session.flush()
    return description


async def _clear_active(session: AsyncSession, job_id: uuid.UUID) -> None:
    """Deactivate first, in the same transaction as the activation.

    The partial unique index rejects two active rows, so this UPDATE must land
    before the INSERT rather than after it.
    """
    await session.execute(
        update(JobDescription)
        .where(JobDescription.job_id == job_id, JobDescription.is_active.is_(True))
        .values(is_active=False)
    )
    await session.flush()
