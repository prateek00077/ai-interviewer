"""Job and job-description schemas.

Update payloads use ``model_fields_set`` (via ``exclude_unset`` at the call site)
rather than sentinel defaults, so "field omitted" and "field explicitly set to
null" stay distinguishable. Clearing a department and not mentioning it are
different requests.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.job import EmploymentType, JobStatus

MAX_DESCRIPTION_CHARS = 50_000


class JobCreate(BaseModel):
    title: str = Field(min_length=2, max_length=200)
    department: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=200)
    employment_type: EmploymentType = EmploymentType.FULL_TIME
    status: JobStatus = JobStatus.DRAFT


class JobUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=2, max_length=200)
    department: str | None = Field(default=None, max_length=120)
    location: str | None = Field(default=None, max_length=200)
    employment_type: EmploymentType | None = None
    status: JobStatus | None = None


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    title: str
    department: str | None = None
    location: str | None = None
    employment_type: EmploymentType
    status: JobStatus
    created_by_user_id: uuid.UUID | None = None
    created_at: datetime
    updated_at: datetime


class JobDescriptionCreate(BaseModel):
    # Bounded because this text is later sent to an LLM as a prompt, where an
    # unbounded field is a cost and context-window problem, not just a DB one.
    content: str = Field(min_length=20, max_length=MAX_DESCRIPTION_CHARS)
    # A new description is the one a plan should be generated from, so activating
    # on create is the useful default rather than an extra round trip.
    activate: bool = True


class JobDescriptionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_id: uuid.UUID
    version: int
    content: str
    requirements: dict
    is_active: bool
    created_by_user_id: uuid.UUID | None = None
    created_at: datetime
