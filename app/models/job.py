"""Job posting and its description.

The description is a separate, versioned table rather than a column on ``jobs``.
Two reasons, both load-bearing later:

- A question plan is generated from one specific description. If the recruiter
  rewrites the posting mid-hiring-round, every plan generated before that edit
  must still point at the text it actually read, or the rubric stops matching the
  questions it was derived from.
- Recruiters edit job descriptions repeatedly and want the previous wording back.

Exactly one description per job is ``is_active``, enforced by a partial unique
index rather than by application code -- two concurrent activations would
otherwise both read "none active" and both write.
"""

import enum
import uuid

from sqlalchemy import (
    Boolean,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantMixin, TimestampMixin, uuid_pk


class JobStatus(enum.StrEnum):
    DRAFT = "DRAFT"
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class EmploymentType(enum.StrEnum):
    FULL_TIME = "FULL_TIME"
    PART_TIME = "PART_TIME"
    CONTRACT = "CONTRACT"
    INTERNSHIP = "INTERNSHIP"


job_status_enum = Enum(
    JobStatus,
    name="job_status",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,  # the migration owns CREATE TYPE
)
employment_type_enum = Enum(
    EmploymentType,
    name="employment_type",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)


class Job(Base, TenantMixin, TimestampMixin):
    """A role being hired for. Interviews hang off it."""

    __tablename__ = "jobs"
    __table_args__ = (
        # Listing a hiring pipeline is always "this org's open jobs, newest first".
        Index("ix_jobs_org_id_status", "org_id", "status"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    department: Mapped[str | None] = mapped_column(String(120))
    location: Mapped[str | None] = mapped_column(String(200))
    employment_type: Mapped[EmploymentType] = mapped_column(
        employment_type_enum, nullable=False, server_default=text("'FULL_TIME'")
    )
    status: Mapped[JobStatus] = mapped_column(
        job_status_enum, nullable=False, server_default=text("'DRAFT'")
    )
    # SET NULL, not CASCADE: deactivating a recruiter must not delete the roles
    # they opened.
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    descriptions: Mapped[list["JobDescription"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
        order_by="desc(JobDescription.version)",
    )

    def __repr__(self) -> str:
        return f"<Job {self.title} {self.status}>"


class JobDescription(Base, TenantMixin, TimestampMixin):
    """One immutable version of a job's description text.

    ``requirements`` is the structured extraction the plan generator reads;
    ``content`` is what the recruiter actually typed. Both are kept: the parse can
    be re-run and improved, the source text cannot be recovered from it.
    """

    __tablename__ = "job_descriptions"
    __table_args__ = (
        UniqueConstraint("job_id", "version", name="uq_job_descriptions_job_id_version"),
        # At most one active description per job, decided by Postgres rather than
        # by a read-then-write in application code.
        Index(
            "ix_job_descriptions_one_active_per_job",
            "job_id",
            unique=True,
            postgresql_where=text("is_active"),
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Filled by the plan generator in a later slice; empty until then.
    requirements: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    job: Mapped["Job"] = relationship(back_populates="descriptions")

    def __repr__(self) -> str:
        return f"<JobDescription job={self.job_id} v{self.version}>"
