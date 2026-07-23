"""RecruiterReport and CandidateReport.

TWO TABLES, NOT ONE WITH AN ``audience`` COLUMN. This is the single most
important decision in the reporting slice, and it is a safety property rather
than a modelling preference.

The recruiter report contains scores, band, hire recommendation, per-criterion
evidence and the proctoring verdict. The candidate report contains feedback and
gaps and must contain none of that. With one table and an ``audience``
discriminator, the only thing standing between a candidate and their own hire
recommendation is a ``WHERE audience = 'CANDIDATE'`` that someone has to
remember to write, every time, forever.

Split into two, the boundary is enforced by Postgres:

- ``recruiter_reports`` is USER_ONLY. A candidate session reads zero rows from
  it no matter what query is issued.
- ``candidate_reports`` is CANDIDATE_SCOPED on ``candidate_id`` -- read-own,
  and candidates never write.

The generated feedback is stored as structured fields rather than only as a
rendered PDF so the API can serve it as JSON without re-running a model, and so
a test can assert on its contents directly.
"""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantMixin, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from app.models.interview import Interview


class ReportStatus(enum.StrEnum):
    PENDING = "PENDING"
    RENDERING = "RENDERING"
    READY = "READY"
    FAILED = "FAILED"


report_status_enum = Enum(
    ReportStatus,
    name="report_status",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,  # the migration owns CREATE TYPE
)


class RecruiterReport(Base, TenantMixin, TimestampMixin):
    """The full assessment PDF. Staff-only at the RLS layer."""

    __tablename__ = "recruiter_reports"
    __table_args__ = (
        UniqueConstraint("interview_id", name="uq_recruiter_reports_interview_id"),
        Index("ix_recruiter_reports_org_id_status", "org_id", "status"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[ReportStatus] = mapped_column(
        report_status_enum, nullable=False, server_default=text("'PENDING'")
    )
    # Where the PDF landed. Served only as a short-lived presigned URL; the key
    # itself is never handed to a client, because a key plus a bucket guess is
    # most of the way to an object.
    s3_key: Mapped[str | None] = mapped_column(String(512))
    error: Mapped[str | None] = mapped_column(Text)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    interview: Mapped["Interview"] = relationship()

    def __repr__(self) -> str:
        return f"<RecruiterReport {self.interview_id} {self.status}>"


class CandidateReport(Base, TenantMixin, TimestampMixin):
    """Feedback and gaps. Structurally incapable of holding a score.

    There is no ``overall``, no ``recommendation`` and no criterion score on
    this table, and the builder that populates it is never given a ``Score``
    object to read one from. That is the guarantee: not "we remembered to omit
    it" but "there was nothing to omit".
    """

    __tablename__ = "candidate_reports"
    __table_args__ = (
        UniqueConstraint("interview_id", name="uq_candidate_reports_interview_id"),
        Index("ix_candidate_reports_org_id_status", "org_id", "status"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalised from the interview so the RLS policy can narrow a candidate
    # to their own rows without a per-row subquery against `interviews`.
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[ReportStatus] = mapped_column(
        report_status_enum, nullable=False, server_default=text("'PENDING'")
    )

    summary: Mapped[str | None] = mapped_column(Text)
    strengths: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    growth_areas: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    s3_key: Mapped[str | None] = mapped_column(String(512))
    error: Mapped[str | None] = mapped_column(Text)
    generated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    interview: Mapped["Interview"] = relationship()

    def __repr__(self) -> str:
        return f"<CandidateReport {self.interview_id} {self.status}>"
