"""Resume and ResumeChunk (pgvector embedding column).

The upload is a two-step handshake, and ``status`` is what makes it safe:

    PENDING  -- a presigned URL was issued; nothing has been uploaded yet
    UPLOADED -- the object exists and passed a server-side HEAD check
    PARSING  -- a worker has it
    READY    -- text extracted, chunked, embedded
    FAILED   -- unparseable; ``error`` says why

A row starts at PENDING with a key that may never be written to. Nothing
downstream may treat a PENDING resume as real, which is why the transition to
UPLOADED happens only after the API has itself confirmed size and content type
with S3 -- never because the client said so.

Chunks live in their own table rather than as an array column so pgvector can
index them. One row per chunk is what makes ``ORDER BY embedding <=> :query``
an index scan instead of a sequential unnest.
"""

import enum
import uuid
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
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

if TYPE_CHECKING:
    from app.models.user import Candidate

# nv-embedqa-e5-v5 returns 1024 floats. Confirmed against the live endpoint by
# scripts/check_nim.py rather than taken from documentation -- the column width
# is not something to discover in production.
EMBEDDING_DIMENSIONS = 1024


class ResumeStatus(enum.StrEnum):
    PENDING = "PENDING"
    UPLOADED = "UPLOADED"
    PARSING = "PARSING"
    READY = "READY"
    FAILED = "FAILED"


resume_status_enum = Enum(
    ResumeStatus,
    name="resume_status",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,  # the migration owns CREATE TYPE
)


class Resume(Base, TenantMixin, TimestampMixin):
    """One uploaded CV belonging to one candidate."""

    __tablename__ = "resumes"
    __table_args__ = (
        UniqueConstraint("s3_key", name="uq_resumes_s3_key"),
        # A candidate re-uploading gets a new row; "their resume" is the newest
        # READY one, so this index serves the only lookup that matters.
        Index("ix_resumes_candidate_id_created_at", "candidate_id", "created_at"),
        CheckConstraint("size_bytes IS NULL OR size_bytes > 0", name="size_bytes_positive"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_type: Mapped[str] = mapped_column(String(120), nullable=False)
    # Null until the HEAD check fills it in -- the size is S3's answer, not the
    # client's claim.
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[ResumeStatus] = mapped_column(
        resume_status_enum, nullable=False, server_default=text("'PENDING'")
    )
    # Structured extraction: contact, skills, experience, education. Shape is the
    # parser's business, so JSONB rather than a dozen sparse columns.
    parsed: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # Populated only on FAILED. Operator-facing; never rendered to a candidate.
    error: Mapped[str | None] = mapped_column(Text)

    candidate: Mapped["Candidate"] = relationship()
    chunks: Mapped[list["ResumeChunk"]] = relationship(
        back_populates="resume",
        cascade="all, delete-orphan",
        order_by="ResumeChunk.ordinal",
    )

    def __repr__(self) -> str:
        return f"<Resume {self.id} {self.status}>"


class ResumeChunk(Base, TenantMixin, TimestampMixin):
    """One embeddable span of a resume, with its vector.

    ``embedding`` is nullable so chunking and embedding can be separate, retryable
    steps: a failed embedding call leaves the text in place to retry rather than
    forcing the whole document to be re-parsed.
    """

    __tablename__ = "resume_chunks"
    __table_args__ = (
        UniqueConstraint("resume_id", "ordinal", name="uq_resume_chunks_resume_id_ordinal"),
        # HNSW over cosine distance. Created in the migration rather than here,
        # because the operator class (vector_cosine_ops) has no SQLAlchemy
        # spelling that survives autogenerate cleanly.
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    resume_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    # Which part of the CV this came from ("experience", "skills", ...). Lets the
    # retriever prefer experience over a bare skills list when both match.
    section: Mapped[str | None] = mapped_column(String(64))
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Nullable on purpose, per the class docstring: chunking and embedding are
    # separate retryable steps.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))

    resume: Mapped["Resume"] = relationship(back_populates="chunks")

    def __repr__(self) -> str:
        return f"<ResumeChunk {self.resume_id}#{self.ordinal}>"
