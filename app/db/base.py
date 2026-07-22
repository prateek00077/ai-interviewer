"""Declarative Base, TimestampMixin, TenantMixin(org_id).

``TENANT_TABLES`` is the single registry of RLS-protected tables. ``db/rls.py``,
the policy migration and ``tests/integration/test_rls.py`` all read from it, so a
new tenant table cannot ship without a policy -- the coverage guard test fails.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, MetaData, func
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

# Explicit naming lets Alembic autogenerate stable names for constraints and
# indexes instead of emitting unnamed ones it can never drop on downgrade.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    # Server-side defaults (gen_random_uuid, now(), and especially the
    # onupdate=now() on updated_at) are expired after a flush and reloaded on next
    # access. Under asyncio that reload is a lazy IO, so serializing a
    # just-updated row raises MissingGreenlet instead of returning a timestamp.
    # eager_defaults makes the flush fetch them inline via RETURNING, which costs
    # nothing extra on Postgres -- the INSERT/UPDATE already round-trips.
    __mapper_args__ = {"eager_defaults": True}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class TenantMixin:
    """Marks a table as org-scoped. Every such table gets an RLS policy."""

    @declared_attr
    @classmethod
    def org_id(cls) -> Mapped[uuid.UUID]:
        return mapped_column(
            PGUUID(as_uuid=True),
            ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        )


def uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )


# Tables carrying org_id, filtered by org alone.
TENANT_TABLES: list[str] = [
    "users",
    "candidates",
    "interviews",
    "interview_turns",
    "invites",
    "jobs",
    "job_descriptions",
    "resumes",
    "resume_chunks",
    "question_plans",
    "planned_questions",
    "rubric_criteria",
]

# Tables where org membership is not sufficient: a candidate actor is narrowed to
# rows it owns. Maps table -> the column holding the owning candidate id.
CANDIDATE_SCOPED: dict[str, str] = {"interviews": "candidate_id", "candidates": "id"}

# Candidate-scoped AND candidate-writable. Separate from CANDIDATE_SCOPED because
# read-own and write-own are different grants: a candidate reads their interview
# but must never create one. Resumes are the one thing a candidate legitimately
# writes, since they are the only person who has the file.
CANDIDATE_WRITABLE: dict[str, str] = {"resumes": "candidate_id"}

# Tables a candidate actor must never read at all.
#
# Jobs are here because a candidate has no business enumerating an org's open
# roles, headcount or salary bands. The interview pipeline does read the job
# description, but it does so server-side under the recruiter/system context that
# assembles the prompt -- never through the candidate's own token.
#
# resume_chunks is here too: the chunk text and its embedding are derived data
# the recruiter's pipeline reads. A candidate has no reason to page through the
# vector representation of their own CV, and not exposing it keeps the retrieval
# index off the candidate-facing attack surface entirely.
#
# The question plan tables are the answer key. A candidate who can read their
# own plan knows the questions and the weights before the interview starts,
# which defeats the entire product.
#
# interview_turns is staff-only for now. A candidate reading back their own
# transcript is defensible in principle, but nothing in the product asks for it,
# and the policy it would need -- "turns of interviews this candidate owns" --
# is a per-row subquery against `interviews`. Add it when a feature wants it.
USER_ONLY_TABLES: list[str] = [
    "users",
    "interview_turns",
    "invites",
    "jobs",
    "job_descriptions",
    "resume_chunks",
    "question_plans",
    "planned_questions",
    "rubric_criteria",
]
