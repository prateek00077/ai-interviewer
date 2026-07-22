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
TENANT_TABLES: list[str] = ["users", "candidates", "interviews", "invites"]

# Tables where org membership is not sufficient: a candidate actor is narrowed to
# rows it owns. Maps table -> the column holding the owning candidate id.
CANDIDATE_SCOPED: dict[str, str] = {"interviews": "candidate_id", "candidates": "id"}

# Tables a candidate actor must never read at all.
USER_ONLY_TABLES: list[str] = ["users", "invites"]
