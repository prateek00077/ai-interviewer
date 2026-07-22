"""User (RECRUITER | ADMIN) and Candidate."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, Enum, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantMixin, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from app.models.interview import Interview
    from app.models.org import Organization


class UserRole(enum.StrEnum):
    ADMIN = "ADMIN"
    RECRUITER = "RECRUITER"


user_role_enum = Enum(
    UserRole,
    name="user_role",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,  # the migration owns CREATE TYPE
)


class User(Base, TenantMixin, TimestampMixin):
    """A recruiter or admin. Authenticates with email + password."""

    __tablename__ = "users"
    __table_args__ = (
        # Global uniqueness is required: login receives an email with no org, so
        # the address must resolve to exactly one user across all tenants.
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("org_id", "email", name="uq_users_org_id_email"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    email: Mapped[str] = mapped_column(CITEXT(), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(200))
    role: Mapped[UserRole] = mapped_column(user_role_enum, nullable=False)
    is_active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    organization: Mapped["Organization"] = relationship(back_populates="users")

    def __repr__(self) -> str:
        return f"<User {self.email} {self.role}>"


class Candidate(Base, TenantMixin, TimestampMixin):
    """An interviewee. Deliberately has no password column.

    Candidates never hold credentials: they arrive via an invite link and trade
    it for a 10-minute interview token. Per-org by design -- someone
    interviewing at two companies is two rows, which keeps every RLS policy a
    single ``org_id`` predicate.
    """

    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint("org_id", "email", name="uq_candidates_org_id_email"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    email: Mapped[str] = mapped_column(CITEXT(), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(200))
    phone: Mapped[str | None] = mapped_column(String(50))
    external_ref: Mapped[str | None] = mapped_column(String(200))

    organization: Mapped["Organization"] = relationship(back_populates="candidates")
    interviews: Mapped[list["Interview"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Candidate {self.email}>"
