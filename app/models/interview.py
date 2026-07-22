"""Interview, InterviewTurn, Invite."""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantMixin, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from app.models.user import Candidate


class InterviewStatus(enum.StrEnum):
    CREATED = "CREATED"
    INVITED = "INVITED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"
    TERMINATED = "TERMINATED"
    EXPIRED = "EXPIRED"


class InviteStatus(enum.StrEnum):
    PENDING = "PENDING"
    REDEEMED = "REDEEMED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


interview_status_enum = Enum(
    InterviewStatus,
    name="interview_status",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)
invite_status_enum = Enum(
    InviteStatus,
    name="invite_status",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)


class Interview(Base, TenantMixin, TimestampMixin):
    __tablename__ = "interviews"

    id: Mapped[uuid.UUID] = uuid_pk()
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Jobs land in a later slice; untyped for now so auth does not depend on it.
    job_id: Mapped[uuid.UUID | None] = mapped_column(PGUUID(as_uuid=True))
    status: Mapped[InterviewStatus] = mapped_column(
        interview_status_enum, nullable=False, server_default=text("'CREATED'")
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    candidate: Mapped["Candidate"] = relationship(back_populates="interviews")
    invites: Mapped[list["Invite"]] = relationship(
        back_populates="interview", cascade="all, delete-orphan"
    )
    turns: Mapped[list["InterviewTurn"]] = relationship(
        back_populates="interview",
        cascade="all, delete-orphan",
        order_by="InterviewTurn.ordinal",
    )

    def __repr__(self) -> str:
        return f"<Interview {self.id} {self.status}>"


class Speaker(enum.StrEnum):
    CANDIDATE = "CANDIDATE"
    INTERVIEWER = "INTERVIEWER"


speaker_enum = Enum(
    Speaker,
    name="speaker",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)


class InterviewTurn(Base, TenantMixin, TimestampMixin):
    """One utterance in the conversation.

    Offsets are milliseconds from session start, not wall-clock timestamps. They
    are what ties a transcript line to a position in the recording, and a
    wall-clock time would drift against the audio the moment anything buffered
    or the process paused.

    ``content`` is the live ASR result and is deliberately mutable: the offline
    full-quality pass corrects the text in place while keeping these timings,
    which are the more trustworthy half of the pair.
    """

    __tablename__ = "interview_turns"
    __table_args__ = (
        # Ordinal is assigned by the voice session, and the constraint is what
        # makes a duplicated event a database error rather than a doubled line
        # in the transcript.
        UniqueConstraint("interview_id", "ordinal", name="uq_interview_turns_interview_id_ordinal"),
        CheckConstraint("ended_offset_ms >= started_offset_ms", name="offsets_ordered"),
        Index("ix_interview_turns_interview_id_ordinal", "interview_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[Speaker] = mapped_column(speaker_enum, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    started_offset_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    ended_offset_ms: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # Which planned question this answers, by ordinal rather than by id: the
    # plan is frozen at session start, so its ordinals are stable, and a foreign
    # key would make a deleted plan cascade away the transcript.
    question_ordinal: Mapped[int | None] = mapped_column(Integer)
    # True once the offline pass has replaced the live ASR text.
    is_final: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    interview: Mapped["Interview"] = relationship(back_populates="turns")

    def __repr__(self) -> str:
        return f"<InterviewTurn {self.interview_id}#{self.ordinal} {self.speaker}>"


class Invite(Base, TenantMixin, TimestampMixin):
    """A magic link, stored so it can be revoked before it expires.

    The invite JWT is a *pointer* to this row: it carries the row id and a
    ``jti`` that must match ``Invite.jti``. Revocation is therefore an UPDATE
    that takes effect immediately, not something that waits for token expiry.
    """

    __tablename__ = "invites"
    __table_args__ = (
        UniqueConstraint("jti", name="uq_invites_jti"),
        CheckConstraint(
            "redemption_count <= max_redemptions", name="redemption_count_within_limit"
        ),
        Index("ix_invites_org_id_interview_id", "org_id", "interview_id"),
        # Supports the expiry reaper without scanning redeemed/revoked rows.
        Index(
            "ix_invites_pending_expires_at",
            "expires_at",
            postgresql_where=text("status = 'PENDING'"),
        ),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
    )
    jti: Mapped[uuid.UUID] = mapped_column(PGUUID(as_uuid=True), nullable=False)
    status: Mapped[InviteStatus] = mapped_column(
        invite_status_enum, nullable=False, server_default=text("'PENDING'")
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    redeemed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    redemption_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # Multi-use by design: a candidate whose browser crashes must be able to
    # rejoin. Each redemption still yields only a 10-minute interview token.
    max_redemptions: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("3")
    )
    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )

    interview: Mapped["Interview"] = relationship(back_populates="invites")

    def __repr__(self) -> str:
        return f"<Invite {self.id} {self.status}>"
