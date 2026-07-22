"""ProctoringPolicy, ProctoringEvent, ProctoringVerdict.

WHAT PROCTORING IS AND IS NOT. Everything recorded here is a *signal*, not a
finding. A candidate who switched tabs twice may have been checking the calendar
invite; one whose face left frame may have a doorbell. The system's job is to
surface what happened with enough evidence for a human to judge it, which is why
a verdict carries its reasons and why FLAGGED never means "cheated".

That framing decides the schema:

- Events are append-only and timestamped by the SERVER. The browser reports
  them, and the browser is controlled by the person being assessed.
- Severity is assigned by our rules from the policy, never accepted from the
  client, so a candidate cannot report their own tab-switch as "info".
- A verdict is recomputed from events rather than accumulated, so re-running the
  analysis after a rule change produces a defensible answer rather than a
  historical artefact.

All three tables are staff-only. A candidate who can read the policy knows the
blur limit; one who can read their events knows exactly what was noticed.
"""

import enum
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TenantMixin, TimestampMixin, uuid_pk

if TYPE_CHECKING:
    from app.models.interview import Interview
    from app.models.job import Job


class ProctorEventType(enum.StrEnum):
    """What the browser and the pipeline can report.

    Closed set on purpose: an open string field would let a candidate's browser
    invent types that no rule scores and no reviewer recognises.
    """

    TAB_BLUR = "TAB_BLUR"
    TAB_FOCUS = "TAB_FOCUS"
    FULLSCREEN_EXIT = "FULLSCREEN_EXIT"
    PASTE = "PASTE"
    COPY = "COPY"
    DEVTOOLS_OPEN = "DEVTOOLS_OPEN"
    WINDOW_RESIZE = "WINDOW_RESIZE"
    # A webcam still was uploaded; s3_key points at it. Analysed offline.
    FACE_FRAME = "FACE_FRAME"
    # Derived server-side, never reported by the client.
    FACE_ABSENT = "FACE_ABSENT"
    MULTIPLE_FACES = "MULTIPLE_FACES"
    SECOND_SPEAKER = "SECOND_SPEAKER"
    ANOMALOUS_SILENCE = "ANOMALOUS_SILENCE"


class ProctorSeverity(enum.StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


class ProctorVerdictKind(enum.StrEnum):
    CLEAN = "CLEAN"
    SUSPICIOUS = "SUSPICIOUS"
    FLAGGED = "FLAGGED"
    # Nothing was reported at all. Distinct from CLEAN: a candidate who
    # disabled JavaScript and one who behaved perfectly both produce zero
    # events, and conflating them would let the first hide behind the second.
    NO_DATA = "NO_DATA"


proctor_event_type_enum = Enum(
    ProctorEventType,
    name="proctor_event_type",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)
proctor_severity_enum = Enum(
    ProctorSeverity,
    name="proctor_severity",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)
proctor_verdict_kind_enum = Enum(
    ProctorVerdictKind,
    name="proctor_verdict_kind",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)


class ProctoringPolicy(Base, TenantMixin, TimestampMixin):
    """Per-job thresholds. One policy per job; absent means the org defaults."""

    __tablename__ = "proctoring_policies"
    __table_args__ = (UniqueConstraint("job_id", name="uq_proctoring_policies_job_id"),)

    id: Mapped[uuid.UUID] = uuid_pk()
    job_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    camera_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    frame_interval_secs: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("10")
    )
    # How many times a candidate may leave the tab before it escalates.
    blur_limit: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("3"))
    fullscreen_required: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    paste_blocked: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    # Off by default, and it should stay off. Ending a real person's interview
    # on a heuristic is a decision a human should make; the evidence is in the
    # report either way.
    auto_terminate: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )

    job: Mapped["Job"] = relationship()

    def __repr__(self) -> str:
        return f"<ProctoringPolicy job={self.job_id}>"


class ProctoringEvent(Base, TenantMixin, TimestampMixin):
    """One observation. Append-only, server-timestamped."""

    __tablename__ = "proctoring_events"
    __table_args__ = (
        Index("ix_proctoring_events_interview_id_at", "interview_id", "at"),
        Index("ix_proctoring_events_interview_id_type", "interview_id", "event_type"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[ProctorEventType] = mapped_column(
        proctor_event_type_enum, nullable=False
    )
    # Assigned by our rules from the policy. Never read off the wire: a
    # candidate reporting their own tab-switch as INFO would be self-grading.
    severity: Mapped[ProctorSeverity] = mapped_column(
        proctor_severity_enum, nullable=False, server_default=text("'INFO'")
    )
    # Server clock. A client-supplied time would let a candidate backdate an
    # event out of the interview window entirely.
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    # Milliseconds since session start, so an event lines up with the recording
    # and the transcript the same way a turn does.
    offset_ms: Mapped[int | None] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # Set for FACE_FRAME: where the still lives, pending offline analysis.
    s3_key: Mapped[str | None] = mapped_column(String(512))

    interview: Mapped["Interview"] = relationship()

    def __repr__(self) -> str:
        return f"<ProctoringEvent {self.event_type} {self.severity}>"


class ProctoringVerdict(Base, TenantMixin, TimestampMixin):
    """The aggregated judgement, recomputed rather than accumulated."""

    __tablename__ = "proctoring_verdicts"
    __table_args__ = (
        UniqueConstraint("interview_id", name="uq_proctoring_verdicts_interview_id"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    verdict: Mapped[ProctorVerdictKind] = mapped_column(
        proctor_verdict_kind_enum, nullable=False, server_default=text("'NO_DATA'")
    )
    # Human-readable, ordered, and shown verbatim in the recruiter report. A
    # verdict without its reasons is an accusation.
    reasons: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # Per-type counts, so a reviewer can see the shape without paging events.
    counts: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    frames_analysed: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    notes: Mapped[str | None] = mapped_column(Text)

    interview: Mapped["Interview"] = relationship()

    def __repr__(self) -> str:
        return f"<ProctoringVerdict {self.interview_id} {self.verdict}>"
