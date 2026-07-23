"""Score and CriterionScore.

WHY THE CRITERION NAME AND WEIGHT ARE COPIED ONTO EVERY ROW rather than read
through the foreign key: a score is a record of a judgement made at a moment,
and it has to stay readable after the thing it judged is gone. The rubric is
FROZEN when the interview starts, so it cannot drift underneath a score -- but a
deleted job cascades away its plan, and with it the criteria. A score row that
answered "how did they do on System Design, weighted 0.35" would then answer
nothing at all. The foreign key is kept alongside the snapshot, SET NULL, purely
as provenance.

WHY CONFIDENCE SIGNALS ARE NOT PART OF THE SCORE: pitch variance, pause length
and filler rate are measurable, and they correlate with nervousness far more
reliably than with competence. Folding them into a number would mark an anxious
candidate down for being anxious, and would do it invisibly. They are stored as
signals on the side, shown to the recruiter as observations, and multiplied by
nothing.
"""

import enum
import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
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

# The band the rubric descriptors are written against ("1", "3", "5"), so the
# scorer, the aggregator and the prompt all agree on what a 3 means.
MIN_BAND = Decimal("1")
MAX_BAND = Decimal("5")

SCORE_PRECISION = 4
SCORE_SCALE = 2


class ScoringStatus(enum.StrEnum):
    """The state of the scoring job, not of the candidate.

    Separate from the recommendation for the same reason PlanGenerationStatus is
    separate from PlanStatus: "the model has not run yet" and "the model ran and
    found little" are different facts, and one field would make a crashed worker
    indistinguishable from a weak interview.
    """

    PENDING = "PENDING"
    SCORING = "SCORING"
    READY = "READY"
    FAILED = "FAILED"


class Recommendation(enum.StrEnum):
    STRONG_HIRE = "STRONG_HIRE"
    HIRE = "HIRE"
    BORDERLINE = "BORDERLINE"
    NO_HIRE = "NO_HIRE"
    # Distinct from NO_HIRE on purpose, exactly as proctoring's NO_DATA is
    # distinct from CLEAN. A candidate whose audio failed, and who therefore has
    # no transcript, has not been assessed -- and must never be filed under the
    # same heading as one who was assessed and did poorly.
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


scoring_status_enum = Enum(
    ScoringStatus,
    name="scoring_status",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,  # the migration owns CREATE TYPE
)
recommendation_enum = Enum(
    Recommendation,
    name="recommendation",
    values_callable=lambda e: [m.value for m in e],
    create_type=False,
)


class Score(Base, TenantMixin, TimestampMixin):
    """One interview's assessment: the weighted overall and how it was reached."""

    __tablename__ = "scores"
    __table_args__ = (
        # One score per interview. Re-running the scorer replaces its contents
        # rather than adding a second row, so "the score" is never ambiguous and
        # a duplicated Celery delivery cannot produce two answers.
        UniqueConstraint("interview_id", name="uq_scores_interview_id"),
        CheckConstraint(
            f"overall IS NULL OR (overall >= {MIN_BAND} AND overall <= {MAX_BAND})",
            name="overall_in_band",
        ),
        Index("ix_scores_org_id_status", "org_id", "status"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    interview_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("interviews.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Provenance only; the numbers that matter are snapshotted per criterion.
    plan_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("question_plans.id", ondelete="SET NULL")
    )

    status: Mapped[ScoringStatus] = mapped_column(
        scoring_status_enum, nullable=False, server_default=text("'PENDING'")
    )
    # Nullable until the job finishes: a PENDING row carrying overall=0 would
    # read as an assessment of zero.
    overall: Mapped[Decimal | None] = mapped_column(Numeric(SCORE_PRECISION, SCORE_SCALE))
    recommendation: Mapped[Recommendation | None] = mapped_column(recommendation_enum)
    # Pitch variance, pause distribution, filler rate. Observations, not inputs.
    confidence_signals: Mapped[dict] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    scored_by: Mapped[str | None] = mapped_column(String(120))
    error: Mapped[str | None] = mapped_column(Text)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    interview: Mapped["Interview"] = relationship()
    criteria: Mapped[list["CriterionScore"]] = relationship(
        back_populates="score_row",
        cascade="all, delete-orphan",
        order_by="CriterionScore.ordinal",
    )

    def __repr__(self) -> str:
        return f"<Score {self.interview_id} {self.status} overall={self.overall}>"


class CriterionScore(Base, TenantMixin, TimestampMixin):
    """One rubric dimension, scored, with the transcript lines that justify it.

    ``evidence`` is a list of ``{quote, turn_ordinal, offset_ms}``. It is what
    makes the score reviewable rather than merely produced: a recruiter can jump
    to the moment in the recording and disagree. A criterion the model could
    find no evidence for is recorded as ungraded and contributes nothing -- see
    ``aggregator`` for what happens to its weight.
    """

    __tablename__ = "criterion_scores"
    __table_args__ = (
        UniqueConstraint("score_id", "ordinal", name="uq_criterion_scores_score_id_ordinal"),
        CheckConstraint(
            f"score IS NULL OR (score >= {MIN_BAND} AND score <= {MAX_BAND})",
            name="score_in_band",
        ),
        CheckConstraint("weight > 0 AND weight <= 1", name="weight_in_range"),
    )

    id: Mapped[uuid.UUID] = uuid_pk()
    score_id: Mapped[uuid.UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey("scores.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    criterion_id: Mapped[uuid.UUID | None] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("rubric_criteria.id", ondelete="SET NULL")
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    # Snapshots. See the module docstring.
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    weight: Mapped[Decimal] = mapped_column(Numeric(5, 4), nullable=False)

    # NULL means "no evidence found", which is materially different from a 1.
    score: Mapped[Decimal | None] = mapped_column(Numeric(SCORE_PRECISION, SCORE_SCALE))
    rationale: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )

    # Named ``score_row`` because ``score`` is already the numeric column above.
    score_row: Mapped["Score"] = relationship(back_populates="criteria")

    @property
    def is_graded(self) -> bool:
        return self.score is not None

    def __repr__(self) -> str:
        return f"<CriterionScore {self.name}={self.score}>"
