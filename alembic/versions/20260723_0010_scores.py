"""scores, criterion scores, and the interview recording key

Revision ID: 0010
Revises: 0009
Create Date: 2026-07-23

``interviews.recording_key`` lands here rather than with the voice slice because
this is the first revision that has a consumer for it: the offline transcript
pass and the confidence signals both start from the recording, and until now the
key was announced on the event bus and thrown away.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_rls, drop_policy, enable_rls, policy_for

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SCORING_STATUS = ("PENDING", "SCORING", "READY", "FAILED")
RECOMMENDATION = (
    "STRONG_HIRE",
    "HIRE",
    "BORDERLINE",
    "NO_HIRE",
    "INSUFFICIENT_EVIDENCE",
)

NEW_TABLES = ("scores", "criterion_scores")


def _timestamps() -> list:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def _pk_org() -> list:
    return [
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
    ]


def _org_fk(table: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["org_id"], ["organizations.id"],
        name=f"fk_{table}_org_id_organizations", ondelete="CASCADE",
    )


def upgrade() -> None:
    op.add_column("interviews", sa.Column("recording_key", sa.Text(), nullable=True))

    status = postgresql.ENUM(*SCORING_STATUS, name="scoring_status", create_type=False)
    recommendation = postgresql.ENUM(*RECOMMENDATION, name="recommendation", create_type=False)
    for enum_type in (status, recommendation):
        enum_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "scores",
        *_pk_org(),
        sa.Column("interview_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("plan_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", status, server_default=sa.text("'PENDING'"), nullable=False),
        sa.Column("overall", sa.Numeric(4, 2), nullable=True),
        sa.Column("recommendation", recommendation, nullable=True),
        sa.Column(
            "confidence_signals",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("scored_by", sa.String(120), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_scores"),
        _org_fk("scores"),
        sa.ForeignKeyConstraint(
            ["interview_id"], ["interviews.id"],
            name="fk_scores_interview_id_interviews", ondelete="CASCADE",
        ),
        # SET NULL, not CASCADE: deleting a job must not delete the assessment
        # of the person who was interviewed for it.
        sa.ForeignKeyConstraint(
            ["plan_id"], ["question_plans.id"],
            name="fk_scores_plan_id_question_plans", ondelete="SET NULL",
        ),
        sa.UniqueConstraint("interview_id", name="uq_scores_interview_id"),
        sa.CheckConstraint(
            "overall IS NULL OR (overall >= 1 AND overall <= 5)",
            name="overall_in_band",
        ),
    )
    op.create_index("ix_scores_org_id", "scores", ["org_id"])
    op.create_index("ix_scores_interview_id", "scores", ["interview_id"])
    op.create_index("ix_scores_org_id_status", "scores", ["org_id", "status"])

    op.create_table(
        "criterion_scores",
        *_pk_org(),
        sa.Column("score_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("criterion_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("weight", sa.Numeric(5, 4), nullable=False),
        sa.Column("score", sa.Numeric(4, 2), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column(
            "evidence", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_criterion_scores"),
        _org_fk("criterion_scores"),
        sa.ForeignKeyConstraint(
            ["score_id"], ["scores.id"],
            name="fk_criterion_scores_score_id_scores", ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["criterion_id"], ["rubric_criteria.id"],
            name="fk_criterion_scores_criterion_id_rubric_criteria", ondelete="SET NULL",
        ),
        sa.UniqueConstraint("score_id", "ordinal", name="uq_criterion_scores_score_id_ordinal"),
        sa.CheckConstraint(
            "score IS NULL OR (score >= 1 AND score <= 5)",
            name="score_in_band",
        ),
        sa.CheckConstraint(
            "weight > 0 AND weight <= 1", name="weight_in_range"
        ),
    )
    op.create_index("ix_criterion_scores_org_id", "criterion_scores", ["org_id"])
    op.create_index("ix_criterion_scores_score_id", "criterion_scores", ["score_id"])

    for table in NEW_TABLES:
        for statement in enable_rls(table):
            op.execute(statement)
        op.execute(drop_policy(table))
        op.execute(policy_for(table))


def downgrade() -> None:
    for table in NEW_TABLES:
        op.execute(drop_policy(table))
        for statement in disable_rls(table):
            op.execute(statement)

    op.drop_table("criterion_scores")
    op.drop_table("scores")

    for name in ("recommendation", "scoring_status"):
        postgresql.ENUM(name=name).drop(op.get_bind(), checkfirst=True)

    op.drop_column("interviews", "recording_key")
