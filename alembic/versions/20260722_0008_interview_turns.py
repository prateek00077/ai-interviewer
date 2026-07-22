"""interview turns

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-22

The transcript. Deferred from 0001, which only needed enough of `interviews` to
anchor an invite and mint a token.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_rls, drop_policy, enable_rls, policy_for

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SPEAKER = ("CANDIDATE", "INTERVIEWER")


def upgrade() -> None:
    speaker = postgresql.ENUM(*SPEAKER, name="speaker", create_type=False)
    speaker.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "interview_turns",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("interview_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("speaker", speaker, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        # Milliseconds from session start, not wall clock: these tie a line to a
        # position in the recording, and a timestamp would drift against it.
        sa.Column(
            "started_offset_ms", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("ended_offset_ms", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("question_ordinal", sa.Integer(), nullable=True),
        sa.Column("is_final", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_interview_turns"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_interview_turns_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["interview_id"],
            ["interviews.id"],
            name="fk_interview_turns_interview_id_interviews",
            ondelete="CASCADE",
        ),
        # A duplicated turn event becomes a database error rather than a doubled
        # line in the transcript.
        sa.UniqueConstraint(
            "interview_id", "ordinal", name="uq_interview_turns_interview_id_ordinal"
        ),
        sa.CheckConstraint(
            "ended_offset_ms >= started_offset_ms", name="ck_interview_turns_offsets_ordered"
        ),
    )
    op.create_index("ix_interview_turns_org_id", "interview_turns", ["org_id"])
    op.create_index(
        "ix_interview_turns_interview_id_ordinal", "interview_turns", ["interview_id", "ordinal"]
    )

    for statement in enable_rls("interview_turns"):
        op.execute(statement)
    op.execute(drop_policy("interview_turns"))
    op.execute(policy_for("interview_turns"))


def downgrade() -> None:
    op.execute(drop_policy("interview_turns"))
    for statement in disable_rls("interview_turns"):
        op.execute(statement)
    op.drop_table("interview_turns")
    postgresql.ENUM(name="speaker").drop(op.get_bind(), checkfirst=True)
