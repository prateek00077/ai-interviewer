"""recruiter and candidate reports

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-23

Two tables rather than one with an ``audience`` column, so the boundary between
what a recruiter may read and what a candidate may read is a Postgres policy
rather than a WHERE clause somebody has to remember. See models/report.py.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_rls, drop_policy, enable_rls, policy_for

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

REPORT_STATUS = ("PENDING", "RENDERING", "READY", "FAILED")

NEW_TABLES = ("recruiter_reports", "candidate_reports")


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


def _interview_fk(table: str) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        ["interview_id"], ["interviews.id"],
        name=f"fk_{table}_interview_id_interviews", ondelete="CASCADE",
    )


def upgrade() -> None:
    status = postgresql.ENUM(*REPORT_STATUS, name="report_status", create_type=False)
    status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "recruiter_reports",
        *_pk_org(),
        sa.Column("interview_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", status, server_default=sa.text("'PENDING'"), nullable=False),
        sa.Column("s3_key", sa.String(512), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_recruiter_reports"),
        _org_fk("recruiter_reports"),
        _interview_fk("recruiter_reports"),
        sa.UniqueConstraint("interview_id", name="uq_recruiter_reports_interview_id"),
    )
    op.create_index("ix_recruiter_reports_org_id", "recruiter_reports", ["org_id"])
    op.create_index("ix_recruiter_reports_interview_id", "recruiter_reports", ["interview_id"])
    op.create_index(
        "ix_recruiter_reports_org_id_status", "recruiter_reports", ["org_id", "status"]
    )

    op.create_table(
        "candidate_reports",
        *_pk_org(),
        sa.Column("interview_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", status, server_default=sa.text("'PENDING'"), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column(
            "strengths", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        sa.Column(
            "growth_areas",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("s3_key", sa.String(512), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_candidate_reports"),
        _org_fk("candidate_reports"),
        _interview_fk("candidate_reports"),
        sa.ForeignKeyConstraint(
            ["candidate_id"], ["candidates.id"],
            name="fk_candidate_reports_candidate_id_candidates", ondelete="CASCADE",
        ),
        sa.UniqueConstraint("interview_id", name="uq_candidate_reports_interview_id"),
    )
    op.create_index("ix_candidate_reports_org_id", "candidate_reports", ["org_id"])
    op.create_index("ix_candidate_reports_interview_id", "candidate_reports", ["interview_id"])
    op.create_index("ix_candidate_reports_candidate_id", "candidate_reports", ["candidate_id"])
    op.create_index(
        "ix_candidate_reports_org_id_status", "candidate_reports", ["org_id", "status"]
    )

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

    op.drop_table("candidate_reports")
    op.drop_table("recruiter_reports")

    postgresql.ENUM(name="report_status").drop(op.get_bind(), checkfirst=True)
