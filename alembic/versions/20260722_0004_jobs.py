"""jobs and versioned job descriptions

Revision ID: 0004
Revises: 0003
Create Date: 2026-07-22

The RLS half is not hand-written: the two new tables are registered in
app.db.base, so app.db.rls generates the same policy shape every other tenant
table already has. Adding the SQL by hand here is exactly how a tenant leak
would get introduced.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_rls, drop_policy, enable_rls, policy_for

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JOB_STATUS = ("DRAFT", "OPEN", "CLOSED")
EMPLOYMENT_TYPE = ("FULL_TIME", "PART_TIME", "CONTRACT", "INTERNSHIP")

NEW_TABLES = ("jobs", "job_descriptions")


def upgrade() -> None:
    job_status = postgresql.ENUM(*JOB_STATUS, name="job_status", create_type=False)
    employment_type = postgresql.ENUM(
        *EMPLOYMENT_TYPE, name="employment_type", create_type=False
    )
    job_status.create(op.get_bind(), checkfirst=True)
    employment_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("department", sa.String(120), nullable=True),
        sa.Column("location", sa.String(200), nullable=True),
        sa.Column(
            "employment_type",
            employment_type,
            server_default=sa.text("'FULL_TIME'"),
            nullable=False,
        ),
        sa.Column("status", job_status, server_default=sa.text("'DRAFT'"), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_jobs_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_jobs_created_by_user_id_users",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_jobs_org_id", "jobs", ["org_id"])
    op.create_index("ix_jobs_org_id_status", "jobs", ["org_id", "status"])

    op.create_table(
        "job_descriptions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "requirements",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id", name="pk_job_descriptions"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_job_descriptions_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name="fk_job_descriptions_job_id_jobs",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"],
            ["users.id"],
            name="fk_job_descriptions_created_by_user_id_users",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("job_id", "version", name="uq_job_descriptions_job_id_version"),
    )
    op.create_index("ix_job_descriptions_org_id", "job_descriptions", ["org_id"])
    op.create_index("ix_job_descriptions_job_id", "job_descriptions", ["job_id"])
    # Partial unique: "at most one active per job" decided by Postgres, so two
    # concurrent activations cannot both observe "none active" and both win.
    op.create_index(
        "ix_job_descriptions_one_active_per_job",
        "job_descriptions",
        ["job_id"],
        unique=True,
        postgresql_where=sa.text("is_active"),
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

    op.drop_table("job_descriptions")
    op.drop_table("jobs")

    postgresql.ENUM(name="employment_type").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="job_status").drop(op.get_bind(), checkfirst=True)
