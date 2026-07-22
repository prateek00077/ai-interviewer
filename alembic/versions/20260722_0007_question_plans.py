"""question plans, planned questions and rubric criteria

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-22

There is no `rubrics` table on purpose: a plan has exactly one rubric, so such a
row would carry an id, a plan_id and nothing else. Criteria hang off the plan.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_rls, drop_policy, enable_rls, policy_for
from app.models.question_plan import WEIGHT_PRECISION, WEIGHT_SCALE

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

PLAN_STATUS = ("DRAFT", "APPROVED", "FROZEN")
PLAN_GENERATION_STATUS = ("PENDING", "GENERATING", "READY", "FAILED")

# Children first on the way down, so the drop order below is this reversed.
NEW_TABLES = ("question_plans", "planned_questions", "rubric_criteria")


def _timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def _pk_and_org(table: str) -> list:
    return [
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("org_id", postgresql.UUID(as_uuid=True), nullable=False),
    ]


def upgrade() -> None:
    plan_status = postgresql.ENUM(*PLAN_STATUS, name="plan_status", create_type=False)
    generation_status = postgresql.ENUM(
        *PLAN_GENERATION_STATUS, name="plan_generation_status", create_type=False
    )
    plan_status.create(op.get_bind(), checkfirst=True)
    generation_status.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "question_plans",
        *_pk_and_org("question_plans"),
        sa.Column("interview_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_description_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("resume_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", plan_status, server_default=sa.text("'DRAFT'"), nullable=False),
        sa.Column(
            "generation_status",
            generation_status,
            server_default=sa.text("'PENDING'"),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("generated_by", sa.String(120), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_question_plans"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_question_plans_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["interview_id"],
            ["interviews.id"],
            name="fk_question_plans_interview_id_interviews",
            ondelete="CASCADE",
        ),
        # SET NULL: deleting a job must not delete the plan an interview was
        # conducted against.
        sa.ForeignKeyConstraint(
            ["job_description_id"],
            ["job_descriptions.id"],
            name="fk_question_plans_job_description_id_job_descriptions",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["resume_id"],
            ["resumes.id"],
            name="fk_question_plans_resume_id_resumes",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("interview_id", name="uq_question_plans_interview_id"),
    )
    op.create_index("ix_question_plans_org_id", "question_plans", ["org_id"])
    op.create_index("ix_question_plans_interview_id", "question_plans", ["interview_id"])
    op.create_index("ix_question_plans_org_id_status", "question_plans", ["org_id", "status"])

    op.create_table(
        "planned_questions",
        *_pk_and_org("planned_questions"),
        sa.Column("plan_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("competency", sa.String(120), nullable=True),
        sa.Column(
            "follow_up_hints",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("time_budget_secs", sa.Integer(), server_default=sa.text("180"), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_planned_questions"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_planned_questions_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["question_plans.id"],
            name="fk_planned_questions_plan_id_question_plans",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("plan_id", "ordinal", name="uq_planned_questions_plan_id_ordinal"),
        sa.CheckConstraint(
            "length(btrim(body)) > 0", name="ck_planned_questions_body_not_blank"
        ),
    )
    op.create_index("ix_planned_questions_org_id", "planned_questions", ["org_id"])
    op.create_index("ix_planned_questions_plan_id", "planned_questions", ["plan_id"])

    op.create_table(
        "rubric_criteria",
        *_pk_and_org("rubric_criteria"),
        sa.Column("plan_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        # Numeric, not float: the scorer multiplies by these and a rubric that
        # sums to 0.9999999999 is a support ticket.
        sa.Column("weight", sa.Numeric(WEIGHT_PRECISION, WEIGHT_SCALE), nullable=False),
        sa.Column(
            "descriptors",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_rubric_criteria"),
        sa.ForeignKeyConstraint(
            ["org_id"],
            ["organizations.id"],
            name="fk_rubric_criteria_org_id_organizations",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["plan_id"],
            ["question_plans.id"],
            name="fk_rubric_criteria_plan_id_question_plans",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("plan_id", "ordinal", name="uq_rubric_criteria_plan_id_ordinal"),
        sa.UniqueConstraint("plan_id", "name", name="uq_rubric_criteria_plan_id_name"),
        sa.CheckConstraint("weight > 0 AND weight <= 1", name="ck_rubric_criteria_weight_in_range"),
    )
    op.create_index("ix_rubric_criteria_org_id", "rubric_criteria", ["org_id"])
    op.create_index("ix_rubric_criteria_plan_id", "rubric_criteria", ["plan_id"])

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

    op.drop_table("rubric_criteria")
    op.drop_table("planned_questions")
    op.drop_table("question_plans")

    postgresql.ENUM(name="plan_generation_status").drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(name="plan_status").drop(op.get_bind(), checkfirst=True)
