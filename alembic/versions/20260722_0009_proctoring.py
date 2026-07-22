"""proctoring policies, events and verdicts

Revision ID: 0009
Revises: 0008
Create Date: 2026-07-22

Events are server-timestamped and append-only; severity is a column we assign
from the policy rather than a value read off the wire. Both matter because the
browser reporting these is controlled by the person being assessed.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from app.db.rls import disable_rls, drop_policy, enable_rls, policy_for

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EVENT_TYPE = (
    "TAB_BLUR", "TAB_FOCUS", "FULLSCREEN_EXIT", "PASTE", "COPY", "DEVTOOLS_OPEN",
    "WINDOW_RESIZE", "FACE_FRAME", "FACE_ABSENT", "MULTIPLE_FACES",
    "SECOND_SPEAKER", "ANOMALOUS_SILENCE",
)
SEVERITY = ("INFO", "WARN", "CRITICAL")
VERDICT = ("CLEAN", "SUSPICIOUS", "FLAGGED", "NO_DATA")

NEW_TABLES = ("proctoring_policies", "proctoring_events", "proctoring_verdicts")


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
    event_type = postgresql.ENUM(*EVENT_TYPE, name="proctor_event_type", create_type=False)
    severity = postgresql.ENUM(*SEVERITY, name="proctor_severity", create_type=False)
    verdict = postgresql.ENUM(*VERDICT, name="proctor_verdict_kind", create_type=False)
    for enum_type in (event_type, severity, verdict):
        enum_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "proctoring_policies",
        *_pk_org(),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("camera_enabled", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column(
            "frame_interval_secs", sa.Integer(), server_default=sa.text("10"), nullable=False
        ),
        sa.Column("blur_limit", sa.Integer(), server_default=sa.text("3"), nullable=False),
        sa.Column(
            "fullscreen_required", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("paste_blocked", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        sa.Column("auto_terminate", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_proctoring_policies"),
        _org_fk("proctoring_policies"),
        sa.ForeignKeyConstraint(
            ["job_id"], ["jobs.id"],
            name="fk_proctoring_policies_job_id_jobs", ondelete="CASCADE",
        ),
        sa.UniqueConstraint("job_id", name="uq_proctoring_policies_job_id"),
    )
    op.create_index("ix_proctoring_policies_org_id", "proctoring_policies", ["org_id"])
    op.create_index("ix_proctoring_policies_job_id", "proctoring_policies", ["job_id"])

    op.create_table(
        "proctoring_events",
        *_pk_org(),
        sa.Column("interview_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("event_type", event_type, nullable=False),
        sa.Column("severity", severity, server_default=sa.text("'INFO'"), nullable=False),
        # Server clock: a client-supplied time would let a candidate backdate an
        # event out of the interview window entirely.
        sa.Column("at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("offset_ms", sa.Integer(), nullable=True),
        sa.Column(
            "payload", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("s3_key", sa.String(512), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_proctoring_events"),
        _org_fk("proctoring_events"),
        sa.ForeignKeyConstraint(
            ["interview_id"], ["interviews.id"],
            name="fk_proctoring_events_interview_id_interviews", ondelete="CASCADE",
        ),
    )
    op.create_index("ix_proctoring_events_org_id", "proctoring_events", ["org_id"])
    op.create_index(
        "ix_proctoring_events_interview_id_at", "proctoring_events", ["interview_id", "at"]
    )
    op.create_index(
        "ix_proctoring_events_interview_id_type",
        "proctoring_events", ["interview_id", "event_type"],
    )

    op.create_table(
        "proctoring_verdicts",
        *_pk_org(),
        sa.Column("interview_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("verdict", verdict, server_default=sa.text("'NO_DATA'"), nullable=False),
        sa.Column(
            "reasons", postgresql.JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        sa.Column(
            "counts", postgresql.JSONB(), server_default=sa.text("'{}'::jsonb"), nullable=False
        ),
        sa.Column("frames_analysed", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        *_timestamps(),
        sa.PrimaryKeyConstraint("id", name="pk_proctoring_verdicts"),
        _org_fk("proctoring_verdicts"),
        sa.ForeignKeyConstraint(
            ["interview_id"], ["interviews.id"],
            name="fk_proctoring_verdicts_interview_id_interviews", ondelete="CASCADE",
        ),
        sa.UniqueConstraint("interview_id", name="uq_proctoring_verdicts_interview_id"),
    )
    op.create_index("ix_proctoring_verdicts_org_id", "proctoring_verdicts", ["org_id"])
    op.create_index("ix_proctoring_verdicts_interview_id", "proctoring_verdicts", ["interview_id"])

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

    op.drop_table("proctoring_verdicts")
    op.drop_table("proctoring_events")
    op.drop_table("proctoring_policies")

    for name in ("proctor_verdict_kind", "proctor_severity", "proctor_event_type"):
        postgresql.ENUM(name=name).drop(op.get_bind(), checkfirst=True)
