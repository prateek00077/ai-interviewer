"""record when a plan was approved, separately from its status

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-23

``freeze()`` overwrites ``question_plans.status`` with FROZEN when the interview
starts, so after any real interview there was no record left of whether a
recruiter had reviewed the questions. APPROVED was written by one line and read
by nothing.

Backfilled for plans still sitting in APPROVED. Anything already FROZEN keeps a
NULL, which is the honest answer: that information was never recorded and cannot
be reconstructed.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "question_plans",
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
    )
    # A plan sitting in APPROVED right now was approved at some point we did not
    # record; updated_at is the closest honest approximation, and it is better
    # than losing the fact entirely.
    op.execute(
        "UPDATE question_plans SET approved_at = updated_at WHERE status = 'APPROVED'"
    )


def downgrade() -> None:
    op.drop_column("question_plans", "approved_at")
