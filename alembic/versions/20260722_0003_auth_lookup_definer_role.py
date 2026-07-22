"""move the login lookup to a dedicated definer role

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-22

FORCE ROW LEVEL SECURITY binds the table *owner* to the policies too, so
``app.lookup_user_for_auth`` -- owned by app_owner -- read zero rows and every
login returned "invalid credentials".

The fix keeps FORCE and narrows the bypass instead: the function is handed to
app_auth, a NOLOGIN role that holds SELECT on exactly two tables and is named by
exactly one policy. Requires scripts/bootstrap_db.sql to have been re-run, since
CREATE ROLE is not a privilege Alembic holds.
"""

from collections.abc import Sequence

from alembic import op
from app.db.rls import CREATE_AUTH_LOOKUP, DROP_AUTH_LOOKUP

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # A fresh database already gets these from 0002; re-running them is safe
    # (CREATE OR REPLACE / DROP IF EXISTS / idempotent GRANTs) and is what
    # brings a database stamped at 0002 into line.
    for statement in CREATE_AUTH_LOOKUP:
        op.execute(statement)


def downgrade() -> None:
    for statement in DROP_AUTH_LOOKUP:
        op.execute(statement)
