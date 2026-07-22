"""admit the system actor to staff-level policies

Revision ID: 0005
Revises: 0004
Create Date: 2026-07-22

Background work -- transcription, scoring, report rendering, invite expiry --
runs in Celery with no logged-in user to attribute it to, so those sessions open
with ``actor_kind = 'system'``. Every policy written so far tests
``actor_kind() = 'user'``, which means a worker currently reads zero rows from
users, invites, jobs and job_descriptions, and cannot write the candidate-scoped
tables at all.

The alternative -- workers claiming ``actor_kind='user'`` -- would make
``app.actor_kind()`` lie about who touched a row and remove the audit value of
the GUC entirely. Widening the predicate to ``ANY (ARRAY['user','system'])`` is
the honest version.

The tenant boundary is unchanged: a system session is still opened with exactly
one org_id, so it sees one org. What widens is the actor dimension only, and
'candidate' is still excluded from every staff branch.

Policies are regenerated from app.db.rls rather than patched, so this migration
stays correct as tables are added.
"""

from collections.abc import Sequence

from alembic import op
from app.db.rls import drop_policy, policy_for

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tables that exist at THIS revision -- everything through 0004, and nothing
# from 0006. Naming them explicitly is what keeps a from-scratch upgrade working
# as later revisions extend the registry.
TABLES_AT_THIS_REVISION = [
    "organizations",
    "users",
    "candidates",
    "interviews",
    "invites",
    "jobs",
    "job_descriptions",
]


def upgrade() -> None:
    for table in TABLES_AT_THIS_REVISION:
        op.execute(drop_policy(table))
        op.execute(policy_for(table))


def downgrade() -> None:
    # Not reversible in a useful sense: policy_for() now emits the widened
    # predicate, so regenerating here would reproduce the upgrade. Rolling this
    # back means checking out the previous revision of app/db/rls.py, which is a
    # code change rather than a schema one. Recreating the policies keeps the
    # database in a valid state either way.
    for table in TABLES_AT_THIS_REVISION:
        op.execute(drop_policy(table))
        op.execute(policy_for(table))
