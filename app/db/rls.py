"""Row-Level Security policy helpers and migration templates.

Policies are *generated*, never hand-written per table. The migration loops the
registries in ``db/base.py`` and calls these, so every tenant table ends up with
an identically shaped policy and adding a table is a one-line diff.

Two Postgres details this module encodes:

- ``current_setting(name, true)`` -- the second argument is ``missing_ok``.
  Without it, an unscoped session raises ``undefined_object`` on every query
  instead of quietly returning zero rows.
- ``nullif(..., '')`` -- ``set_config`` to an empty string yields ``''``, not
  NULL, and ``''::uuid`` throws.
"""

from app.db.base import CANDIDATE_SCOPED, TENANT_TABLES, USER_ONLY_TABLES

APP_SCHEMA = "app"
APP_ROLE = "app_user"
# Owns the login lookup function and nothing else. See CREATE_AUTH_LOOKUP.
AUTH_ROLE = "app_auth"

# --- GUC accessor functions -------------------------------------------------

# NOTE: every constant below is a SINGLE statement. asyncpg sends statements
# through the extended query protocol, which rejects multi-command strings with
# "cannot insert multiple commands into a prepared statement", so batching them
# into one string breaks the migration.

CREATE_APP_SCHEMA = f"CREATE SCHEMA IF NOT EXISTS {APP_SCHEMA}"

CREATE_GUC_FUNCTIONS = [
    f"""
CREATE OR REPLACE FUNCTION {APP_SCHEMA}.current_org() RETURNS uuid
LANGUAGE sql STABLE AS $$
  SELECT nullif(current_setting('app.current_org', true), '')::uuid
$$
""",
    f"""
CREATE OR REPLACE FUNCTION {APP_SCHEMA}.actor_kind() RETURNS text
LANGUAGE sql STABLE AS $$
  SELECT coalesce(nullif(current_setting('app.actor_kind', true), ''), 'none')
$$
""",
    f"""
CREATE OR REPLACE FUNCTION {APP_SCHEMA}.actor_id() RETURNS uuid
LANGUAGE sql STABLE AS $$
  SELECT nullif(current_setting('app.actor_id', true), '')::uuid
$$
""",
]

DROP_GUC_FUNCTIONS = [
    f"DROP FUNCTION IF EXISTS {APP_SCHEMA}.current_org()",
    f"DROP FUNCTION IF EXISTS {APP_SCHEMA}.actor_kind()",
    f"DROP FUNCTION IF EXISTS {APP_SCHEMA}.actor_id()",
]

# --- The one deliberate RLS bypass ------------------------------------------

# Login has an email but no org yet: chicken and egg. Rather than punching a
# "current_org IS NULL" hole in the users policy -- which would be wide enough to
# drive the whole threat model through -- this returns exactly six auth columns
# for exactly one email.
#
# The function is owned by app_auth, NOT by app_owner. FORCE ROW LEVEL SECURITY
# binds the table owner to the policies too, so a definer function owned by
# app_owner would also read zero rows. app_auth is a NOLOGIN role that exists for
# this one function and is named by exactly one policy, below.
#
# SET search_path is mandatory on any SECURITY DEFINER function: omitting it lets
# a caller shadow `users` with a temp table and escalate.
CREATE_AUTH_LOOKUP = [
    f"GRANT SELECT ON users, organizations TO {AUTH_ROLE}",
    # The narrowest possible bypass: SELECT only, two tables only, one role only.
    # app_auth cannot connect, so this is reachable solely through the function.
    "DROP POLICY IF EXISTS users_auth_lookup ON users",
    f"""
CREATE POLICY users_auth_lookup ON users
  FOR SELECT TO {AUTH_ROLE}
  USING (true)
""",
    "DROP POLICY IF EXISTS organizations_auth_lookup ON organizations",
    f"""
CREATE POLICY organizations_auth_lookup ON organizations
  FOR SELECT TO {AUTH_ROLE}
  USING (true)
""",
    f"""
CREATE OR REPLACE FUNCTION {APP_SCHEMA}.lookup_user_for_auth(p_email citext)
RETURNS TABLE (
  id uuid, org_id uuid, hashed_password text, role text, is_active boolean, org_active boolean
)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
  SELECT u.id, u.org_id, u.hashed_password, u.role::text, u.is_active, o.is_active
  FROM users u JOIN organizations o ON o.id = u.org_id
  WHERE u.email = p_email
  LIMIT 1
$$
""",
    # Ownership transfer is what makes the definer app_auth rather than app_owner.
    f"ALTER FUNCTION {APP_SCHEMA}.lookup_user_for_auth(citext) OWNER TO {AUTH_ROLE}",
    f"REVOKE ALL ON FUNCTION {APP_SCHEMA}.lookup_user_for_auth(citext) FROM PUBLIC",
    f"GRANT EXECUTE ON FUNCTION {APP_SCHEMA}.lookup_user_for_auth(citext) TO {APP_ROLE}",
]

DROP_AUTH_LOOKUP = [
    f"DROP FUNCTION IF EXISTS {APP_SCHEMA}.lookup_user_for_auth(citext)",
    "DROP POLICY IF EXISTS users_auth_lookup ON users",
    "DROP POLICY IF EXISTS organizations_auth_lookup ON organizations",
    f"REVOKE ALL ON users, organizations FROM {AUTH_ROLE}",
]


# --- Policy generators ------------------------------------------------------


def enable_rls(table: str) -> list[str]:
    """Enable and FORCE RLS.

    FORCE is what subjects the table *owner* to policies too. Without it a
    misconfigured DATABASE_URL pointing at the owner role silently sees
    everything, and the isolation tests pass while proving nothing.
    """
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
    ]


def disable_rls(table: str) -> list[str]:
    return [
        f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY",
    ]


def _policy_name(table: str) -> str:
    return f"{table}_tenant_isolation"


def drop_policy(table: str) -> str:
    return f"DROP POLICY IF EXISTS {_policy_name(table)} ON {table}"


def org_root_policy() -> str:
    """organizations is the tenant root: it matches on `id`, not `org_id`."""
    return f"""
CREATE POLICY {_policy_name("organizations")} ON organizations
  FOR ALL TO {APP_ROLE}
  USING (id = {APP_SCHEMA}.current_org())
  WITH CHECK (id = {APP_SCHEMA}.current_org())
"""


def tenant_policy(table: str) -> str:
    """Plain org isolation.

    WITH CHECK is not optional: USING alone filters reads but still permits
    INSERTing a row carrying another org's org_id.
    """
    return f"""
CREATE POLICY {_policy_name(table)} ON {table}
  FOR ALL TO {APP_ROLE}
  USING (org_id = {APP_SCHEMA}.current_org())
  WITH CHECK (org_id = {APP_SCHEMA}.current_org())
"""


def user_only_policy(table: str) -> str:
    """Org isolation plus: candidate actors read nothing at all."""
    return f"""
CREATE POLICY {_policy_name(table)} ON {table}
  FOR ALL TO {APP_ROLE}
  USING (org_id = {APP_SCHEMA}.current_org() AND {APP_SCHEMA}.actor_kind() = 'user')
  WITH CHECK (org_id = {APP_SCHEMA}.current_org() AND {APP_SCHEMA}.actor_kind() = 'user')
"""


def candidate_scoped_policy(table: str, owner_col: str) -> str:
    """Org isolation, narrowed to own rows for candidate actors.

    A candidate does get an org context -- RLS needs one -- but actor_kind
    confines them to rows they own within it. WITH CHECK omits the candidate
    branch entirely: candidates never write to these tables.
    """
    return f"""
CREATE POLICY {_policy_name(table)} ON {table}
  FOR ALL TO {APP_ROLE}
  USING (
    org_id = {APP_SCHEMA}.current_org()
    AND (
      {APP_SCHEMA}.actor_kind() = 'user'
      OR ({APP_SCHEMA}.actor_kind() = 'candidate' AND {owner_col} = {APP_SCHEMA}.actor_id())
    )
  )
  WITH CHECK (
    org_id = {APP_SCHEMA}.current_org() AND {APP_SCHEMA}.actor_kind() = 'user'
  )
"""


def policy_for(table: str) -> str:
    """Dispatch a table to its policy shape. One place, so nothing is ad hoc."""
    if table == "organizations":
        return org_root_policy()
    if table in USER_ONLY_TABLES:
        return user_only_policy(table)
    if table in CANDIDATE_SCOPED:
        return candidate_scoped_policy(table, CANDIDATE_SCOPED[table])
    return tenant_policy(table)


def all_tables() -> list[str]:
    return ["organizations", *TENANT_TABLES]


def upgrade_statements() -> list[str]:
    """Every statement the RLS migration runs, in order."""
    stmts = [CREATE_APP_SCHEMA, *CREATE_GUC_FUNCTIONS, *CREATE_AUTH_LOOKUP]
    for table in all_tables():
        stmts.extend(enable_rls(table))
        stmts.append(drop_policy(table))
        stmts.append(policy_for(table))
    return stmts


def downgrade_statements() -> list[str]:
    stmts: list[str] = []
    for table in all_tables():
        stmts.append(drop_policy(table))
        stmts.extend(disable_rls(table))
    stmts.extend(DROP_AUTH_LOOKUP)
    stmts.extend(DROP_GUC_FUNCTIONS)
    return stmts
