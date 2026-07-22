-- Row-Level Security policies for every tenant table.
--
-- GENERATED FILE -- do not edit. app/db/rls.py is the source of truth, and the
-- Alembic migrations call it directly. This dump exists so the policies can be
-- read, diffed, and reviewed as plain SQL without running Python.
--
-- Regenerate with:
--   python -c "from app.db.rls import upgrade_statements; \
--     print(';\n'.join(s.strip() for s in upgrade_statements()) + ';')" \
--     > scripts/create_rls_policies.sql

CREATE SCHEMA IF NOT EXISTS app;

CREATE OR REPLACE FUNCTION app.current_org() RETURNS uuid
LANGUAGE sql STABLE AS $$
  SELECT nullif(current_setting('app.current_org', true), '')::uuid
$$;

CREATE OR REPLACE FUNCTION app.actor_kind() RETURNS text
LANGUAGE sql STABLE AS $$
  SELECT coalesce(nullif(current_setting('app.actor_kind', true), ''), 'none')
$$;

CREATE OR REPLACE FUNCTION app.actor_id() RETURNS uuid
LANGUAGE sql STABLE AS $$
  SELECT nullif(current_setting('app.actor_id', true), '')::uuid
$$;

GRANT SELECT ON users, organizations TO app_auth;

DROP POLICY IF EXISTS users_auth_lookup ON users;

CREATE POLICY users_auth_lookup ON users
  FOR SELECT TO app_auth
  USING (true);

DROP POLICY IF EXISTS organizations_auth_lookup ON organizations;

CREATE POLICY organizations_auth_lookup ON organizations
  FOR SELECT TO app_auth
  USING (true);

CREATE OR REPLACE FUNCTION app.lookup_user_for_auth(p_email citext)
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
$$;

ALTER FUNCTION app.lookup_user_for_auth(citext) OWNER TO app_auth;

REVOKE ALL ON FUNCTION app.lookup_user_for_auth(citext) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION app.lookup_user_for_auth(citext) TO app_user;

ALTER TABLE organizations ENABLE ROW LEVEL SECURITY;

ALTER TABLE organizations FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS organizations_tenant_isolation ON organizations;

CREATE POLICY organizations_tenant_isolation ON organizations
  FOR ALL TO app_user
  USING (id = app.current_org())
  WITH CHECK (id = app.current_org());

ALTER TABLE users ENABLE ROW LEVEL SECURITY;

ALTER TABLE users FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS users_tenant_isolation ON users;

CREATE POLICY users_tenant_isolation ON users
  FOR ALL TO app_user
  USING (org_id = app.current_org() AND app.actor_kind() = 'user')
  WITH CHECK (org_id = app.current_org() AND app.actor_kind() = 'user');

ALTER TABLE candidates ENABLE ROW LEVEL SECURITY;

ALTER TABLE candidates FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS candidates_tenant_isolation ON candidates;

CREATE POLICY candidates_tenant_isolation ON candidates
  FOR ALL TO app_user
  USING (
    org_id = app.current_org()
    AND (
      app.actor_kind() = 'user'
      OR (app.actor_kind() = 'candidate' AND id = app.actor_id())
    )
  )
  WITH CHECK (
    org_id = app.current_org() AND app.actor_kind() = 'user'
  );

ALTER TABLE interviews ENABLE ROW LEVEL SECURITY;

ALTER TABLE interviews FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS interviews_tenant_isolation ON interviews;

CREATE POLICY interviews_tenant_isolation ON interviews
  FOR ALL TO app_user
  USING (
    org_id = app.current_org()
    AND (
      app.actor_kind() = 'user'
      OR (app.actor_kind() = 'candidate' AND candidate_id = app.actor_id())
    )
  )
  WITH CHECK (
    org_id = app.current_org() AND app.actor_kind() = 'user'
  );

ALTER TABLE invites ENABLE ROW LEVEL SECURITY;

ALTER TABLE invites FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS invites_tenant_isolation ON invites;

CREATE POLICY invites_tenant_isolation ON invites
  FOR ALL TO app_user
  USING (org_id = app.current_org() AND app.actor_kind() = 'user')
  WITH CHECK (org_id = app.current_org() AND app.actor_kind() = 'user');
