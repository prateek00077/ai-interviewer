-- One-time database and role bootstrap. Run as a superuser, before Alembic.
--
--   psql -v owner_pw="'...'" -v app_pw="'...'" -f scripts/bootstrap_db.sql ai_interviewer
--
-- This is NOT an Alembic migration: it needs CREATE ROLE, a privilege Alembic
-- should not hold, and the credentials differ per environment.
--
-- WHY TWO ROLES: Postgres silently bypasses RLS for superusers, for roles with
-- BYPASSRLS, and for the table owner. If the application connects as `postgres`
-- or as the owner, tests/integration/test_rls.py passes while proving nothing.
-- So migrations run as app_owner (owns the schema) and the app runs as app_user
-- (owns nothing, holds only DML grants). Combined with FORCE ROW LEVEL SECURITY,
-- even app_owner is subject to the policies -- a misconfigured DATABASE_URL then
-- degrades to "sees nothing" instead of "sees everything".
--
-- WHY A THIRD ROLE: because FORCE binds app_owner to the policies too, a
-- SECURITY DEFINER function owned by app_owner also sees nothing -- which breaks
-- the one lookup that legitimately has no org yet (login by email). app_auth
-- exists solely to own that function. It cannot log in, holds SELECT on exactly
-- two tables, and is named by exactly one policy. That is a far narrower hole
-- than either dropping FORCE or giving anything BYPASSRLS.

\set ON_ERROR_STOP on

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_owner') THEN
    CREATE ROLE app_owner LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_user') THEN
    CREATE ROLE app_user LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
  END IF;
  -- NOLOGIN: nothing ever connects as app_auth. It is a function owner, not an
  -- identity, so it has no password to leak.
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_auth') THEN
    CREATE ROLE app_auth NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
  END IF;
END
$$;

ALTER ROLE app_owner WITH PASSWORD :owner_pw;
ALTER ROLE app_user  WITH PASSWORD :app_pw;

-- Belt and braces: re-assert on an existing role that may predate this script.
ALTER ROLE app_owner NOSUPERUSER NOBYPASSRLS;
ALTER ROLE app_user  NOSUPERUSER NOBYPASSRLS;
ALTER ROLE app_auth  NOLOGIN NOSUPERUSER NOBYPASSRLS;

-- app_owner must be a member of app_auth to hand the function over to it.
GRANT app_auth TO app_owner;

ALTER DATABASE ai_interviewer OWNER TO app_owner;
ALTER SCHEMA public OWNER TO app_owner;

CREATE SCHEMA IF NOT EXISTS app AUTHORIZATION app_owner;

GRANT CONNECT ON DATABASE ai_interviewer TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT USAGE ON SCHEMA app    TO app_user;

-- app_user may read and write rows but never create or alter objects.
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
REVOKE CREATE ON SCHEMA public FROM app_user;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES    IN SCHEMA public TO app_user;
GRANT USAGE, SELECT                  ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- app_auth reads nothing on its own account; the table grants it needs are made
-- by the RLS migration, alongside the single policy that names it.
GRANT USAGE ON SCHEMA public TO app_auth;
-- CREATE is required to *own* an object in the schema, which is what
-- ALTER FUNCTION ... OWNER TO app_auth needs. Harmless: app_auth is NOLOGIN.
GRANT USAGE, CREATE ON SCHEMA app TO app_auth;

-- Without these, every future migration creates tables app_user cannot touch and
-- the app breaks with a permission error the moment the new table is queried.
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner IN SCHEMA public
  GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner IN SCHEMA public
  GRANT USAGE, SELECT ON SEQUENCES TO app_user;
ALTER DEFAULT PRIVILEGES FOR ROLE app_owner IN SCHEMA app
  GRANT EXECUTE ON FUNCTIONS TO app_user;
