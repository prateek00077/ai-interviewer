"""Proves org A cannot read org B's rows even without a WHERE clause.

The failure mode of every RLS suite is passing vacuously: if the connecting role
is a superuser, has BYPASSRLS, or owns the tables, Postgres skips policies
silently and every assertion below holds for the wrong reason. So the file opens
by asserting the environment is capable of failing at all.
"""

import uuid

import pytest
from sqlalchemy import delete, select, text, update
from sqlalchemy.exc import ProgrammingError

from app.db.base import CANDIDATE_SCOPED, TENANT_TABLES
from app.db.rls import all_tables
from app.models.interview import Interview
from app.models.job import Job, JobDescription
from app.models.org import Organization
from app.models.resume import Resume, ResumeChunk
from app.models.user import Candidate, User

pytestmark = pytest.mark.integration


# --- Preconditions ----------------------------------------------------------


async def test_app_role_cannot_bypass_rls(tenant_session, org_a):
    """If this fails, every other test in this file is meaningless."""
    async with tenant_session(org_a.org_id) as s:
        current_user = (await s.execute(text("SELECT current_user"))).scalar_one()

        role = (
            await s.execute(
                text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
            )
        ).one()
        assert role.rolsuper is False, f"{current_user} is a superuser; RLS is skipped"
        assert role.rolbypassrls is False, f"{current_user} has BYPASSRLS; RLS is skipped"

        for table in all_tables():
            row = (
                await s.execute(
                    text(
                        "SELECT pg_get_userbyid(relowner) AS owner, relrowsecurity, "
                        "relforcerowsecurity FROM pg_class WHERE relname = :t AND relkind = 'r'"
                    ),
                    {"t": table},
                )
            ).one()
            assert row.owner != current_user, f"{current_user} owns {table}; RLS is skipped"
            assert row.relrowsecurity, f"{table} has RLS disabled"
            # FORCE is what subjects the owner to policies too.
            assert row.relforcerowsecurity, f"{table} is missing FORCE ROW LEVEL SECURITY"


async def test_every_org_scoped_table_has_a_policy(tenant_session, org_a):
    """Stops a later slice adding a table and quietly shipping a tenant leak."""
    async with tenant_session(org_a.org_id) as s:
        rows = (
            await s.execute(
                text(
                    "SELECT table_name FROM information_schema.columns "
                    "WHERE column_name = 'org_id' AND table_schema = 'public'"
                )
            )
        ).scalars().all()
    assert set(rows) == set(TENANT_TABLES), (
        "a table carries org_id but is absent from TENANT_TABLES, so it has no RLS policy"
    )


# --- Reads ------------------------------------------------------------------


async def test_bare_select_returns_only_own_org(tenant_session, org_a, org_b):
    """No WHERE clause at all -- the isolation must come from the policy."""
    async with tenant_session(org_a.org_id) as s:
        rows = (await s.execute(select(Candidate))).scalars().all()

    assert len(rows) == 1
    assert rows[0].id == org_a.candidate_id
    assert rows[0].org_id == org_a.org_id


async def test_unscoped_session_sees_nothing(unscoped_session, org_a, org_b):
    async with unscoped_session() as s:
        for model in (Organization, User, Candidate, Interview):
            rows = (await s.execute(select(model))).scalars().all()
            assert rows == [], f"{model.__name__} leaked to a session with no org context"


async def test_cannot_read_other_org_by_explicit_id(tenant_session, org_a, org_b):
    """Even naming B's primary key from inside A returns nothing."""
    async with tenant_session(org_a.org_id) as s:
        assert await s.get(Candidate, org_b.candidate_id) is None
        assert await s.get(Organization, org_b.org_id) is None


# --- Writes -----------------------------------------------------------------


async def test_blind_update_touches_only_own_org(tenant_session, org_a, org_b):
    async with tenant_session(org_a.org_id) as s:
        result = await s.execute(update(Candidate).values(full_name="Rewritten"))
        assert result.rowcount == 1

    async with tenant_session(org_b.org_id) as s:
        b_candidate = (await s.execute(select(Candidate))).scalars().one()
        assert b_candidate.full_name == "Candidate bravo", "org A's blind UPDATE reached org B"


async def test_blind_delete_touches_only_own_org(tenant_session, org_a, org_b):
    async with tenant_session(org_a.org_id) as s:
        result = await s.execute(delete(Interview))
        assert result.rowcount == 1

    async with tenant_session(org_b.org_id) as s:
        remaining = (await s.execute(select(Interview))).scalars().all()
        assert len(remaining) == 1
        assert remaining[0].id == org_b.interview_id


async def test_cannot_insert_row_into_another_org(tenant_session, org_a, org_b):
    """WITH CHECK: USING alone would filter reads but still permit this."""
    with pytest.raises(ProgrammingError, match="row-level security"):
        async with tenant_session(org_a.org_id) as s:
            s.add(
                Candidate(
                    id=uuid.uuid4(),
                    org_id=org_b.org_id,  # someone else's tenant
                    email="smuggled@evil.test",
                )
            )
            await s.flush()


# --- Connection reuse -------------------------------------------------------


async def test_org_does_not_leak_across_pooled_connections(
    tenant_session, unscoped_session, org_a, org_b
):
    """The test that catches ``SET`` where ``SET LOCAL`` was meant.

    The fixture pool holds exactly one connection, so these three blocks
    provably run on the same physical connection. If the GUC were set without
    LOCAL, block two would still see org A.
    """
    async with tenant_session(org_a.org_id) as s:
        assert [c.org_id for c in (await s.execute(select(Candidate))).scalars()] == [
            org_a.org_id
        ]

    async with tenant_session(org_b.org_id) as s:
        assert [c.org_id for c in (await s.execute(select(Candidate))).scalars()] == [
            org_b.org_id
        ]

    async with unscoped_session() as s:
        assert (await s.execute(select(Candidate))).scalars().all() == []


async def test_org_context_survives_mid_session_commit(tenant_session, org_a, org_b):
    """``SET LOCAL`` dies at COMMIT; the after_begin listener must re-apply it."""
    async with tenant_session(org_a.org_id) as s:
        assert len((await s.execute(select(Candidate))).scalars().all()) == 1
        await s.commit()  # ends the transaction the GUCs were scoped to

        # SQLAlchemy silently begins a new transaction here. Without the
        # listener this returns zero rows and the org context is gone.
        rows = (await s.execute(select(Candidate))).scalars().all()
        assert len(rows) == 1, "org context was lost after a mid-session commit"
        assert rows[0].org_id == org_a.org_id


# --- Candidate scoping ------------------------------------------------------


async def test_candidate_actor_sees_only_own_rows(tenant_session, org_a):
    """A second candidate in the SAME org must still be invisible."""
    other_candidate = uuid.uuid4()
    other_interview = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(Candidate(id=other_candidate, org_id=org_a.org_id, email="other@alpha.test"))
        await s.flush()
        s.add(Interview(id=other_interview, org_id=org_a.org_id, candidate_id=other_candidate))

    async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
        interviews = (await s.execute(select(Interview))).scalars().all()
        assert [i.id for i in interviews] == [org_a.interview_id]

        candidates = (await s.execute(select(Candidate))).scalars().all()
        assert [c.id for c in candidates] == [org_a.candidate_id]


async def test_candidate_actor_cannot_read_users_or_invites(tenant_session, org_a):
    """Recruiter identities and invite rows are off-limits to candidate tokens."""
    async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
        assert (await s.execute(select(User))).scalars().all() == []
        assert (await s.execute(text("SELECT * FROM invites"))).all() == []


# --- Jobs -------------------------------------------------------------------


async def test_jobs_are_isolated_by_org(tenant_session, org_a, org_b):
    """A bare SELECT over jobs and descriptions must see only the caller's org."""
    for org, title in ((org_a, "Alpha Role"), (org_b, "Bravo Role")):
        async with tenant_session(org.org_id, "user", org.user_id) as s:
            job = Job(org_id=org.org_id, title=title, created_by_user_id=org.user_id)
            s.add(job)
            await s.flush()
            s.add(
                JobDescription(
                    org_id=org.org_id,
                    job_id=job.id,
                    content=f"Description for {title}.",
                    is_active=True,
                )
            )

    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        assert [j.title for j in (await s.execute(select(Job))).scalars()] == ["Alpha Role"]
        descriptions = (await s.execute(select(JobDescription))).scalars().all()
        assert [d.content for d in descriptions] == ["Description for Alpha Role."]


async def test_candidate_actor_cannot_read_jobs(tenant_session, org_a):
    """A candidate must not be able to enumerate an org's open roles."""
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        job = Job(org_id=org_a.org_id, title="Confidential Role", created_by_user_id=org_a.user_id)
        s.add(job)
        await s.flush()
        s.add(JobDescription(org_id=org_a.org_id, job_id=job.id, content="Salary band included."))

    async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
        assert (await s.execute(select(Job))).scalars().all() == []
        assert (await s.execute(select(JobDescription))).scalars().all() == []


async def test_cannot_insert_a_job_into_another_org(tenant_session, org_a, org_b):
    with pytest.raises(ProgrammingError, match="row-level security"):
        async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
            s.add(Job(org_id=org_b.org_id, title="Smuggled Role"))
            await s.flush()


# --- Resumes: the one candidate-writable table ------------------------------


def _resume(org_id, candidate_id, key="k"):
    return Resume(
        org_id=org_id,
        candidate_id=candidate_id,
        s3_key=f"{org_id}/{candidate_id}/{key}.pdf",
        filename="cv.pdf",
        content_type="application/pdf",
    )


async def test_a_candidate_may_write_their_own_resume(tenant_session, org_a):
    """The candidate holds the file, so they must be able to record the upload."""
    async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
        s.add(_resume(org_a.org_id, org_a.candidate_id))
        await s.flush()

    async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
        assert len((await s.execute(select(Resume))).scalars().all()) == 1


async def test_a_candidate_cannot_write_a_resume_for_someone_else(tenant_session, org_a):
    """WITH CHECK, not just USING: the insert names another owner."""
    other = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(Candidate(id=other, org_id=org_a.org_id, email="other@alpha.test"))

    with pytest.raises(ProgrammingError, match="row-level security"):
        async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
            s.add(_resume(org_a.org_id, other))
            await s.flush()


async def test_a_candidate_cannot_reassign_their_resume_to_another_candidate(
    tenant_session, org_a
):
    """The reason WITH CHECK repeats the ownership predicate rather than
    deferring to USING: USING governs the rows an UPDATE may find, WITH CHECK
    the rows it may leave behind."""
    other = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(Candidate(id=other, org_id=org_a.org_id, email="victim@alpha.test"))
        await s.flush()
        s.add(_resume(org_a.org_id, org_a.candidate_id))

    with pytest.raises(ProgrammingError, match="row-level security"):
        async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
            await s.execute(update(Resume).values(candidate_id=other))


async def test_a_candidate_cannot_see_another_candidates_resume(tenant_session, org_a):
    other = uuid.uuid4()
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        s.add(Candidate(id=other, org_id=org_a.org_id, email="other2@alpha.test"))
        await s.flush()
        s.add(_resume(org_a.org_id, other))

    async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
        assert (await s.execute(select(Resume))).scalars().all() == []


async def test_a_candidate_cannot_read_resume_chunks(tenant_session, org_a):
    """The retrieval index is derived data the recruiter's pipeline reads. A
    candidate paging through the vector form of their own CV serves nobody."""
    async with tenant_session(org_a.org_id, "user", org_a.user_id) as s:
        resume = _resume(org_a.org_id, org_a.candidate_id)
        s.add(resume)
        await s.flush()
        s.add(
            ResumeChunk(
                org_id=org_a.org_id, resume_id=resume.id, ordinal=0, content="[skills] Python"
            )
        )

    async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
        assert (await s.execute(select(ResumeChunk))).scalars().all() == []


# --- The system actor -------------------------------------------------------


async def test_the_system_actor_reads_staff_tables_within_its_org(
    tenant_session, org_a, org_b
):
    """Celery workers run as 'system'. Without this they read nothing at all and
    every background task is a silent no-op."""
    async with tenant_session(org_a.org_id, "system", None) as s:
        assert len((await s.execute(select(User))).scalars().all()) == 1
        assert (await s.execute(text("SELECT count(*) FROM invites"))).scalar() is not None

        # Still exactly one org. The widening is on the actor axis only.
        assert [c.org_id for c in (await s.execute(select(Candidate))).scalars()] == [
            org_a.org_id
        ]


async def test_the_system_actor_cannot_cross_orgs(tenant_session, org_a, org_b):
    async with tenant_session(org_a.org_id, "system", None) as s:
        assert await s.get(Candidate, org_b.candidate_id) is None


async def test_candidate_actor_cannot_write(tenant_session, org_a):
    """WITH CHECK omits the candidate branch entirely."""
    with pytest.raises(ProgrammingError, match="row-level security"):
        async with tenant_session(org_a.org_id, "candidate", org_a.candidate_id) as s:
            s.add(
                Interview(
                    id=uuid.uuid4(),
                    org_id=org_a.org_id,
                    candidate_id=org_a.candidate_id,
                )
            )
            await s.flush()


def test_candidate_scoped_registry_matches_reality():
    """The scoped-table registry must name columns that exist on the models."""
    models = {"interviews": Interview, "candidates": Candidate}
    for table, column in CANDIDATE_SCOPED.items():
        assert column in models[table].__table__.columns, (
            f"CANDIDATE_SCOPED names {table}.{column}, which does not exist"
        )
