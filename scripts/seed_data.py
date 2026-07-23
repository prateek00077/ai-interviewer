"""Seed an org, recruiter, job, and sample candidate for local development.

Goes through ``auth_service.register_org`` rather than inserting an
Organization directly. That is not just reuse: creating a tenant is the one
operation that has to work with no org context yet, and the way it does -- by
generating the id in Python and opening the session with it already set -- is a
property of the RLS policies worth exercising. A seed script with its own
bootstrap path could keep working after those policies broke.

Everything after the org runs org-scoped, exactly as the API does, so this
script cannot create rows the application would be unable to read.

Idempotent on the slug: re-running prints the credentials again rather than
failing. The most common reason to run this twice is "what was the password".

    python scripts/seed_data.py [--slug acme]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.exceptions import ConflictError  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.db.session import tenant_session  # noqa: E402
from app.models.job import Job, JobDescription, JobStatus  # noqa: E402
from app.models.user import Candidate, User, UserRole  # noqa: E402
from app.modules.auth import service as auth_service  # noqa: E402

# Development only. This value is in the source of a public repository, which
# is why the guard below refuses to run against production.
SEED_PASSWORD = "correct-horse-battery-staple"  # noqa: S105

JOB_TITLE = "Senior Backend Engineer"
JOB_DESCRIPTION = """\
We are hiring a senior backend engineer to work on a multi-tenant Python
platform: FastAPI, Postgres, async SQLAlchemy, Celery and S3-compatible object
storage.

You will own service boundaries end to end, from schema design through to the
queries that run against it in production. We care most about how you reason
about tradeoffs -- consistency against latency, isolation against complexity --
and about whether you can explain a decision you made and what it cost you.

Requirements: strong Python, real experience with relational data modelling and
concurrency, and comfort operating what you build.
"""


async def seed(slug: str) -> None:
    if settings.is_production:
        raise SystemExit("refusing to seed a production environment")

    admin_email = f"admin@{slug}.example.com"
    recruiter_email = f"recruiter@{slug}.example.com"
    candidate_email = f"candidate@{slug}.example.com"

    try:
        principal = await auth_service.register_org(
            org_name=slug.title(),
            slug=slug,
            admin_email=admin_email,
            admin_password=SEED_PASSWORD,
            admin_full_name="Seed Admin",
        )
    except ConflictError:
        # register_org does not disclose which unique constraint collided --
        # deliberately, since that would make signup an account-existence
        # oracle. Here we are allowed to look.
        print(f"org '{slug}' already exists; credentials unchanged")
        _print_credentials(None, admin_email, recruiter_email, candidate_email)
        return

    org_id = principal.org_id

    async with tenant_session(org_id, "user", principal.user_id) as session:
        session.add(
            User(
                id=uuid.uuid4(),
                org_id=org_id,
                email=recruiter_email,
                hashed_password=hash_password(SEED_PASSWORD),
                full_name="Seed Recruiter",
                role=UserRole.RECRUITER,
            )
        )

        job = Job(
            id=uuid.uuid4(),
            org_id=org_id,
            title=JOB_TITLE,
            status=JobStatus.OPEN,
            created_by_user_id=principal.user_id,
        )
        session.add(job)
        await session.flush()

        session.add(
            JobDescription(
                org_id=org_id,
                job_id=job.id,
                content=JOB_DESCRIPTION,
                version=1,
                is_active=True,
            )
        )
        session.add(
            Candidate(
                id=uuid.uuid4(),
                org_id=org_id,
                email=candidate_email,
                full_name="Sample Candidate",
            )
        )
        job_id = job.id

    # Read back through a scoped session, which proves the policies admit the
    # rows we just wrote rather than merely that the INSERTs did not error.
    async with tenant_session(org_id, "user", principal.user_id) as session:
        assert (await session.execute(select(Job).where(Job.id == job_id))).scalar_one()

    print(f"seeded org '{slug}' ({org_id})")
    _print_credentials(org_id, admin_email, recruiter_email, candidate_email)


def _print_credentials(
    org_id: uuid.UUID | None, admin: str, recruiter: str, candidate: str
) -> None:
    if org_id is not None:
        print(f"  org_id     {org_id}")
    print(f"  admin      {admin} / {SEED_PASSWORD}")
    print(f"  recruiter  {recruiter} / {SEED_PASSWORD}")
    print(f"  candidate  {candidate}  (no password -- invited by link)")
    print()
    print("  curl -s localhost:8000/api/v1/auth/login -H 'content-type: application/json' \\")
    print(f"""    -d '{{"email":"{admin}","password":"{SEED_PASSWORD}"}}'""")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="acme", help="org slug to create (default: acme)")
    asyncio.run(seed(parser.parse_args().slug))


if __name__ == "__main__":
    main()
