"""Test database, factories, and auth fixtures.

The app engine is pinned to a SINGLE physical connection (pool_size=1,
max_overflow=0). That is deliberate: it guarantees consecutive "requests" reuse
the same connection, which is the only way the org-leak test in test_rls.py can
actually catch a ``SET`` that should have been ``SET LOCAL``.

Seeding uses ordinary org-scoped sessions rather than a privileged bypass. It
works because the organizations policy matches on ``id = app.current_org()``:
generate the org id first, open a session with it, then insert. If seeding needs
a bypass, the policies are wrong.
"""

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.core.security import hash_password
from app.db.session import tenant_session as _tenant_session
from app.models.interview import Interview
from app.models.org import Organization
from app.models.user import Candidate, User, UserRole


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def app_engine_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """A sessionmaker over a one-connection pool, as the unprivileged app role.

    Session-scoped, and so is the loop (see asyncio_default_*_loop_scope in
    pyproject.toml). Both have to agree: a session-scoped engine on a per-test
    loop hands asyncpg a closed loop at disposal time.
    """
    engine = create_async_engine(
        settings.database_url,
        pool_size=1,
        max_overflow=0,
        pool_pre_ping=True,
    )
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture
def tenant_session(app_engine_factory):
    """tenant_session bound to the pinned test pool."""

    def _factory(org_id, actor_kind="user", actor_id=None):
        return _tenant_session(org_id, actor_kind, actor_id, factory=app_engine_factory)

    return _factory


@pytest.fixture
def unscoped_session(app_engine_factory):
    from app.db.session import unscoped_session as _unscoped

    def _factory():
        return _unscoped(factory=app_engine_factory)

    return _factory


class OrgFixture:
    """One tenant's seeded rows, for readable assertions."""

    def __init__(
        self,
        org_id: uuid.UUID,
        user_id: uuid.UUID,
        candidate_id: uuid.UUID,
        interview_id: uuid.UUID,
        slug: str,
        email: str,
    ) -> None:
        self.org_id = org_id
        self.user_id = user_id
        self.candidate_id = candidate_id
        self.interview_id = interview_id
        self.slug = slug
        self.email = email


async def _seed_org(tenant_session, label: str) -> OrgFixture:
    org_id = uuid.uuid4()
    user_id = uuid.uuid4()
    candidate_id = uuid.uuid4()
    interview_id = uuid.uuid4()
    slug = f"{label}-{org_id.hex[:8]}"

    async with tenant_session(org_id, "user", user_id) as s:
        s.add(Organization(id=org_id, name=f"Org {label}", slug=slug))
        await s.flush()
        s.add(
            User(
                id=user_id,
                org_id=org_id,
                email=f"admin@{slug}.example.com",
                hashed_password=hash_password("correct-horse-battery"),
                full_name=f"Admin {label}",
                role=UserRole.ADMIN,
            )
        )
        s.add(
            Candidate(
                id=candidate_id,
                org_id=org_id,
                email=f"candidate@{slug}.test",
                full_name=f"Candidate {label}",
            )
        )
        await s.flush()
        s.add(Interview(id=interview_id, org_id=org_id, candidate_id=candidate_id))

    email = f"admin@{slug}.example.com"
    return OrgFixture(org_id, user_id, candidate_id, interview_id, slug, email)


@pytest_asyncio.fixture
async def org_a(tenant_session) -> AsyncIterator[OrgFixture]:
    fixture = await _seed_org(tenant_session, "alpha")
    yield fixture
    await _cleanup(tenant_session, fixture.org_id)


@pytest_asyncio.fixture
async def org_b(tenant_session) -> AsyncIterator[OrgFixture]:
    fixture = await _seed_org(tenant_session, "bravo")
    yield fixture
    await _cleanup(tenant_session, fixture.org_id)


async def _cleanup(tenant_session, org_id: uuid.UUID) -> None:
    # Deleting the org cascades to every tenant row beneath it.
    async with tenant_session(org_id) as s:
        org = await s.get(Organization, org_id)
        if org is not None:
            await s.delete(org)


# --- HTTP ------------------------------------------------------------------


@pytest_asyncio.fixture(loop_scope="session")
async def api_client() -> AsyncIterator[httpx.AsyncClient]:
    """The real app over ASGI, with its lifespan actually run.

    The lifespan is what creates the Redis pool and the refresh-token store, so
    skipping it (as a bare ASGITransport does) leaves ``app.state`` empty and
    every auth route fails on an attribute error rather than on its own logic.

    Routes here use the application's own engine, not the pinned single-connection
    test pool -- an HTTP test that shared one connection across concurrent
    requests would deadlock.
    """
    from app.main import create_app

    app = create_app()
    async with app.router.lifespan_context(app):
        # The rate limiter is real and every test registers from 127.0.0.1, so
        # without this the suite trips its own 429 after five tests. Cleared per
        # test rather than disabled -- test_rate_limiting.py still exercises it.
        await _clear_rate_limits(app.state.redis)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _clear_rate_limits(redis) -> None:
    keys = [key async for key in redis.scan_iter(match="rl:*")]
    if keys:
        await redis.delete(*keys)


@pytest_asyncio.fixture(loop_scope="session")
async def registered_org(api_client) -> AsyncIterator[dict]:
    """A tenant created through the public endpoint, torn down afterwards."""
    slug = f"acme-{uuid.uuid4().hex[:10]}"
    payload = {
        "org_name": "Acme Inc",
        "slug": slug,
        "admin_email": f"admin@{slug}.example.com",
        "admin_password": "correct-horse-battery-staple",
        "admin_full_name": "Ada Admin",
    }
    response = await api_client.post("/api/v1/auth/register-org", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()

    yield {**payload, **body}

    async with _tenant_session(body["org_id"]) as s:
        org = await s.get(Organization, uuid.UUID(body["org_id"]))
        if org is not None:
            await s.delete(org)
