"""A mutating response must not arrive before its transaction commits.

THE BUG. ``get_db`` is a dependency with ``yield``, and FastAPI runs the exit
code of those -- where the commit lives -- *after* the response has been sent.
So a client could be told a row existed about a millisecond before it did.

MEASURED before the fix: 7 of 20 ``POST /jobs`` calls had not committed when the
201 arrived. The browser console posted a job, immediately posted its
description, and got "Job not found." for an id it had just been handed. curl
and httpx were usually slow enough to miss the window, which is exactly why no
test caught it -- and why the coverage-style assertions below matter more than
a timing test that would be flaky either way.
"""

import uuid

import pytest

from app.api.routing import CommittingRoute

pytestmark = pytest.mark.integration


def _auth(org: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {org['tokens']['access_token']}"}


# --- The behaviour ----------------------------------------------------------


async def test_a_created_row_is_usable_by_the_very_next_request(
    api_client, registered_org
):
    """The console's exact sequence: create a job, then immediately use its id.

    No sleep, no retry. If the commit moves back to dependency teardown this
    goes back to failing intermittently.
    """
    for _ in range(10):
        job = await api_client.post(
            "/api/v1/jobs", headers=_auth(registered_org), json={"title": "Staff Engineer"}
        )
        assert job.status_code == 201, job.text

        described = await api_client.post(
            f"/api/v1/jobs/{job.json()['id']}/descriptions",
            headers=_auth(registered_org),
            json={"content": "A description long enough to pass validation." * 3},
        )
        assert described.status_code == 201, (
            f"the job was invisible to the next request: {described.text}"
        )


async def test_a_created_row_is_visible_to_an_independent_connection(
    api_client, registered_org
):
    """The stronger form. The API's own next request might see uncommitted data
    through a pooled connection; a session opened from scratch cannot.
    """
    from sqlalchemy import select

    from app.db.session import tenant_session
    from app.models.job import Job

    job = await api_client.post(
        "/api/v1/jobs", headers=_auth(registered_org), json={"title": "Staff Engineer"}
    )
    job_id = uuid.UUID(job.json()["id"])

    org_id = uuid.UUID(registered_org["org_id"])
    async with tenant_session(org_id, "system", None) as session:
        found = (await session.execute(select(Job).where(Job.id == job_id))).scalar_one_or_none()
    assert found is not None, "the row was not committed when the response arrived"


async def test_a_rejected_request_commits_nothing(api_client, registered_org):
    """The commit runs on the success path only. A 4xx must leave no trace."""
    from sqlalchemy import func, select

    from app.db.session import tenant_session
    from app.models.job import Job

    org_id = uuid.UUID(registered_org["org_id"])
    async with tenant_session(org_id, "system", None) as session:
        before = await session.scalar(select(func.count()).select_from(Job))

    rejected = await api_client.post(
        "/api/v1/jobs", headers=_auth(registered_org), json={"title": ""}
    )
    assert rejected.status_code == 422

    async with tenant_session(org_id, "system", None) as session:
        after = await session.scalar(select(func.count()).select_from(Job))
    assert after == before


# --- The wiring that makes it hold ------------------------------------------


def _v1_router_modules() -> list:
    """Every module under app/api/v1 that defines a router, discovered.

    Enumerated rather than listed, so a new endpoint module is covered the day
    it is added instead of the day someone remembers to add it here.
    """
    import importlib
    import pkgutil

    import app.api.v1 as v1

    modules = []
    for info in pkgutil.iter_modules(v1.__path__):
        if info.name == "router":  # the aggregator holds no routes of its own
            continue
        module = importlib.import_module(f"app.api.v1.{info.name}")
        if hasattr(module, "router"):
            modules.append(module)
    return modules


def test_every_v1_router_commits_before_responding():
    """The coverage guard.

    ``include_router`` does not attach a route class, so each router names it at
    construction -- which is forgettable, and forgetting it fails silently and
    intermittently on exactly the routes that write. This fails loudly instead.
    """
    modules = _v1_router_modules()
    # Without this the assertion below passes on an empty list, which is how a
    # coverage guard quietly stops guarding.
    assert len(modules) >= 10, f"discovery found only {len(modules)} router modules"

    offenders = [
        m.__name__ for m in modules if m.router.route_class is not CommittingRoute
    ]
    assert not offenders, (
        "these routers use the default APIRoute, so their transactions commit "
        f"after the response is sent: {offenders}"
    )


def test_every_mounted_api_route_uses_the_committing_class():
    """The same property one layer down, against what is actually mounted.

    Guards the case where a router is constructed correctly but its routes are
    added somewhere that bypasses it.
    """
    from app.main import create_app

    app = create_app()

    def collect(router, out):
        for route in getattr(router, "routes", []):
            if hasattr(route, "original_router"):
                collect(route.original_router, out)
            elif hasattr(route, "path"):
                out.append(route)

    routes: list = []
    collect(app.router, routes)

    # Paths that legitimately hold no session: ops probes, the dev console, and
    # the generated schema.
    exempt = {"/health", "/ready", "/metrics", "/dev", "/openapi.json", "/docs",
              "/docs/oauth2-redirect"}
    api_routes = [r for r in routes if r.path not in exempt and hasattr(r, "methods")]
    assert len(api_routes) >= 40, f"only {len(api_routes)} routes found; walk is broken"

    offenders = [
        f"{sorted(r.methods)} {r.path}"
        for r in api_routes
        if not isinstance(r, CommittingRoute)
    ]
    assert not offenders, offenders


def test_the_route_class_skips_reads():
    """A read holds nothing worth committing, and the extra round trip to
    Postgres would land on the hot path."""
    from app.api.routing import READ_ONLY_METHODS

    assert READ_ONLY_METHODS == frozenset({"GET", "HEAD", "OPTIONS"})
