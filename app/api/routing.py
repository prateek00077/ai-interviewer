"""The route class every API router uses: commit before the response is sent.

THE BUG THIS FIXES. ``get_db`` is a dependency with ``yield``, and FastAPI runs
the exit code of those -- where ``tenant_session`` commits -- *after* the
response has gone out. The client is therefore told a row exists roughly a
millisecond before it does.

MEASURED against this application: 7 of 20 ``POST /jobs`` calls had not
committed at the moment the 201 arrived. The test console posts a job and
immediately posts its description, and got "Job not found." for a job whose id
it had just been handed. curl and httpx were usually slow enough to miss the
window; a browser on localhost is not. Every create-then-use flow in the product
had it.

WHY NOT MIDDLEWARE. Tried first, and it is actively wrong here:
``BaseHTTPMiddleware`` runs the downstream app in a separate anyio task, so
``call_next`` returns while the endpoint is still unwinding. Committing there
races the dependency's ``session.close()`` and raises
``IllegalStateChangeError: Method 'close()' can't be called here; method
'commit()' is already in progress``.

WHY NOT AN EXPLICIT COMMIT IN EVERY MUTATING ROUTE. About twenty routes, each
failing silently and intermittently if it is missed. Two routes already did it
by hand, which in hindsight was this bug being worked around locally instead of
found.

A route class wraps the endpoint call itself, so the commit lands after the
handler and before the dependency teardown -- verified ordering, and pinned by
``tests/integration/test_commit_ordering.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

import structlog
from fastapi import Request, Response
from fastapi.routing import APIRoute

log = structlog.get_logger(__name__)

# Verbs that cannot have written anything worth committing, so they skip the
# extra round trip to Postgres on the hot read path.
READ_ONLY_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class CommittingRoute(APIRoute):
    """Commits ``request.state.db`` after the handler, before the response."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original = super().get_route_handler()

        async def handler(request: Request) -> Response:
            response = await original(request)

            if request.method in READ_ONLY_METHODS:
                return response

            session = getattr(request.state, "db", None)
            if session is None:
                # Unauthenticated routes, and anything opening its own
                # ``tenant_session``, manage their own transactions.
                return response

            # A handler that raised never reaches here -- ``original`` propagates
            # and the dependency rolls back. This is the success path only, so a
            # 4xx returned rather than raised is the one case worth checking.
            if response.status_code >= 400:
                return response

            await session.commit()
            return response

        return handler
