"""A manual test console, served only outside production.

WHY THIS IS SERVED BY THE API rather than opened as a file:// page. The console
talks to this origin, and same-origin means the CORS allowlist does not have to
grow a `null` entry to accommodate a test page -- which would be a permanent
hole punched for a temporary convenience. It also means the page is impossible
to run against the wrong server by accident.

WHY IT IS MOUNTED CONDITIONALLY. ``create_app`` skips this router entirely when
``ENVIRONMENT=production``, the same way ``/docs`` is skipped. A test console
that can mint invites is not something to leave reachable on a deployed
instance and rely on nobody guessing the path.

The page holds no secrets of its own: every call it makes is one a real client
would make, with a token the operator obtained through the ordinary login or
redeem flow. It is a driver, not a backdoor.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

log = structlog.get_logger(__name__)

router = APIRouter(tags=["dev"], include_in_schema=False)

CONSOLE = Path(__file__).resolve().parents[2] / "static" / "console.html"


@router.get("/dev", response_class=HTMLResponse, summary="Manual test console")
async def console() -> HTMLResponse:
    if not CONSOLE.exists():
        return HTMLResponse("<h1>console.html is missing</h1>", status_code=500)
    # Read per request rather than cached at import: the whole point of this
    # page is that you edit it and hit refresh.
    return HTMLResponse(CONSOLE.read_text())
