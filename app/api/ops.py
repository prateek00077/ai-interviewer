"""Liveness, readiness, and metrics.

THREE ENDPOINTS BECAUSE THEY ANSWER THREE DIFFERENT QUESTIONS, and conflating
them is how a deployment takes itself down:

- ``/health`` is liveness. Is this process running? It touches nothing external
  and always returns 200 while the event loop turns. An orchestrator restarts a
  container on this signal, so wiring it to Postgres would mean a database
  blip restarts every API pod simultaneously -- turning a recoverable outage
  into a thundering herd against a database that is already struggling.

- ``/ready`` is readiness. Can this process serve a request? It checks
  Postgres, Redis and S3, and a failure pulls the pod out of the load balancer
  without killing it. That is the correct response to a dependency being down:
  stop sending traffic, keep the process alive so it can recover.

- ``/metrics`` is what to page on.

``/ready` also reports draining. During shutdown the process still has live
voice sessions to finish, and it must stop receiving new ones well before it
stops existing -- see ``draining`` below.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from fastapi import APIRouter, Request, Response, status
from sqlalchemy import text

from app.core.config import settings
from app.db.session import SessionLocal

log = structlog.get_logger(__name__)

router = APIRouter(tags=["ops"])

# A readiness probe that hangs is worse than one that fails: the orchestrator
# waits out its own timeout on every check instead of getting a fast negative.
CHECK_TIMEOUT_SECS = 2.0

_started_at = time.monotonic()

# Flipped by the lifespan before it starts closing anything. New voice sessions
# are refused from that moment, while sessions already running are allowed to
# finish -- each one is a real person mid-sentence.
_draining = False


def set_draining(value: bool) -> None:
    global _draining
    _draining = value
    log.info("ops.draining", draining=value)


def is_draining() -> bool:
    return _draining


async def _check_postgres() -> None:
    async with SessionLocal() as session:
        await session.execute(text("SELECT 1"))


async def _check_redis(request: Request) -> None:
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise RuntimeError("redis pool is not initialised")
    await redis.ping()


async def _check_storage() -> None:
    from app.integrations import storage

    await storage.check_bucket(settings.s3_bucket_reports)


async def _run(name: str, coro: Any) -> tuple[str, str | None]:
    try:
        await asyncio.wait_for(coro, timeout=CHECK_TIMEOUT_SECS)
    except TimeoutError:
        return name, f"timed out after {CHECK_TIMEOUT_SECS}s"
    except Exception as exc:  # noqa: BLE001 - the probe reports, it does not raise
        return name, str(exc)[:200]
    return name, None


@router.get("/health", summary="Liveness: is the process running")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness: can the process serve traffic")
async def ready(request: Request, response: Response) -> dict[str, Any]:
    """Check every dependency, concurrently, and report each one by name.

    Concurrently because the probe interval is the budget: three sequential
    two-second timeouts is a six-second probe, which most orchestrators would
    have given up on.

    All checks run even when the first fails, so one probe tells an operator
    everything that is wrong rather than only the first thing.
    """
    checks = await asyncio.gather(
        _run("postgres", _check_postgres()),
        _run("redis", _check_redis(request)),
        _run("storage", _check_storage()),
    )
    failures = {name: error for name, error in checks if error is not None}

    if failures or _draining:
        # 503, not 500: this instance is temporarily unfit, not broken. The
        # distinction is what tells a load balancer to retry elsewhere.
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "draining" if _draining else ("degraded" if failures else "ready"),
        "checks": {name: ("ok" if error is None else error) for name, error in checks},
    }


@router.get("/metrics", summary="Operational counters")
async def metrics(request: Request) -> dict[str, Any]:
    """A small JSON snapshot rather than a Prometheus exposition format.

    Deliberate: nothing in this deployment scrapes Prometheus yet, and adding
    prometheus-client to emit four numbers would be a dependency carried for a
    format nobody reads. The shape is stable, so a scraper can be added in
    front of it without changing what is measured.
    """
    from app.core.events import bus
    from app.modules.voice import session_manager

    return {
        "uptime_seconds": round(time.monotonic() - _started_at, 1),
        "environment": settings.environment,
        "draining": _draining,
        # The one number worth paging on: these are live calls with real people
        # on them, and a deploy that drops them is visible to candidates.
        "voice_sessions_active": session_manager.active_count(),
        "event_bus_pending": bus.pending_count(),
    }
