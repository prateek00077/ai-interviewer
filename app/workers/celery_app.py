"""Celery application and broker configuration.

Every task in this codebase is async underneath -- the DB session, the NIM
client, S3 -- while Celery workers are synchronous. ``run_async`` bridges that
with one event loop per worker process rather than ``asyncio.run`` per task:
``asyncio.run`` closes the loop it creates, and SQLAlchemy's asyncpg pool holds
connections bound to whichever loop opened them, so the second task in a worker
would fail with "Event loop is closed".

Task settings worth naming:

- ``acks_late`` with ``reject_on_worker_lost``: a task is acknowledged after it
  finishes, so a worker killed mid-parse redelivers rather than silently losing
  the job. This makes every task at-least-once, which is why they are written to
  be idempotent on their subject id.
- ``worker_prefetch_multiplier=1``: these tasks are long and uneven (a 30-second
  transcription next to a 200ms status flip). Prefetching would let one worker
  sit on a queue of jobs while another idles.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

import structlog
from celery import Celery

from app.core.config import settings
from app.core.logging import configure_logging

log = structlog.get_logger(__name__)

celery_app = Celery(
    "ai_interviewer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.workers.tasks.resume_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    # A stuck NIM call must not hold a worker forever. soft fires first and
    # raises SoftTimeLimitExceeded, giving a task the chance to record why it
    # failed before hard termination.
    task_soft_time_limit=600,
    task_time_limit=660,
    result_expires=86_400,
    broker_connection_retry_on_startup=True,
)


@celery_app.on_after_configure.connect  # type: ignore[misc]
def _setup_logging(sender: Any, **_: Any) -> None:
    configure_logging()


T = TypeVar("T")

# One loop per worker process, created lazily. Module-level creation would bind
# it at import time in the parent process, and forked children would inherit a
# loop that is not theirs.
_loop: asyncio.AbstractEventLoop | None = None


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on this worker's persistent event loop."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop.run_until_complete(coro)
