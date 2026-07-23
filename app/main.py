"""FastAPI app factory: lifespan, middleware, router mounting.

The Redis pool and refresh-token store are created once in the lifespan and
parked on ``app.state``. Creating them at import time would bind them to
whichever event loop happened to import the module, which breaks under both
pytest and multi-worker uvicorn.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from app.api import ops
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.events import bus
from app.core.exceptions import register_exception_handlers
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.session import dispose_engine
from app.modules.auth.tokens import RefreshTokenStore
from app.modules.interview import service as interview_service
from app.modules.interview import transcript
from app.modules.proctoring import voice_signals

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()

    # Wire the event bus once, at startup. Subscribers are module-level
    # functions, so registering at import time would attach a second handler
    # every time a test re-imported the app.
    interview_service.register()
    transcript.register()
    voice_signals.register()

    # Explicitly cleared, not merely defaulted. The flag is a module global, so
    # a process that starts a second app after shutting one down -- every test
    # run, and uvicorn --reload -- would otherwise come up permanently draining
    # and refuse every voice session with no obvious reason.
    ops.set_draining(False)

    redis = Redis.from_url(settings.redis_url)
    app.state.redis = redis
    app.state.token_store = RefreshTokenStore(redis)
    log.info("startup", environment=settings.environment)
    try:
        yield
    finally:
        # Refuse NEW voice sessions before tearing anything down. A candidate
        # who connects during shutdown would otherwise get a pipeline built
        # against a Redis pool that is about to close, which fails a minute
        # later mid-answer rather than at the door.
        ops.set_draining(True)

        # Live voice sessions first: each one is a real person mid-sentence, and
        # ending them cleanly is what flushes the final turn and uploads the
        # recording. Only then drain the bus that carries those events.
        from app.modules.voice import session_manager

        await session_manager.stop_all(reason="abandoned")

        # Let in-flight handlers finish before the loop closes, so the last turn
        # of a live interview is not lost on shutdown.
        await bus.drain()
        bus.clear()
        await redis.aclose()
        await dispose_engine()
        log.info("shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AI Interviewer API",
        version="0.1.0",
        lifespan=lifespan,
        # Docs are useful in development and an attack-surface map in production.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        # An explicit allowlist. "*" plus credentials is rejected by browsers
        # anyway, and would be wrong here regardless.
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)
    app.include_router(api_router)
    # Unversioned and unauthenticated, deliberately: a probe is issued by an
    # orchestrator that has no credentials, and /health pinned to /api/v1 would
    # have to be re-pointed at every API version bump.
    app.include_router(ops.router)

    if not settings.is_production:
        # A console that can mint invites has no business being reachable on a
        # deployed instance, path-obscurity notwithstanding. Same rule as /docs.
        from app.api import dev

        app.include_router(dev.router)

    return app


app = create_app()
