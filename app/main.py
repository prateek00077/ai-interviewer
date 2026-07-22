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

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import RequestContextMiddleware, configure_logging
from app.db.session import dispose_engine
from app.modules.auth.tokens import RefreshTokenStore

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    redis = Redis.from_url(settings.redis_url)
    app.state.redis = redis
    app.state.token_store = RefreshTokenStore(redis)
    log.info("startup", environment=settings.environment)
    try:
        yield
    finally:
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

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
