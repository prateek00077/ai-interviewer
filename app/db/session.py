"""Async SQLAlchemy engine/session. Sets app.current_org per request for RLS.

Two Postgres/SQLAlchemy traps are handled here deliberately:

1. ``SET LOCAL app.current_org = $1`` is a *syntax error* -- SET does not accept
   bind parameters. The usual workaround is f-stringing the value in, which is a
   SQL injection hole. ``set_config(name, value, is_local => true)`` is an
   ordinary function call: it takes real binds and is transaction-scoped.

2. ``SET LOCAL`` dies at COMMIT. Applying the GUCs once at the top of a request
   works right up until application code commits mid-request, after which
   SQLAlchemy silently opens a fresh transaction with no org context and every
   subsequent query returns zero rows. Hooking ``after_begin`` re-applies them on
   *every* transaction the session opens, so that failure mode cannot occur.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Literal

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import settings

ActorKind = Literal["user", "candidate", "system"]

_GUC_SQL = text(
    "SELECT set_config('app.current_org', :org, true), "
    "       set_config('app.actor_kind',  :kind, true), "
    "       set_config('app.actor_id',    :actor, true)"
)

engine: AsyncEngine = create_async_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    echo=False,
)

SessionLocal = async_sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)


@event.listens_for(AsyncSession.sync_session_class, "after_begin")
def _apply_tenant_gucs(session: Any, transaction: Any, connection: Any) -> None:
    """Re-apply tenant GUCs on every transaction this session opens.

    Runs inside SQLAlchemy's greenlet, so the sync execute API is correct here.
    """
    ctx = session.info.get("tenant")
    if ctx is None:
        # Unscoped session. Under RLS this sees nothing, which is the safe default.
        return
    connection.execute(_GUC_SQL, ctx)


def _tenant_ctx(
    org_id: uuid.UUID | str, actor_kind: ActorKind, actor_id: uuid.UUID | str | None
) -> dict[str, str]:
    return {
        "org": str(org_id),
        "kind": actor_kind,
        # Empty string rather than None: set_config rejects NULL, and the
        # app.actor_id() accessor maps '' back to NULL.
        "actor": str(actor_id) if actor_id is not None else "",
    }


@asynccontextmanager
async def tenant_session(
    org_id: uuid.UUID | str,
    actor_kind: ActorKind = "user",
    actor_id: uuid.UUID | str | None = None,
    factory: async_sessionmaker[AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    """An org-scoped session. Commits on clean exit, rolls back on error.

    ``factory`` exists so tests can supply a single-connection pool and prove
    that org context does not leak when a connection is reused.

    The transaction is left to SQLAlchemy's autobegin rather than opened with
    ``session.begin()``: that context manager forbids a commit inside the block,
    which would make the mid-request commit -- the exact case ``after_begin``
    exists to survive -- impossible to reach *and* impossible to test.
    """
    session = (factory or SessionLocal)()
    session.info["tenant"] = _tenant_ctx(org_id, actor_kind, actor_id)
    try:
        yield session
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def unscoped_session(
    factory: async_sessionmaker[AsyncSession] | None = None,
) -> AsyncIterator[AsyncSession]:
    """A session with no org context.

    Under RLS this reads zero rows from every tenant table. It exists for the
    login path, which reaches the DB only through ``app.lookup_user_for_auth``
    (a narrow SECURITY DEFINER function), and for health checks.
    """
    session = (factory or SessionLocal)()
    try:
        yield session
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    finally:
        await session.close()


async def dispose_engine() -> None:
    await engine.dispose()
