"""API-layer dependency wiring.

A re-export shim over ``app.core.dependencies`` so routers have exactly one
import site. If the dependency layer is ever split or relocated, only this file
changes rather than every router.
"""

from app.core.dependencies import (
    CurrentPrincipal,
    Principal,
    ScopedSession,
    client_ip,
    get_current_candidate,
    get_current_org,
    get_current_user,
    get_db,
    get_principal,
    get_redis,
    get_token_store,
    get_unscoped_db,
    login_rate_limit,
    rate_limit,
    register_rate_limit,
    require_role,
)

__all__ = [
    "CurrentPrincipal",
    "Principal",
    "ScopedSession",
    "client_ip",
    "get_current_candidate",
    "get_current_org",
    "get_current_user",
    "get_db",
    "get_principal",
    "get_redis",
    "get_token_store",
    "get_unscoped_db",
    "login_rate_limit",
    "rate_limit",
    "register_rate_limit",
    "require_role",
]
