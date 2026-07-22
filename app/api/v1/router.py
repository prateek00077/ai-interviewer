"""Aggregates all v1 routers under /api/v1.

The remaining v1 modules are still docstring stubs; each lands with the feature
that needs it.
"""

from fastapi import APIRouter

from app.api.v1 import auth, candidates, jobs, users

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(candidates.router)
api_router.include_router(jobs.router)
