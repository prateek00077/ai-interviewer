"""Aggregates all v1 routers under /api/v1.

Only ``auth`` is wired in this slice. The remaining v1 modules are still
docstring stubs; each lands with the feature that needs it.
"""

from fastapi import APIRouter

from app.api.v1 import auth

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
