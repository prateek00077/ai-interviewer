"""Domain exception types and their HTTP handlers.

Every error the API returns has a stable machine-readable ``code``. Client-facing
messages are deliberately coarse: auth failures must not tell an attacker *which*
part of their attempt was wrong. Detail goes to the logs, not the response body.
"""

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = structlog.get_logger(__name__)


class AppError(Exception):
    """Base for every error we raise deliberately."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, **context: Any) -> None:
        self.message = message or self.message
        # Context is logged, never serialized into the response.
        self.context = context
        super().__init__(self.message)


# --- 401 ---
class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"
    message = "Authentication required."


class InvalidCredentialsError(AuthenticationError):
    code = "invalid_credentials"
    # Identical for unknown user, wrong password, inactive user and inactive org.
    message = "Incorrect email or password."


class InvalidTokenError(AuthenticationError):
    code = "invalid_token"
    message = "Token is invalid or has expired."


class TokenReuseDetectedError(AuthenticationError):
    code = "refresh_reuse_detected"
    message = "Session revoked. Please sign in again."


# --- 403 / 404 / 409 / 410 / 429 ---
class PermissionDeniedError(AppError):
    status_code = 403
    code = "permission_denied"
    message = "You do not have permission to perform this action."


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "Resource not found."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "Resource already exists."


class InviteUnusableError(AppError):
    status_code = 410
    code = "invite_unusable"
    # Expired, revoked and exhausted are indistinguishable to the caller by design.
    message = "This invitation link is no longer valid."


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "Too many attempts. Please try again later."

    def __init__(self, message: str | None = None, retry_after: int = 60, **context: Any) -> None:
        super().__init__(message, **context)
        self.retry_after = retry_after


def _body(code: str, message: str, request: Request) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": getattr(request.state, "request_id", None),
        }
    }


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(request: Request, exc: AppError) -> JSONResponse:
        logger = log.bind(code=exc.code, status=exc.status_code, **exc.context)
        if exc.status_code >= 500:
            logger.error("app_error", exc_info=exc)
        else:
            logger.info("app_error", detail=str(exc))

        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitedError):
            headers["Retry-After"] = str(exc.retry_after)
        if isinstance(exc, AuthenticationError):
            headers["WWW-Authenticate"] = "Bearer"

        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.code, exc.message, request),
            headers=headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request payload failed validation.",
                    "request_id": getattr(request.state, "request_id", None),
                    "fields": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()],
                }
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_body("http_error", str(exc.detail), request),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Never leak a traceback or driver message to the client.
        log.exception("unhandled_exception", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content=_body("internal_error", "An unexpected error occurred.", request),
        )
