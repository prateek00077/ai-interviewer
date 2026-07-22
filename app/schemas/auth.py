"""Auth request/response schemas.

Validation here is a security boundary, not a convenience. Two things it buys:

- ``EmailStr`` and length bounds keep pathological input away from Argon2, which
  will happily spend real CPU hashing a megabyte-long password.
- No response model in this file carries a password hash. That is enforced by
  construction: the fields simply do not exist.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# Argon2 has no practical input limit, so an upper bound is ours to impose.
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_LENGTH = 128

_SLUG_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"


class RegisterOrgRequest(BaseModel):
    org_name: str = Field(min_length=2, max_length=200)
    slug: str = Field(min_length=2, max_length=63, pattern=_SLUG_PATTERN)
    admin_email: EmailStr
    admin_password: str = Field(min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH)
    admin_full_name: str | None = Field(default=None, max_length=200)

    @field_validator("slug")
    @classmethod
    def _lowercase(cls, v: str) -> str:
        return v.lower()


class LoginRequest(BaseModel):
    email: EmailStr
    # No min_length: a length rule on login would reject a legacy short password
    # and answer "is this even a plausible password" before any lookup happens.
    password: str = Field(max_length=MAX_PASSWORD_LENGTH)


class RefreshRequest(BaseModel):
    # In the body rather than a cookie: API-first, which sidesteps CSRF entirely.
    refresh_token: str = Field(min_length=1)


class LogoutRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RegisterOrgResponse(BaseModel):
    org_id: uuid.UUID
    user_id: uuid.UUID
    tokens: TokenResponse


class CreateInviteRequest(BaseModel):
    candidate_email: EmailStr
    candidate_name: str | None = Field(default=None, max_length=200)
    job_id: uuid.UUID | None = None
    max_redemptions: int | None = Field(default=None, ge=1, le=10)


class InviteResponse(BaseModel):
    invite_id: uuid.UUID
    interview_id: uuid.UUID
    candidate_id: uuid.UUID
    invite_token: str
    expires_at: datetime


class RedeemInviteRequest(BaseModel):
    invite_token: str = Field(min_length=1)


class InterviewTokenResponse(BaseModel):
    interview_token: str
    token_type: str = "bearer"
    expires_in: int
    interview_id: uuid.UUID
    candidate_id: uuid.UUID


class PrincipalResponse(BaseModel):
    """Whoami. Reflects the token, so it needs no database round trip."""

    model_config = ConfigDict(from_attributes=True)

    org_id: uuid.UUID
    actor_kind: str
    actor_id: uuid.UUID
    role: str
