"""User and candidate schemas.

``UserRead`` has no ``hashed_password`` field, and that omission is the control:
with ``from_attributes=True`` pydantic copies only declared fields, so a hash
cannot reach a response even if someone returns an ORM object directly.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models.user import UserRole


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    email: EmailStr
    full_name: str | None = None
    role: UserRole
    is_active: bool
    last_login_at: datetime | None = None
    created_at: datetime


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    full_name: str | None = Field(default=None, max_length=200)
    role: UserRole = UserRole.RECRUITER


class UserUpdate(BaseModel):
    """Admin-editable fields.

    Note what is absent: ``email`` and ``password``. Changing either is an
    identity operation that belongs behind re-authentication, not behind a
    general-purpose PATCH -- and ``org_id`` is absent because moving a user
    between tenants is not an edit, it is a security event.
    """

    full_name: str | None = Field(default=None, max_length=200)
    role: UserRole | None = None
    is_active: bool | None = None


class CandidateCreate(BaseModel):
    email: EmailStr
    full_name: str | None = Field(default=None, max_length=200)
    phone: str | None = Field(default=None, max_length=50)
    # The recruiter's own ATS identifier, so imported candidates can be matched back.
    external_ref: str | None = Field(default=None, max_length=200)


class CandidateUpdate(BaseModel):
    full_name: str | None = Field(default=None, max_length=200)
    phone: str | None = Field(default=None, max_length=50)
    external_ref: str | None = Field(default=None, max_length=200)


class CandidateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    email: EmailStr
    full_name: str | None = None
    phone: str | None = None
    external_ref: str | None = None
    created_at: datetime
