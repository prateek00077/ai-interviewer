"""Resume upload and parsed-field schemas.

The upload is two calls, not one. ``/presign`` reserves a row and returns a URL;
``/complete`` is the candidate saying "I finished", which the server then
verifies against S3 itself. Nothing here trusts the client's claim about what it
uploaded -- ``declared_size`` exists only to reject an obviously-too-large file
before a URL is issued at all.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.integrations.storage import RESUME_CONTENT_TYPES
from app.models.resume import ResumeStatus


class ResumePresignRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(max_length=120)
    # Advisory. A plain presigned PUT cannot bind a size limit into the
    # signature, so this only saves a round trip; the authoritative check is the
    # HEAD against S3 in /complete.
    declared_size: int | None = Field(default=None, ge=1)

    @field_validator("content_type")
    @classmethod
    def _supported(cls, v: str) -> str:
        # Split on ";" so "application/pdf; charset=binary" is accepted.
        base = v.split(";")[0].strip().lower()
        if base not in RESUME_CONTENT_TYPES:
            supported = ", ".join(sorted(RESUME_CONTENT_TYPES))
            raise ValueError(f"Unsupported content type. Expected one of: {supported}")
        return base

    @field_validator("filename")
    @classmethod
    def _no_path(cls, v: str) -> str:
        # The filename is stored and later shown to a recruiter. It never becomes
        # a path -- the S3 key is server-generated -- but stripping separators
        # keeps a traversal-shaped string out of the UI and out of any future
        # download-as-filename header.
        return v.replace("/", "_").replace("\\", "_").strip()


class ResumePresignResponse(BaseModel):
    resume_id: uuid.UUID
    upload_url: str
    # Must be echoed back as the PUT's Content-Type: the signature covers it, so
    # a different header makes S3 reject the upload.
    content_type: str
    expires_in: int
    max_bytes: int


class ResumeRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    candidate_id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int | None = None
    status: ResumeStatus
    created_at: datetime
    # Parsed fields and the failure reason are recruiter-facing only; see
    # CandidateResumeRead below.
    parsed: dict
    error: str | None = None


class CandidateResumeRead(BaseModel):
    """What the candidate sees about their own upload.

    No ``parsed`` and no ``error``: the extraction is how the interviewer will be
    briefed, and showing a candidate exactly which skills were pulled out invites
    them to re-upload until the summary flatters them. ``error`` is operator
    text and may name internals.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    status: ResumeStatus
    created_at: datetime


class ResumeDownload(BaseModel):
    url: str
    expires_in: int
