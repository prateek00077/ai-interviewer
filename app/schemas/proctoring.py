"""Proctoring policy, event, and verdict schemas.

The write schemas are narrower than the read schemas on purpose. A client may
say *what* happened; it may not say how serious that is, when it happened, or
which interview it belongs to -- those are the server's to decide, and a field
here would be a field to forge.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.proctoring import ProctorEventType, ProctorSeverity, ProctorVerdictKind


class PolicyWrite(BaseModel):
    camera_enabled: bool = True
    frame_interval_secs: int = Field(default=10, ge=5, le=120)
    blur_limit: int = Field(default=3, ge=0, le=100)
    fullscreen_required: bool = False
    paste_blocked: bool = True
    # Deliberately defaults off. Ending a real person's interview on a
    # heuristic is a decision a human should make.
    auto_terminate: bool = False


class PolicyRead(PolicyWrite):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    job_id: uuid.UUID


class ClientEvent(BaseModel):
    """What a browser may send over the proctoring socket.

    Note the absences: no severity, no timestamp, no interview id. Severity is
    assigned by the rules, the timestamp is the server clock, and the interview
    comes from the token that authenticated the socket.
    """

    type: str = Field(max_length=64)
    # Milliseconds since the candidate's session began. Advisory -- used to line
    # an event up with the recording, never to order or filter events, because
    # a client could send anything.
    offset_ms: int | None = Field(default=None, ge=0)
    payload: dict | None = None
    # Set only for FACE_FRAME: where the browser uploaded the still.
    s3_key: str | None = Field(default=None, max_length=512)


class FramePresignRequest(BaseModel):
    content_type: str = Field(default="image/jpeg", pattern="^image/(jpeg|png|webp)$")


class FramePresignResponse(BaseModel):
    upload_url: str
    s3_key: str
    content_type: str
    expires_in: int
    max_bytes: int


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    event_type: ProctorEventType
    severity: ProctorSeverity
    at: datetime
    offset_ms: int | None = None
    payload: dict
    s3_key: str | None = None


class VerdictRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    interview_id: uuid.UUID
    verdict: ProctorVerdictKind
    # Never rendered without these. A verdict without its reasons is an
    # accusation.
    reasons: list
    counts: dict
    frames_analysed: int
    updated_at: datetime


class ProctoringReport(BaseModel):
    verdict: VerdictRead | None = None
    events: list[EventRead] = Field(default_factory=list)
