"""Interview, turn, and invite schemas.

``CandidateInterviewRead`` is the narrow view a candidate gets of their own
interview. It carries no scores, no plan, no proctoring, and no recruiter
identity -- everything the candidate needs to know is what is happening now and
how long they have.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.interview import InterviewStatus, Speaker


class InterviewCreate(BaseModel):
    candidate_id: uuid.UUID
    job_id: uuid.UUID | None = None
    scheduled_at: datetime | None = None


class InterviewRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    org_id: uuid.UUID
    candidate_id: uuid.UUID
    job_id: uuid.UUID | None = None
    status: InterviewStatus
    scheduled_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    created_at: datetime


class CandidateInterviewRead(BaseModel):
    """What the candidate sees about their own interview."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    status: InterviewStatus
    scheduled_at: datetime | None = None
    started_at: datetime | None = None


class TurnRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ordinal: int
    speaker: Speaker
    content: str
    started_offset_ms: int
    ended_offset_ms: int
    question_ordinal: int | None = None
    # False while this is still the live ASR text; true once the offline
    # full-quality pass has corrected it.
    is_final: bool


class TranscriptRead(BaseModel):
    interview_id: uuid.UUID
    status: InterviewStatus
    turns: list[TurnRead] = Field(default_factory=list)


class TerminateRequest(BaseModel):
    reason: str = Field(default="recruiter", max_length=200)
