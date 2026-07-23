"""Recruiter and candidate report schemas.

``CandidateFeedbackRead`` has no field that can hold a score, a band, or a
recommendation, and it is built from ``CandidateReport`` -- a table that has no
such column either. Two independent layers saying the same thing, which is what
you want when the failure mode is "the candidate learns they scored 2.1".

The two are NOT related by inheritance. A shared base is the mechanism by which
a field added "for the recruiter view" silently appears in the candidate one.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.report import ReportStatus


class ReportLink(BaseModel):
    """A short-lived download URL for a rendered PDF.

    The S3 key is deliberately absent. A bucket name is guessable and a key is
    structured, so returning both is most of the way to handing out the object.
    """

    download_url: str
    expires_in: int
    generated_at: datetime | None = None


class RecruiterReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    interview_id: uuid.UUID
    status: ReportStatus
    generated_at: datetime | None
    error: str | None


class FeedbackItemRead(BaseModel):
    title: str
    detail: str


class CandidateFeedbackRead(BaseModel):
    """What the candidate is shown. Note what cannot appear here."""

    model_config = ConfigDict(from_attributes=True)

    interview_id: uuid.UUID
    status: ReportStatus
    summary: str | None
    strengths: list[FeedbackItemRead]
    growth_areas: list[FeedbackItemRead]
    generated_at: datetime | None
