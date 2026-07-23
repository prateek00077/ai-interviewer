"""Score read schemas. Recruiter-facing only.

There is no candidate-facing score schema here, and that absence is the point.
The candidate report in Phase 8 is feedback and gaps; it does not get a number,
a band, or a recommendation. Building it from a shared serializer is how a score
field ends up in it by accident six months from now, so the two never share one.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from app.models.score import Recommendation, ScoringStatus


class EvidenceRead(BaseModel):
    """One verified quote and where to find it.

    ``offset_ms`` is what makes the evidence checkable: a recruiter who doubts a
    score can jump to that moment in the recording and hear it.
    """

    quote: str
    turn_ordinal: int | None = None
    offset_ms: int | None = None


class CriterionScoreRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ordinal: int
    name: str
    weight: Decimal
    # None means the interview produced no verifiable evidence for this
    # dimension. Distinct from a low score, and shown as such.
    score: Decimal | None
    rationale: str | None
    evidence: list[EvidenceRead]


class ScoreRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    interview_id: uuid.UUID
    status: ScoringStatus
    overall: Decimal | None
    recommendation: Recommendation | None
    # Pitch, pauses, fillers, plus how much of the rubric was actually graded.
    # Observations that sit beside the score; nothing here was multiplied into
    # it. See modules/scoring/confidence for why.
    confidence_signals: dict
    scored_by: str | None
    scored_at: datetime | None
    error: str | None
    criteria: list[CriterionScoreRead]
