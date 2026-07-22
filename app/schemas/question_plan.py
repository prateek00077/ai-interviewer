"""Question plan and rubric schemas.

Every schema here is recruiter-facing. There is no candidate-facing variant on
purpose: this is the answer key, and the closest a candidate ever comes to it is
hearing a question asked out loud.
"""

import uuid
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.models.question_plan import PlanGenerationStatus, PlanStatus


class RubricCriterionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ordinal: int
    name: str
    description: str | None = None
    weight: Decimal
    descriptors: dict


class PlannedQuestionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ordinal: int
    body: str
    competency: str | None = None
    follow_up_hints: list
    time_budget_secs: int


class QuestionPlanRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    interview_id: uuid.UUID
    job_description_id: uuid.UUID | None = None
    resume_id: uuid.UUID | None = None
    status: PlanStatus
    generation_status: PlanGenerationStatus
    version: int
    generated_by: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    questions: list[PlannedQuestionRead] = Field(default_factory=list)
    criteria: list[RubricCriterionRead] = Field(default_factory=list)


class GeneratePlanRequest(BaseModel):
    question_count: int = Field(default=8, ge=3, le=25)
    duration_minutes: int = Field(default=30, ge=10, le=120)


class QuestionWrite(BaseModel):
    body: str = Field(min_length=10, max_length=2000)
    # Must name an existing criterion; the service rejects anything else rather
    # than silently storing a dangling tag.
    competency: str | None = Field(default=None, max_length=120)
    follow_up_hints: list[str] = Field(default_factory=list, max_length=6)
    time_budget_secs: int = Field(default=180, ge=30, le=1800)


class QuestionsReplace(BaseModel):
    """Wholesale replacement: reorder, delete and add are one intent."""

    questions: list[QuestionWrite] = Field(min_length=1, max_length=40)
    # The version the client read. Omit to skip the concurrency check; send it
    # and a competing edit becomes a 409 rather than a silent overwrite.
    expected_version: int | None = None


class CriterionWrite(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    weight: Decimal = Field(gt=0, le=1)
    descriptors: dict[str, str] = Field(default_factory=dict)


class CriteriaReplace(BaseModel):
    criteria: list[CriterionWrite] = Field(min_length=1, max_length=10)
    expected_version: int | None = None


class ApproveRequest(BaseModel):
    expected_version: int | None = None
