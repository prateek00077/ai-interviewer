"""Every ORM model, imported here so the mapper registry is always complete.

SQLAlchemy resolves ``relationship("Candidate")`` by name at mapper-configuration
time, which is the first query -- not at import time. A process that imports only
some model modules therefore fails with "expression 'Candidate' failed to locate
a name" the moment it touches a model whose relationships point at one it never
loaded.

The API process hid this: importing the auth chain happens to pull in every
model. A Celery worker importing only ``app.models.resume`` did not, and died on
its first task. Importing the package is now sufficient, because ``app.models.X``
executes this file first.

Alembic's autogenerate reads the same registry, so a model missing here is also a
model missing from migrations.
"""

from app.db.base import Base
from app.models.interview import (
    Interview,
    InterviewStatus,
    InterviewTurn,
    Invite,
    InviteStatus,
    Speaker,
)
from app.models.job import EmploymentType, Job, JobDescription, JobStatus
from app.models.org import Organization
from app.models.question_plan import (
    PlanGenerationStatus,
    PlannedQuestion,
    PlanStatus,
    QuestionPlan,
    RubricCriterion,
)
from app.models.resume import Resume, ResumeChunk, ResumeStatus
from app.models.user import Candidate, User, UserRole

__all__ = [
    "Base",
    "Candidate",
    "EmploymentType",
    "Interview",
    "InterviewStatus",
    "InterviewTurn",
    "Invite",
    "InviteStatus",
    "Job",
    "JobDescription",
    "JobStatus",
    "Organization",
    "PlanGenerationStatus",
    "PlanStatus",
    "PlannedQuestion",
    "QuestionPlan",
    "RubricCriterion",
    "Resume",
    "Speaker",
    "ResumeChunk",
    "ResumeStatus",
    "User",
    "UserRole",
]
