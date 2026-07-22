"""Interview orchestration and lifecycle operations.

Only the lookup exists so far. Scheduling, state transitions, transcript
accumulation and the expiry reaper land with the interview slice; this is here
because the question-plan routes need to resolve an interview before hanging a
plan off it, and reaching into the ORM from a router would put tenancy-sensitive
queries outside the module that owns them.
"""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.interview import Interview


async def get_interview(session: AsyncSession, interview_id: uuid.UUID) -> Interview:
    """One interview, or 404.

    The session is org-scoped, so another tenant's interview is simply not
    found -- the caller is never told it exists.
    """
    interview = await session.get(Interview, interview_id)
    if interview is None:
        raise NotFoundError("Interview not found.", interview_id=str(interview_id))
    return interview
