"""The head of the post-interview chain, and the expiry reaper.

``finalize_interview`` does almost nothing on purpose. Its whole job is to be a
cheap, idempotent gate the rest of the chain hangs off: it confirms the
interview really has ended and creates the PENDING score row, so a recruiter
refreshing the page sees "scoring in progress" within a second of the call
dropping rather than a 404 for however long the transcription takes.

Anything expensive belongs in the links after it, which retry independently.
"""

from __future__ import annotations

import uuid

import structlog
from celery import shared_task

from app.db.session import tenant_session
from app.modules.interview import service as interview_service
from app.modules.interview import state_machine
from app.modules.scoring import service as scoring_service
from app.workers.celery_app import run_async

log = structlog.get_logger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF_SECS = 15


@shared_task(
    bind=True,
    name="interview.finalize",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def finalize_interview(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    return run_async(_finalize(uuid.UUID(org_id), uuid.UUID(interview_id)))


async def _finalize(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)

        if not state_machine.is_terminal(interview.status):
            # The chain is enqueued from the session-ended handler, which runs
            # off a fire-and-forget bus -- so the task can win the race against
            # the transition it depends on. Retrying is right: this resolves in
            # milliseconds and the alternative is scoring a live interview.
            log.info(
                "interview.finalize_not_ready",
                interview_id=str(interview_id),
                status=interview.status.value,
            )
            raise RuntimeError(f"interview is still {interview.status.value}")

        await scoring_service.ensure_score(
            session, org_id=org_id, interview_id=interview_id
        )
        return {
            "interview_id": str(interview_id),
            "status": interview.status.value,
            "recording_key": interview.recording_key,
        }


@shared_task(bind=True, name="interview.expire_stale")
def expire_stale_interviews(self, org_id: str) -> dict:  # type: ignore[no-untyped-def]
    """Expire interviews nobody ever joined. Scheduled, one org per call.

    Per-org rather than global because the reaper runs under a tenant session
    like everything else -- there is no cross-org context in this codebase and
    adding one for a housekeeping job would be the wrong place to start.
    """
    return run_async(_expire(uuid.UUID(org_id)))


async def _expire(org_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        count = await interview_service.expire_stale(session)
    return {"org_id": str(org_id), "expired": count}
