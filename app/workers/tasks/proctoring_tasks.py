"""Offline vision analysis and final proctoring verdict.

Frames are analysed after the interview, never during it. A VLM call takes
around 1.6 seconds; on a 1.5-second turn budget that is not a latency problem,
it is a broken conversation. And there is nothing a live analysis could do
differently, because the product does not interrupt a real person on a
machine's say-so.

Both tasks are idempotent on ``interview_id``. Frames already analysed are
skipped, and the verdict is recomputed from scratch rather than accumulated, so
a duplicate delivery produces the same answer instead of a doubled one.
"""

from __future__ import annotations

import uuid

import structlog
from celery import shared_task
from sqlalchemy import select

from app.core.config import settings
from app.db.session import tenant_session
from app.integrations import storage
from app.models.proctoring import ProctorEventType, ProctoringEvent
from app.modules.proctoring import verdict as verdict_module
from app.modules.proctoring import vision
from app.workers.celery_app import run_async

log = structlog.get_logger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECS = 30

# A 45-minute interview at one frame per 10 seconds is 270 stills. Analysing
# every one costs ~7 minutes of VLM time for a signal that is obvious from a
# sample, so frames are taken evenly across the interview rather than
# exhaustively -- and the count analysed is recorded on the verdict so a
# reviewer knows how much was looked at.
MAX_FRAMES_ANALYSED = 40

# Marks a frame as processed so a retry does not pay for it twice.
ANALYSED_FLAG = "analysed"


@shared_task(
    bind=True,
    name="proctoring.analyze_frames",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def analyze_frames(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    return run_async(_analyze(uuid.UUID(org_id), uuid.UUID(interview_id)))


def _sample(rows: list, limit: int) -> list:
    """Evenly spaced frames across the interview, not the first N.

    The first N would analyse the opening minutes and ignore everything after,
    which is exactly backwards -- the interesting part of an interview is the
    hard questions in the middle.
    """
    if len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)]


async def _analyze(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        rows = (
            (
                await session.execute(
                    select(ProctoringEvent)
                    .where(
                        ProctoringEvent.interview_id == interview_id,
                        ProctoringEvent.event_type == ProctorEventType.FACE_FRAME,
                        ProctoringEvent.s3_key.is_not(None),
                    )
                    .order_by(ProctoringEvent.at)
                )
            )
            .scalars()
            .all()
        )
        pending = [r for r in rows if not r.payload.get(ANALYSED_FLAG)]
        selected = [(r.id, r.s3_key, r.offset_ms) for r in _sample(pending, MAX_FRAMES_ANALYSED)]

    if not selected:
        return {"interview_id": str(interview_id), "analysed": 0, "findings": 0}

    analysed = 0
    findings_written = 0

    for event_id, s3_key, offset_ms in selected:
        try:
            image = await storage.get_bytes(
                bucket=settings.s3_bucket_proctoring, key=s3_key
            )
            analysis = await vision.analyse_frame(image)
        except Exception as exc:  # noqa: BLE001
            # One unreadable frame must not sink the batch. The verdict records
            # how many were actually analysed, so a partial pass is visible
            # rather than silently indistinguishable from a clean one.
            log.warning(
                "proctor.frame_failed", event_id=str(event_id), error=str(exc)[:200]
            )
            continue

        analysed += 1
        async with tenant_session(org_id, "system", None) as session:
            source = await session.get(ProctoringEvent, event_id)
            if source is not None:
                # Reassigned rather than mutated: SQLAlchemy does not track
                # in-place changes to a JSONB dict, and the flag would never be
                # written -- making every retry re-analyse every frame.
                source.payload = {**source.payload, ANALYSED_FLAG: True}

            for finding in vision.findings_for(analysis):
                session.add(
                    ProctoringEvent(
                        org_id=org_id,
                        interview_id=interview_id,
                        event_type=finding.event_type,
                        severity=finding.severity,
                        offset_ms=offset_ms,
                        payload={"note": finding.note, "source_event": str(event_id)},
                        s3_key=s3_key,
                    )
                )
                findings_written += 1

    log.info(
        "proctor.frames_analysed",
        interview_id=str(interview_id),
        analysed=analysed,
        findings=findings_written,
    )
    return {
        "interview_id": str(interview_id),
        "analysed": analysed,
        "findings": findings_written,
    }


@shared_task(
    bind=True,
    name="proctoring.finalize_verdict",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
)
def finalize_verdict(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    """Recompute and store the verdict. Runs after the vision pass."""
    return run_async(_finalize(uuid.UUID(org_id), uuid.UUID(interview_id)))


async def _finalize(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        stored = await verdict_module.finalise(
            session, org_id=org_id, interview_id=interview_id
        )
        return {
            "interview_id": str(interview_id),
            "verdict": stored.verdict.value,
            "reasons": list(stored.reasons),
        }
