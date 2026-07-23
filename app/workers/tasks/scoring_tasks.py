"""The offline scoring steps: transcript correction, signals, rubric scoring.

Three tasks rather than one, because they fail for different reasons and should
be retried independently. Re-transcription fails when ASR is down; scoring fails
when the LLM is down; neither should force the other to run again. They are
separate links in the chain in ``workers/pipeline.py``.

All three are idempotent on ``interview_id``:

- ``correct_transcript`` marks the turns it has processed final and skips them.
- ``measure_signals`` overwrites the same JSONB field with the same measurement.
- ``score_interview`` replaces the score's contents rather than appending.

Model calls happen OUTSIDE the database transaction throughout. A rubric of six
criteria is six sequential LLM calls; holding a pooled connection across them
would starve the API for a minute at a time.
"""

from __future__ import annotations

import uuid

import structlog
from celery import shared_task

from app.core.config import settings
from app.db.session import tenant_session
from app.integrations import storage
from app.modules.interview import service as interview_service
from app.modules.interview import transcript
from app.modules.question_plan import service as plan_service
from app.modules.scoring import aggregator, confidence, rubric_scorer, transcript_pass
from app.modules.scoring import service as scoring_service
from app.workers.celery_app import run_async

log = structlog.get_logger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_SECS = 30


# --- Transcript correction --------------------------------------------------


@shared_task(
    bind=True,
    name="scoring.correct_transcript",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def correct_transcript(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    return run_async(_correct(uuid.UUID(org_id), uuid.UUID(interview_id)))


async def _correct(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)
        recording_key = interview.recording_key

    # The pass reads S3 and streams gRPC; it opens its own short transaction to
    # write once it has the words.
    async with tenant_session(org_id, "system", None) as session:
        result = await transcript_pass.apply(
            session, interview_id=interview_id, recording_key=recording_key
        )
    return {"interview_id": str(interview_id), **result}


# --- Delivery signals -------------------------------------------------------


@shared_task(
    bind=True,
    name="scoring.measure_signals",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def measure_signals(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    """Pitch, pauses and fillers. These are reported, never scored."""
    return run_async(_measure(uuid.UUID(org_id), uuid.UUID(interview_id)))


async def _measure(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        interview = await interview_service.get_interview(session, interview_id)
        recording_key = interview.recording_key
        turns = await transcript.list_turns(session, interview_id)
        # Detached from the session before the download so the audio fetch does
        # not happen with a transaction open.
        session.expunge_all()

    recording: bytes | None = None
    if recording_key:
        try:
            recording = await storage.get_bytes(
                bucket=settings.s3_bucket_recordings, key=recording_key
            )
        except Exception as exc:  # noqa: BLE001 - signals degrade, they do not fail
            log.warning("scoring.recording_fetch_failed", error=str(exc)[:200])

    signals = confidence.measure(turns, recording)

    async with tenant_session(org_id, "system", None) as session:
        score = await scoring_service.ensure_score(
            session, org_id=org_id, interview_id=interview_id
        )
        # Merged, not replaced: the rubric-coverage keys the scoring step writes
        # into the same field must survive whichever step runs last.
        score.confidence_signals = {**score.confidence_signals, **signals.as_dict()}

    return {"interview_id": str(interview_id), **signals.as_dict()}


# --- Rubric scoring ---------------------------------------------------------


@shared_task(
    bind=True,
    name="scoring.score_interview",
    max_retries=MAX_RETRIES,
    autoretry_for=(Exception,),
    retry_backoff=RETRY_BACKOFF_SECS,
    retry_jitter=True,
)
def score_interview(self, org_id: str, interview_id: str) -> dict:  # type: ignore[no-untyped-def]
    return run_async(_score(uuid.UUID(org_id), uuid.UUID(interview_id)))


async def _score(org_id: uuid.UUID, interview_id: uuid.UUID) -> dict:
    async with tenant_session(org_id, "system", None) as session:
        plan = await plan_service.get_for_interview(session, interview_id)
        if plan is None or not plan.criteria:
            # No rubric means nothing to score against. Not an error: an
            # interview whose plan generation failed still ended, and the report
            # should say "not assessed" rather than the job retrying forever.
            log.warning("scoring.no_rubric", interview_id=str(interview_id))
            return {"interview_id": str(interview_id), "skipped": "no rubric"}

        score = await scoring_service.ensure_score(
            session, org_id=org_id, interview_id=interview_id, plan_id=plan.id
        )
        await scoring_service.mark_scoring(session, score)
        plan_id = plan.id

        turns = await transcript.list_turns(session, interview_id)
        criteria = list(plan.criteria)
        # Everything below runs without a transaction, so both collections are
        # detached first: a lazy load out there is a MissingGreenlet, not a query.
        session.expunge_all()

    # A transcript with nothing in it is stored as INSUFFICIENT_EVIDENCE rather
    # than skipped. A recruiter opening the report needs to see that the
    # interview produced no usable audio -- an absent score row looks like a job
    # that has not run yet.
    if any(t.content.strip() for t in turns):
        results, model_name = await rubric_scorer.score_all(criteria, turns)
        outcome = aggregator.aggregate(scoring_service.weights_and_scores(results))
    else:
        log.warning("scoring.empty_transcript", interview_id=str(interview_id))
        results, model_name = [], ""
        outcome = aggregator.aggregate([])

    async with tenant_session(org_id, "system", None) as session:
        stored = await scoring_service.require_for_interview(session, interview_id)
        stored.plan_id = plan_id
        await scoring_service.store(
            session,
            score=stored,
            results=results,
            outcome=outcome,
            signals=stored.confidence_signals,
            model_name=model_name,
        )

    return {
        "interview_id": str(interview_id),
        "overall": str(outcome.overall) if outcome.overall is not None else None,
        "recommendation": outcome.recommendation.value,
        "coverage": str(outcome.coverage),
        "graded": outcome.graded_count,
        "criteria": outcome.total_count,
    }
