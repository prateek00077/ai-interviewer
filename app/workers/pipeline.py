"""The post-interview chain: everything that happens after the call drops.

    finalize -> correct_transcript -> [measure_signals | analyze_frames]
             -> score_interview -> finalize_verdict

ORDER IS A DEPENDENCY GRAPH, NOT A PREFERENCE.

- ``correct_transcript`` runs before everything downstream of it because the
  scorer quotes the transcript and verifies those quotes against it. Score first
  and every citation is checked against text that is about to change.
- ``measure_signals`` and ``analyze_frames`` are a group: audio analysis and
  webcam vision touch nothing the other reads, and both are slow. Running them
  concurrently takes the chain's wall clock down to the slower of the two
  instead of their sum.
- ``score_interview`` waits on that group only because a chord needs a join
  point. It reads neither result.
- ``finalize_verdict`` runs last because the vision pass writes proctoring
  events, and the verdict is recomputed from the full set.

EVERY LINK IS SIGNATURE-IMMUTABLE. Tasks take ``(org_id, interview_id)`` and
return a dict nobody consumes, rather than piping results into the next link.
Celery chains pass the previous result as the first positional argument, so a
task that accepted one would be undebuggable to re-run by hand and impossible to
retry in isolation. ``.si()`` -- immutable signature -- is what suppresses that,
and it is the reason every step here is independently re-runnable against an
interview id and nothing else.

The chain is fire-and-forget. It is enqueued as the voice session ends, and
nothing waits on it: a recruiter polls the score row, which carries its own
status.
"""

from __future__ import annotations

import uuid

import structlog
from celery import chain, chord, group

from app.workers.tasks import interview_tasks, proctoring_tasks, scoring_tasks

log = structlog.get_logger(__name__)


def build(org_id: uuid.UUID, interview_id: uuid.UUID) -> chain:
    """The chain for one interview, unsent."""
    org, interview = str(org_id), str(interview_id)

    parallel = group(
        scoring_tasks.measure_signals.si(org, interview),
        proctoring_tasks.analyze_frames.si(org, interview),
    )

    return chain(
        interview_tasks.finalize_interview.si(org, interview),
        scoring_tasks.correct_transcript.si(org, interview),
        chord(parallel, scoring_tasks.score_interview.si(org, interview)),
        proctoring_tasks.finalize_verdict.si(org, interview),
    )


def enqueue(org_id: uuid.UUID, interview_id: uuid.UUID) -> str | None:
    """Send the chain. Never raises.

    A broker that is down must not take the voice session's shutdown with it --
    the interview is already over and the transcript is already persisted. The
    work is recoverable by re-enqueuing; a crash in the shutdown path is not.
    """
    try:
        result = build(org_id, interview_id).apply_async()
    except Exception as exc:  # noqa: BLE001
        log.error(
            "pipeline.enqueue_failed",
            interview_id=str(interview_id),
            error=str(exc)[:300],
        )
        return None

    log.info("pipeline.enqueued", interview_id=str(interview_id), task_id=result.id)
    return str(result.id)
