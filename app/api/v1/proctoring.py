"""Policy CRUD, event ingest (WS), flag retrieval.

Three audiences, three access levels:

- The WebSocket is candidate-only, authenticated by the interview token, and is
  the one place in the API where the caller is the person being assessed. It is
  written accordingly -- see ``modules/proctoring/collector`` for the reasoning.
- Policy CRUD is recruiter-only. A candidate who could read a policy would know
  the blur limit.
- The report is recruiter-only, and the verdict always travels with its reasons.
"""

from __future__ import annotations

import uuid
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy import select

from app.api.deps import Principal, ScopedSession, get_current_candidate, require_role
from app.api.routing import CommittingRoute
from app.core.config import settings
from app.core.exceptions import NotFoundError
from app.core.security import InterviewClaims, TokenType, decode_token
from app.db.session import tenant_session
from app.integrations import storage
from app.models.interview import Interview
from app.models.proctoring import (
    ProctorEventType,
    ProctoringEvent,
    ProctoringPolicy,
    ProctoringVerdict,
)
from app.models.user import UserRole
from app.modules.interview import service as interview_service
from app.modules.interview import state_machine
from app.modules.jobs import service as jobs_service
from app.modules.proctoring import collector, rules
from app.schemas.proctoring import (
    ClientEvent,
    EventRead,
    FramePresignRequest,
    FramePresignResponse,
    PolicyRead,
    PolicyWrite,
    ProctoringReport,
    VerdictRead,
)

log = structlog.get_logger(__name__)

router = APIRouter(tags=["proctoring"], route_class=CommittingRoute)

Recruiter = Annotated[Principal, Depends(require_role(UserRole.ADMIN, UserRole.RECRUITER))]
CurrentCandidate = Annotated[Principal, Depends(get_current_candidate)]


# --- Recruiter: per-job policy ----------------------------------------------


async def _policy_for_job(db: ScopedSession, job_id: uuid.UUID) -> ProctoringPolicy | None:
    return (
        await db.execute(select(ProctoringPolicy).where(ProctoringPolicy.job_id == job_id))
    ).scalar_one_or_none()


@router.get(
    "/jobs/{job_id}/proctoring-policy",
    response_model=PolicyRead,
    summary="The job's proctoring policy",
)
async def get_policy(job_id: uuid.UUID, db: ScopedSession, _: Recruiter) -> PolicyRead:
    await jobs_service.get_job(db, job_id)
    policy = await _policy_for_job(db, job_id)
    if policy is None:
        raise NotFoundError("No proctoring policy for this job.", job_id=str(job_id))
    return PolicyRead.model_validate(policy)


@router.put(
    "/jobs/{job_id}/proctoring-policy",
    response_model=PolicyRead,
    summary="Create or replace the job's proctoring policy",
)
async def upsert_policy(
    job_id: uuid.UUID, payload: PolicyWrite, db: ScopedSession, principal: Recruiter
) -> PolicyRead:
    await jobs_service.get_job(db, job_id)
    policy = await _policy_for_job(db, job_id)

    if policy is None:
        policy = ProctoringPolicy(org_id=principal.org_id, job_id=job_id)
        db.add(policy)

    for name, value in payload.model_dump().items():
        setattr(policy, name, value)
    await db.flush()
    return PolicyRead.model_validate(policy)


# --- Recruiter: the report --------------------------------------------------


@router.get(
    "/interviews/{interview_id}/proctoring",
    response_model=ProctoringReport,
    summary="Proctoring events and verdict for an interview",
)
async def get_report(
    interview_id: uuid.UUID,
    db: ScopedSession,
    _: Recruiter,
    limit: Annotated[int, Query(ge=1, le=1000)] = 500,
) -> ProctoringReport:
    await interview_service.get_interview(db, interview_id)

    verdict = (
        await db.execute(
            select(ProctoringVerdict).where(ProctoringVerdict.interview_id == interview_id)
        )
    ).scalar_one_or_none()
    events = (
        (
            await db.execute(
                select(ProctoringEvent)
                .where(ProctoringEvent.interview_id == interview_id)
                .order_by(ProctoringEvent.at)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )

    return ProctoringReport(
        verdict=VerdictRead.model_validate(verdict) if verdict else None,
        events=[EventRead.model_validate(e) for e in events],
    )


# --- Candidate: webcam frame upload -----------------------------------------


@router.post(
    "/proctoring/frames/presign",
    response_model=FramePresignResponse,
    summary="Get a URL to upload a webcam still to",
)
async def presign_frame(
    payload: FramePresignRequest, principal: CurrentCandidate
) -> FramePresignResponse:
    """A short-lived PUT URL for one still.

    The key is server-chosen and org-prefixed, exactly as for resumes: a client
    that could name the key could overwrite another interview's evidence.
    """
    if principal.interview_id is None:
        raise NotFoundError("This token is not tied to an interview.")

    extension = payload.content_type.split("/")[-1]
    key = f"{principal.org_id}/{principal.interview_id}/{uuid.uuid4().hex}.{extension}"
    upload = await storage.presign_put(
        bucket=settings.s3_bucket_proctoring,
        key=key,
        content_type=payload.content_type,
        max_bytes=settings.max_proctor_frame_bytes,
    )
    return FramePresignResponse(
        upload_url=upload.url,
        s3_key=key,
        content_type=upload.content_type,
        expires_in=upload.expires_in,
        max_bytes=settings.max_proctor_frame_bytes,
    )


# --- Candidate: the event socket --------------------------------------------


async def _ack(websocket: WebSocket, accepted: int) -> None:
    """Confirm the server has processed one message.

    Delivery confirmation, not a courtesy. Without it the browser has no way to
    know an event landed, and on a flaky network a candidate's reports vanish
    silently -- which looks identical to a candidate who reported nothing.

    It carries only the running accepted count, so a rejected message is
    distinguishable from an accepted one but never says WHY it was rejected.
    The forger already knows what they sent; telling them which rule caught it
    would be a tuning loop.
    """
    await websocket.send_json({"accepted": accepted})


async def _authenticate_socket(websocket: WebSocket) -> InterviewClaims | None:
    """Verify the interview token BEFORE accepting the connection.

    The token arrives as a query parameter because a browser WebSocket cannot
    set an Authorization header. It is the same short-lived, single-interview
    credential used everywhere else.
    """
    try:
        return InterviewClaims.parse(
            decode_token(websocket.query_params.get("token", ""), TokenType.INTERVIEW)
        )
    except Exception:  # noqa: BLE001 - never explain an auth failure to a socket
        return None


@router.websocket("/proctoring/ws")
async def proctoring_socket(websocket: WebSocket) -> None:
    """Ingest browser events for the duration of an interview."""
    claims = await _authenticate_socket(websocket)
    if claims is None:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    org_id, interview_id = claims.org_id, claims.interview_id

    async with tenant_session(org_id, "system", None) as session:
        interview = await session.get(Interview, interview_id)
        if interview is None or state_machine.is_terminal(interview.status):
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        thresholds = rules.Thresholds.from_policy(
            await collector.policy_for_interview(session, interview_id)
        )
        counters = collector.SessionCounters()
        # Primed from earlier connections: otherwise a candidate resets their
        # own escalation by reconnecting, which the multi-use invite makes easy.
        await counters.prime(session, interview_id)

    await websocket.accept()
    limiter = collector.RateLimiter()
    accepted = 0
    rejected = 0

    log.info("proctor.socket_open", interview_id=str(interview_id))
    try:
        while True:
            message = await websocket.receive_json()

            if not limiter.allow():
                rejected += 1
                await _ack(websocket, accepted)
                continue
            try:
                parsed = ClientEvent.model_validate(message)
            except Exception:  # noqa: BLE001 - a forger does not get a tuning loop
                rejected += 1
                await _ack(websocket, accepted)
                continue

            event_type = collector.parse_event_type(parsed.type)
            if event_type is None:
                # Unknown, or one of the server-derived types a client is not
                # permitted to claim. Counted, not explained.
                rejected += 1
                await _ack(websocket, accepted)
                continue

            # The transaction is opened, used and closed here, with no network
            # I/O inside it. Sending the ack while the session was still open
            # held a database transaction across a socket write -- and a client
            # disconnecting mid-write propagated the cancellation straight into
            # the connection teardown.
            async with tenant_session(org_id, "system", None) as session:
                event = await collector.record(
                    session,
                    org_id=org_id,
                    interview_id=interview_id,
                    event_type=event_type,
                    thresholds=thresholds,
                    counters=counters,
                    payload=parsed.payload,
                    offset_ms=parsed.offset_ms,
                    s3_key=(
                        parsed.s3_key if event_type is ProctorEventType.FACE_FRAME else None
                    ),
                )
                terminate = rules.should_terminate(event.severity, thresholds)
                if terminate:
                    # Through the state machine, never a direct status write.
                    await interview_service.terminate(
                        session, interview_id, reason="proctoring"
                    )

            accepted += 1
            await _ack(websocket, accepted)

            if terminate:
                log.warning(
                    "proctor.auto_terminated",
                    interview_id=str(interview_id),
                    trigger=event_type.value,
                )
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001 - a broken socket must not raise into the app
        log.warning("proctor.socket_error", interview_id=str(interview_id), exc_info=True)
    finally:
        log.info(
            "proctor.socket_closed",
            interview_id=str(interview_id),
            accepted=accepted,
            rejected=rejected + limiter.dropped,
        )
