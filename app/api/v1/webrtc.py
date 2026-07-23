"""WebRTC offer/answer signaling; issues ephemeral session tokens.

Candidate-only, and there is no interview id in the path. The interview comes
from the token's own signed claim, so a candidate can only ever open a session
for the interview they were invited to -- there is nowhere to put someone
else's id.

The ephemeral token the architecture calls for is the existing 10-minute
interview JWT. A second token type would add a rotation path and another thing
to revoke without buying anything: the interview token is already short-lived,
already scoped to one interview, and already the credential this route
authenticates.
"""

from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, Request

from app.api import ops
from app.api.deps import Principal, ScopedSession, get_current_candidate, get_redis
from app.api.routing import CommittingRoute
from app.core.exceptions import ConflictError, NotFoundError
from app.modules.interview import service as interview_service
from app.modules.interview import state_machine
from app.modules.voice import session_manager, transport
from app.schemas.webrtc import WebRTCAnswer, WebRTCOffer

router = APIRouter(prefix="/webrtc", tags=["webrtc"], route_class=CommittingRoute)

CurrentCandidate = Annotated[Principal, Depends(get_current_candidate)]


@router.post("/offer", response_model=WebRTCAnswer, summary="Open a voice session")
async def offer(
    payload: WebRTCOffer,
    request: Request,
    db: ScopedSession,
    principal: CurrentCandidate,
) -> WebRTCAnswer:
    """Answer an SDP offer and start the interview pipeline.

    Two states may connect: INVITED (first join) and IN_PROGRESS (reconnect
    after a crash, which is why the invite is multi-use). Anything terminal is
    refused -- the interview is over and its transcript is the record.
    """
    if principal.interview_id is None:
        raise NotFoundError("This token is not tied to an interview.")

    if ops.is_draining():
        # Refused at the door during shutdown. Building a pipeline against a
        # Redis pool that is seconds from closing would fail a minute later,
        # mid-answer, which is far worse for the candidate than a clean refusal
        # they can retry against the next instance.
        raise ConflictError(
            "This server is shutting down. Please rejoin with the same link.",
            retryable=True,
        )

    interview = await interview_service.get_interview(db, principal.interview_id)
    if state_machine.is_terminal(interview.status):
        raise ConflictError(
            "This interview has already ended.", current_status=interview.status.value
        )

    connection = transport.create_connection()
    await connection.initialize(sdp=payload.sdp, type=payload.type)

    # The session is created and left running; the request returns as soon as
    # the answer is ready. Everything after this happens on the connection.
    await session_manager.start(
        org_id=principal.org_id,
        interview_id=interview.id,
        candidate_id=principal.actor_id,
        connection=connection,
        redis=get_redis(request),
    )

    answer = connection.get_answer()
    structlog.get_logger(__name__).info(
        "webrtc.answered", interview_id=str(interview.id), pc_id=answer.get("pc_id")
    )
    return WebRTCAnswer(sdp=answer["sdp"], type=answer["type"], pc_id=answer.get("pc_id"))
