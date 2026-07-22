"""Interview states and legal transitions.

CREATED -> INVITED -> IN_PROGRESS -> COMPLETED
                          |-> ABANDONED | TERMINATED | EXPIRED

Every status change in the codebase goes through ``transition``. That is the
point of the module: an interview's status decides whether a candidate may
connect, whether a plan may still be edited, and whether scoring should run, so
a stray ``interview.status = ...`` anywhere else is a way for those three to
disagree.

WHY TERMINAL STATES ARE ABSORBING: a completed interview that could be moved
back to IN_PROGRESS would let a candidate rejoin after their answers were
scored. The table below has no outgoing edges from any terminal state, so that
is not a rule someone has to remember -- it is a rule they cannot break.

The four ways an interview ends without completing are kept distinct rather than
collapsed into one FAILED, because they mean different things to a recruiter:
the candidate walked away, we stopped them, the link ran out, or nobody came.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.core.exceptions import ConflictError
from app.models.interview import Interview, InterviewStatus

log = structlog.get_logger(__name__)

S = InterviewStatus

# The whole state machine, in one readable place.
LEGAL: dict[InterviewStatus, frozenset[InterviewStatus]] = {
    S.CREATED: frozenset({S.INVITED, S.EXPIRED, S.ABANDONED}),
    # An invited interview can expire before anyone joins, or be terminated by a
    # recruiter who changed their mind.
    S.INVITED: frozenset({S.IN_PROGRESS, S.EXPIRED, S.ABANDONED, S.TERMINATED}),
    S.IN_PROGRESS: frozenset({S.COMPLETED, S.ABANDONED, S.TERMINATED}),
    # Terminal. No outgoing edges, deliberately.
    S.COMPLETED: frozenset(),
    S.ABANDONED: frozenset(),
    S.TERMINATED: frozenset(),
    S.EXPIRED: frozenset(),
}

TERMINAL: frozenset[InterviewStatus] = frozenset(
    status for status, allowed in LEGAL.items() if not allowed
)

# Which timestamp column each arrival stamps. Kept here rather than at the call
# sites so a transition cannot land without its timestamp.
TIMESTAMP_COLUMN: dict[InterviewStatus, str] = {
    S.IN_PROGRESS: "started_at",
    S.COMPLETED: "completed_at",
    S.ABANDONED: "completed_at",
    S.TERMINATED: "completed_at",
    S.EXPIRED: "completed_at",
}


class IllegalTransitionError(ConflictError):
    code = "illegal_transition"

    def __init__(self, current: InterviewStatus, target: InterviewStatus) -> None:
        super().__init__(
            f"An interview cannot go from {current.value} to {target.value}.",
            current=current.value,
            target=target.value,
        )


def can_transition(current: InterviewStatus, target: InterviewStatus) -> bool:
    return target in LEGAL[current]


def is_terminal(status: InterviewStatus) -> bool:
    return status in TERMINAL


def is_live(status: InterviewStatus) -> bool:
    """Whether a candidate may hold an open voice session right now."""
    return status is S.IN_PROGRESS


def transition(interview: Interview, target: InterviewStatus, *, reason: str | None = None) -> bool:
    """Move an interview to ``target``. Returns whether anything changed.

    A no-op re-entry (already IN_PROGRESS, asked for IN_PROGRESS) returns False
    rather than raising, because the callers that drive this -- a reconnecting
    candidate, a redelivered Celery task, a duplicate WebSocket close -- are all
    at-least-once and should not have to check first.

    An actually illegal move raises. Those are bugs, not races.
    """
    current = interview.status
    if current is target:
        return False
    if not can_transition(current, target):
        raise IllegalTransitionError(current, target)

    interview.status = target
    column = TIMESTAMP_COLUMN.get(target)
    if column is not None and getattr(interview, column) is None:
        # Only if unset: a TERMINATED interview that later expires must keep the
        # moment it actually stopped.
        setattr(interview, column, datetime.now(UTC))

    log.info(
        "interview.transition",
        interview_id=str(interview.id),
        was=current.value,
        now=target.value,
        reason=reason,
    )
    return True
