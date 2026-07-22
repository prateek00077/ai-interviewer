"""Aggregates all signals into CLEAN | SUSPICIOUS | FLAGGED.

RECOMPUTED, NEVER ACCUMULATED. The verdict is derived from the stored events
every time it runs, so re-running after a rule change gives the answer the
current rules imply rather than a fossil of the rules that happened to be live
during the interview. That also makes it explainable: every verdict carries the
reasons that produced it, and a verdict without its reasons is an accusation.

FLAGGED does not mean "cheated". It means a human should look at this before
deciding, and the reasons tell them where to look.

NO_DATA is a distinct outcome and an important one. A candidate who disabled
JavaScript and one who behaved impeccably both generate zero browser events;
reporting the first as CLEAN would let the most deliberate evasion produce the
best possible result.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import structlog
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.proctoring import (
    ProctorEventType,
    ProctoringEvent,
    ProctoringVerdict,
    ProctorSeverity,
    ProctorVerdictKind,
)

log = structlog.get_logger(__name__)

T = ProctorEventType
S = ProctorSeverity
V = ProctorVerdictKind

# One CRITICAL is enough to warrant a look. These are signals with no innocent
# reading -- a second voice, a second face -- not accumulations of small things.
FLAG_ON_CRITICAL = 1
# Warnings are individually explicable; a cluster of them is not.
FLAG_ON_WARNINGS = 4
SUSPICIOUS_ON_WARNINGS = 1

# Human phrasing per type. Kept here rather than built ad hoc so a reason
# reads the same in the report as it does in a log line.
REASON_TEMPLATES: dict[ProctorEventType, str] = {
    T.TAB_BLUR: "left the interview tab {count} time(s)",
    T.FULLSCREEN_EXIT: "exited fullscreen {count} time(s)",
    T.PASTE: "pasted content {count} time(s)",
    T.DEVTOOLS_OPEN: "opened developer tools {count} time(s)",
    T.MULTIPLE_FACES: "more than one face visible in {count} frame(s)",
    T.FACE_ABSENT: "no face visible in {count} frame(s)",
    T.SECOND_SPEAKER: "a second speaker was detected {count} time(s)",
    T.ANOMALOUS_SILENCE: "unusually long silences occurred {count} time(s)",
}

# Types that say nothing on their own. Counted, never quoted as a reason.
BOOKKEEPING: frozenset[ProctorEventType] = frozenset(
    {T.TAB_FOCUS, T.WINDOW_RESIZE, T.COPY, T.FACE_FRAME}
)


@dataclass(frozen=True, slots=True)
class Assessment:
    kind: ProctorVerdictKind
    reasons: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    frames_analysed: int = 0


def assess(
    counts_by_type: dict[ProctorEventType, int],
    severity_counts: dict[ProctorSeverity, int],
    *,
    frames_analysed: int = 0,
    had_connection: bool = True,
) -> Assessment:
    """Pure function: counts in, verdict out.

    Separated from the database so the thresholds can be reasoned about and
    tested without seeding an interview.
    """
    counts = {t.value: c for t, c in counts_by_type.items() if c}
    critical = severity_counts.get(S.CRITICAL, 0)
    warnings = severity_counts.get(S.WARN, 0)

    reported = sum(c for t, c in counts_by_type.items() if t not in BOOKKEEPING)
    if not counts_by_type and not frames_analysed:
        # Nothing at all arrived. If a session existed, that is itself notable:
        # a working client reports at least focus changes over 30 minutes.
        return Assessment(
            kind=V.NO_DATA,
            reasons=(
                ["no proctoring signals were received from the candidate's browser"]
                if had_connection
                else ["no proctoring session was established"]
            ),
            counts=counts,
            frames_analysed=frames_analysed,
        )

    reasons = [
        REASON_TEMPLATES[t].format(count=c)
        for t, c in sorted(counts_by_type.items(), key=lambda kv: -kv[1])
        if c and t in REASON_TEMPLATES
    ]

    if critical >= FLAG_ON_CRITICAL or warnings >= FLAG_ON_WARNINGS:
        kind = V.FLAGGED
    elif warnings >= SUSPICIOUS_ON_WARNINGS:
        kind = V.SUSPICIOUS
    else:
        kind = V.CLEAN
        # A clean interview with real activity should say so, rather than
        # presenting an empty reason list that reads like missing data.
        reasons = reasons or [f"{reported} routine event(s), nothing notable"]

    return Assessment(
        kind=kind, reasons=reasons, counts=counts, frames_analysed=frames_analysed
    )


async def compute(session: AsyncSession, interview_id: uuid.UUID) -> Assessment:
    """Assess one interview from its stored events."""
    rows = (
        await session.execute(
            select(ProctoringEvent.event_type, ProctoringEvent.severity, func.count())
            .where(ProctoringEvent.interview_id == interview_id)
            .group_by(ProctoringEvent.event_type, ProctoringEvent.severity)
        )
    ).all()

    counts_by_type: dict[ProctorEventType, int] = {}
    severity_counts: dict[ProctorSeverity, int] = {}
    for event_type, severity, count in rows:
        counts_by_type[event_type] = counts_by_type.get(event_type, 0) + int(count)
        severity_counts[severity] = severity_counts.get(severity, 0) + int(count)

    frames = counts_by_type.get(T.FACE_FRAME, 0)
    return assess(counts_by_type, severity_counts, frames_analysed=frames)


async def store(
    session: AsyncSession,
    *,
    org_id: uuid.UUID,
    interview_id: uuid.UUID,
    assessment: Assessment,
) -> ProctoringVerdict:
    """Upsert the verdict. Idempotent, because it is recomputed not accumulated."""
    existing = (
        await session.execute(
            select(ProctoringVerdict).where(ProctoringVerdict.interview_id == interview_id)
        )
    ).scalar_one_or_none()

    if existing is None:
        existing = ProctoringVerdict(org_id=org_id, interview_id=interview_id)
        session.add(existing)

    existing.verdict = assessment.kind
    existing.reasons = assessment.reasons
    existing.counts = assessment.counts
    existing.frames_analysed = assessment.frames_analysed
    await session.flush()

    log.info(
        "proctor.verdict",
        interview_id=str(interview_id),
        verdict=assessment.kind.value,
        reasons=len(assessment.reasons),
    )
    return existing


async def finalise(
    session: AsyncSession, *, org_id: uuid.UUID, interview_id: uuid.UUID
) -> ProctoringVerdict:
    return await store(
        session,
        org_id=org_id,
        interview_id=interview_id,
        assessment=await compute(session, interview_id),
    )
