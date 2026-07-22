"""Evaluates events against the interview's ProctoringPolicy thresholds.

SEVERITY IS ASSIGNED HERE, NEVER ACCEPTED FROM THE CLIENT. The browser reports
what happened; this module decides what it is worth. A candidate who could send
their own severity would be grading their own conduct.

The severities themselves are deliberately conservative:

- One tab switch is INFO. People check the time, dismiss a notification, or get
  a calendar popup. Treating that as evidence of cheating produces false
  accusations against nervous honest candidates, which is a worse failure than
  missing a real one.
- Repetition is what escalates. Leaving the tab once is noise; leaving it eight
  times during the technical questions is a pattern.
- A second voice in the room is CRITICAL immediately, because there is no
  innocent reading of it that the recruiter should not see.

Nothing here decides an outcome. It labels evidence for a human.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.core.config import settings
from app.models.proctoring import ProctorEventType, ProctoringPolicy, ProctorSeverity

log = structlog.get_logger(__name__)

T = ProctorEventType
S = ProctorSeverity


@dataclass(frozen=True, slots=True)
class Thresholds:
    """A policy flattened to what the rules need.

    Exists so the rules can be evaluated without a database row -- an interview
    whose job has no policy still needs proctoring, using the org defaults.
    """

    blur_limit: int = 3
    fullscreen_required: bool = False
    paste_blocked: bool = True
    camera_enabled: bool = True
    frame_interval_secs: int = 10
    auto_terminate: bool = False

    @classmethod
    def from_policy(cls, policy: ProctoringPolicy | None) -> Thresholds:
        if policy is None:
            return cls(
                blur_limit=settings.proctor_blur_limit,
                camera_enabled=settings.proctor_camera_enabled,
                frame_interval_secs=settings.proctor_frame_interval_secs,
                auto_terminate=settings.proctor_auto_terminate,
            )
        return cls(
            blur_limit=policy.blur_limit,
            fullscreen_required=policy.fullscreen_required,
            paste_blocked=policy.paste_blocked,
            camera_enabled=policy.camera_enabled,
            frame_interval_secs=policy.frame_interval_secs,
            auto_terminate=policy.auto_terminate,
        )


# Severity for an event type regardless of how often it has happened. Types
# absent from this map are scored by _escalating below.
BASE_SEVERITY: dict[ProctorEventType, ProctorSeverity] = {
    # Derived server-side from audio or vision, not reported by the browser.
    T.SECOND_SPEAKER: S.CRITICAL,
    T.MULTIPLE_FACES: S.CRITICAL,
    T.FACE_ABSENT: S.WARN,
    T.ANOMALOUS_SILENCE: S.INFO,
    # A developer console open during a technical interview has one obvious
    # reading, and unlike a tab switch it takes deliberate effort.
    T.DEVTOOLS_OPEN: S.WARN,
    # Pure bookkeeping: the pair of a blur, and useful only for durations.
    T.TAB_FOCUS: S.INFO,
    T.WINDOW_RESIZE: S.INFO,
    T.COPY: S.INFO,
    # The upload itself is not a signal; what the vision pass finds in it is.
    T.FACE_FRAME: S.INFO,
}


def severity_for(
    event_type: ProctorEventType, *, prior_count: int, thresholds: Thresholds
) -> ProctorSeverity:
    """How serious this event is, given how often it has already happened.

    ``prior_count`` is occurrences of this same type earlier in the interview,
    which is what turns a one-off into a pattern.
    """
    fixed = BASE_SEVERITY.get(event_type)
    if fixed is not None:
        return fixed

    occurrence = prior_count + 1

    if event_type is T.TAB_BLUR:
        if occurrence <= thresholds.blur_limit:
            return S.INFO
        # Well past the limit rather than one over it: a candidate on a flaky
        # laptop can rack up blurs without ever leaving the interview.
        return S.CRITICAL if occurrence > thresholds.blur_limit * 2 else S.WARN

    if event_type is T.FULLSCREEN_EXIT:
        if not thresholds.fullscreen_required:
            return S.INFO
        return S.WARN if occurrence == 1 else S.CRITICAL

    if event_type is T.PASTE:
        # Pasting into a spoken interview means text arrived from somewhere.
        return S.WARN if thresholds.paste_blocked else S.INFO

    return S.INFO


def should_terminate(severity: ProctorSeverity, thresholds: Thresholds) -> bool:
    """Whether this event ends the interview.

    Off unless a recruiter explicitly enabled it, and only on CRITICAL. Ending a
    real person's interview on a heuristic is a decision a human should make --
    the evidence reaches the report either way, and a false termination cannot
    be undone.
    """
    return thresholds.auto_terminate and severity is S.CRITICAL
