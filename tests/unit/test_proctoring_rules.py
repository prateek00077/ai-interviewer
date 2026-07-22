"""Severity assignment and verdict aggregation.

These encode judgement calls, not mechanics, so they are worth stating as tests:
what counts as suspicious, what counts as noise, and what a system that will
influence hiring decisions is allowed to conclude on its own.

The bias throughout is toward under-flagging. A false accusation against a
nervous honest candidate is a worse failure than a missed signal, because the
missed signal still reaches a human in the event timeline while the false
accusation arrives pre-labelled.
"""

import pytest

from app.models.proctoring import ProctorEventType, ProctorSeverity, ProctorVerdictKind
from app.modules.proctoring import verdict as verdict_module
from app.modules.proctoring.rules import Thresholds, severity_for, should_terminate
from app.modules.proctoring.vision import FrameAnalysis, findings_for
from app.modules.proctoring.voice_signals import assess_silence, assess_speakers

T = ProctorEventType
S = ProctorSeverity
V = ProctorVerdictKind

DEFAULTS = Thresholds()


def _sev(event_type, prior=0, thresholds=DEFAULTS):
    return severity_for(event_type, prior_count=prior, thresholds=thresholds)


# --- Tab switching: the most common false positive ---------------------------


def test_a_single_tab_switch_is_not_a_signal():
    """People check the time and dismiss notifications. Treating that as
    evidence produces false accusations against honest candidates."""
    assert _sev(T.TAB_BLUR, prior=0) is S.INFO


def test_tab_switches_within_the_limit_stay_informational():
    for prior in range(DEFAULTS.blur_limit):
        assert _sev(T.TAB_BLUR, prior=prior) is S.INFO


def test_exceeding_the_limit_warns_rather_than_flagging():
    """One over the limit is a flaky laptop, not a decision."""
    assert _sev(T.TAB_BLUR, prior=DEFAULTS.blur_limit) is S.WARN


def test_far_exceeding_the_limit_is_critical():
    assert _sev(T.TAB_BLUR, prior=DEFAULTS.blur_limit * 2) is S.CRITICAL


def test_a_stricter_policy_escalates_sooner():
    strict = Thresholds(blur_limit=1)
    assert severity_for(T.TAB_BLUR, prior_count=0, thresholds=strict) is S.INFO
    assert severity_for(T.TAB_BLUR, prior_count=1, thresholds=strict) is S.WARN


# --- Everything else ---------------------------------------------------------


def test_a_second_voice_is_immediately_critical():
    """There is no innocent reading a recruiter should not see for themselves."""
    assert _sev(T.SECOND_SPEAKER) is S.CRITICAL


def test_multiple_faces_is_immediately_critical():
    assert _sev(T.MULTIPLE_FACES) is S.CRITICAL


def test_an_absent_face_warns_rather_than_flagging():
    """Doorbells, water, and badly-aimed webcams all look like this."""
    assert _sev(T.FACE_ABSENT) is S.WARN


def test_fullscreen_exit_is_only_meaningful_when_fullscreen_is_required():
    assert _sev(T.FULLSCREEN_EXIT) is S.INFO
    required = Thresholds(fullscreen_required=True)
    assert severity_for(T.FULLSCREEN_EXIT, prior_count=0, thresholds=required) is S.WARN
    assert severity_for(T.FULLSCREEN_EXIT, prior_count=1, thresholds=required) is S.CRITICAL


def test_paste_respects_the_policy():
    assert _sev(T.PASTE) is S.WARN
    allowed = Thresholds(paste_blocked=False)
    assert severity_for(T.PASTE, prior_count=0, thresholds=allowed) is S.INFO


def test_bookkeeping_events_never_escalate():
    for event_type in (T.TAB_FOCUS, T.WINDOW_RESIZE, T.COPY, T.FACE_FRAME):
        assert _sev(event_type, prior=50) is S.INFO


# --- Auto-termination --------------------------------------------------------


def test_auto_termination_is_off_by_default():
    """Ending a real person's interview on a heuristic is a human's decision,
    and a false termination cannot be undone."""
    assert DEFAULTS.auto_terminate is False
    assert should_terminate(S.CRITICAL, DEFAULTS) is False


def test_auto_termination_requires_critical_even_when_enabled():
    enabled = Thresholds(auto_terminate=True)
    assert should_terminate(S.CRITICAL, enabled) is True
    assert should_terminate(S.WARN, enabled) is False
    assert should_terminate(S.INFO, enabled) is False


# --- Verdict aggregation -----------------------------------------------------


def _assess(counts, severities, **kw):
    return verdict_module.assess(counts, severities, **kw)


def test_no_events_at_all_is_no_data_not_clean():
    """A candidate who disabled JavaScript and one who behaved impeccably both
    produce zero events. Reporting the first as CLEAN would make the most
    deliberate evasion produce the best possible result."""
    result = _assess({}, {})
    assert result.kind is V.NO_DATA
    assert result.reasons


def test_routine_activity_is_clean_and_says_so():
    result = _assess({T.TAB_BLUR: 2, T.TAB_FOCUS: 2}, {S.INFO: 4})
    assert result.kind is V.CLEAN
    assert result.reasons, "a clean verdict with no reasons reads like missing data"


def test_one_warning_is_suspicious_not_flagged():
    result = _assess({T.TAB_BLUR: 4}, {S.INFO: 3, S.WARN: 1})
    assert result.kind is V.SUSPICIOUS


def test_a_cluster_of_warnings_is_flagged():
    """Individually explicable; collectively not."""
    result = _assess({T.TAB_BLUR: 8}, {S.WARN: 4})
    assert result.kind is V.FLAGGED


def test_a_single_critical_flags_immediately():
    result = _assess({T.SECOND_SPEAKER: 1}, {S.CRITICAL: 1})
    assert result.kind is V.FLAGGED


def test_a_verdict_always_carries_its_reasons():
    """A verdict without its reasons is an accusation."""
    for counts, severities in (
        ({}, {}),
        ({T.TAB_BLUR: 2}, {S.INFO: 2}),
        ({T.SECOND_SPEAKER: 1}, {S.CRITICAL: 1}),
        ({T.TAB_BLUR: 8}, {S.WARN: 4}),
    ):
        assert _assess(counts, severities).reasons


def test_reasons_are_human_readable_and_quantified():
    result = _assess({T.TAB_BLUR: 5, T.PASTE: 1}, {S.WARN: 2, S.INFO: 4})
    joined = " ".join(result.reasons)
    assert "left the interview tab 5 time(s)" in joined
    assert "pasted content 1 time(s)" in joined


def test_bookkeeping_events_are_counted_but_not_quoted_as_reasons():
    result = _assess({T.TAB_FOCUS: 9}, {S.INFO: 9})
    assert result.counts["TAB_FOCUS"] == 9
    assert not any("TAB_FOCUS" in r for r in result.reasons)


# --- Vision findings ---------------------------------------------------------


def test_a_normal_frame_produces_no_finding():
    """Recording "one face present" thousands of times would bury what matters."""
    assert findings_for(FrameAnalysis(faces=1)) == []


def test_an_empty_frame_warns():
    findings = findings_for(FrameAnalysis(faces=0))
    assert [f.event_type for f in findings] == [T.FACE_ABSENT]
    assert findings[0].severity is S.WARN


def test_two_faces_is_critical():
    findings = findings_for(FrameAnalysis(faces=2, note="two people visible"))
    assert [f.event_type for f in findings] == [T.MULTIPLE_FACES]
    assert findings[0].severity is S.CRITICAL


def test_looking_away_alone_is_not_a_finding():
    """Candidates look at notes, a second monitor, or out of the window. On its
    own that is not something to put in front of a recruiter as a flag."""
    assert findings_for(FrameAnalysis(faces=1, looking_at_screen=False)) == []


# --- Voice signals -----------------------------------------------------------


def test_one_speaker_produces_no_finding():
    assert assess_speakers({0: 200}) is None


def test_the_interviewer_voice_is_excluded():
    """Its own audio bleeds into the candidate's mic through laptop speakers."""
    assert assess_speakers({0: 200, 1: 150}, interviewer_tag=1) is None


def test_a_persistent_second_voice_is_critical():
    finding = assess_speakers({0: 200, 2: 25}, interviewer_tag=1)
    assert finding is not None
    assert finding.event_type is T.SECOND_SPEAKER
    assert finding.severity is S.CRITICAL


def test_a_single_stray_tag_is_ignored_as_diarization_noise():
    """Sortformer splits one speaker across ids on a cough or a laugh."""
    assert assess_speakers({0: 200, 2: 1}, interviewer_tag=1) is None


def test_unreadable_diarization_yields_nothing_rather_than_raising():
    """Proctoring must never be the reason a live interview breaks."""
    from app.modules.proctoring.voice_signals import speaker_ids

    assert speaker_ids(None) == []
    assert speaker_ids(object()) == []


@pytest.mark.parametrize(
    "gap_ms,expected",
    [(1_000, None), (14_000, None), (20_000, T.ANOMALOUS_SILENCE)],
)
def test_long_silences_are_informational_only(gap_ms, expected):
    finding = assess_silence(gap_ms)
    if expected is None:
        assert finding is None
    else:
        assert finding.event_type is expected
        # Thinking also looks like this.
        assert finding.severity is S.INFO
