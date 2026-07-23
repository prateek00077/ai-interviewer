"""Second-speaker and anomalous-silence detection from the audio stream.

Nearly free, because the ASR model is already doing the work. The hosted
function serves `cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer`, and
sortformer IS speaker diarization -- the transcription results carry speaker
tags whether or not anything reads them. Enabling `speaker_diarization` on the
STT settings (see voice/nvidia/stt.py) turns that into a proctoring signal
without a second model, a second pass, or a second bill.

TWO SPEAKERS IS NOT TWO PEOPLE CHEATING. The interviewer's own voice can bleed
into the candidate's microphone through speakers rather than headphones, which
is why the interviewer's diarized speaker id is excluded before anything is
reported, and why a single tagged utterance is not enough.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.core.events import VoiceSignalObserved, subscribe
from app.models.proctoring import ProctorEventType, ProctorSeverity

log = structlog.get_logger(__name__)

# Below this, a stray tag is more likely diarization noise on a short utterance
# than a real second person. Sortformer will occasionally split one speaker
# across two ids on a cough or a laugh.
MIN_UTTERANCES_FOR_SECOND_SPEAKER = 2

# A candidate silent this long mid-answer is worth noting -- they may be reading
# from something. INFO only: thinking is also a thing people do.
ANOMALOUS_SILENCE_MS = 15_000


@dataclass(frozen=True, slots=True)
class VoiceFinding:
    event_type: ProctorEventType
    severity: ProctorSeverity
    note: str


def speaker_ids(result: object) -> list[int]:
    """Pull diarized speaker tags out of a Riva result.

    Defensive by design: the shape of ``result`` is the vendor's, it varies with
    the model, and proctoring must never be the reason a live interview breaks.
    An unreadable result yields no signal rather than an exception.
    """
    tags: list[int] = []
    try:
        alternatives = getattr(result, "alternatives", None) or []
        for alternative in alternatives:
            for word in getattr(alternative, "words", None) or []:
                speaker = getattr(word, "speaker_tag", None)
                if isinstance(speaker, int) and speaker >= 0:
                    tags.append(speaker)
    except Exception:  # noqa: BLE001 - never break a call over a signal
        log.debug("proctor.diarization_unreadable")
    return tags


def assess_speakers(
    tag_counts: dict[int, int], *, interviewer_tag: int | None = None
) -> VoiceFinding | None:
    """Decide whether the tags indicate a second person in the room.

    ``tag_counts`` maps a diarized speaker id to how many words it produced.
    """
    counts = {tag: n for tag, n in tag_counts.items() if tag != interviewer_tag}
    # The candidate is the dominant remaining speaker; anyone else is the signal.
    others = sorted(counts.values(), reverse=True)[1:]
    if not others:
        return None

    supporting = sum(n for n in others if n >= MIN_UTTERANCES_FOR_SECOND_SPEAKER)
    if not supporting:
        return None

    return VoiceFinding(
        ProctorEventType.SECOND_SPEAKER,
        ProctorSeverity.CRITICAL,
        f"a second voice contributed {supporting} word(s)",
    )


def assess_silence(gap_ms: int) -> VoiceFinding | None:
    """A long mid-answer pause. Informational: thinking looks like this too."""
    if gap_ms < ANOMALOUS_SILENCE_MS:
        return None
    return VoiceFinding(
        ProctorEventType.ANOMALOUS_SILENCE,
        ProctorSeverity.INFO,
        f"{gap_ms // 1000}s of silence mid-answer",
    )


# --- Bus wiring -------------------------------------------------------------


async def _on_voice_signal(event: VoiceSignalObserved) -> None:
    """Turn a raw acoustic observation into a stored proctoring event.

    Subscribed rather than called, so ``voice/`` never imports this module and
    the pipeline can move to its own process. Everything here is best-effort:
    a proctoring signal that fails must not disturb a live interview, and the
    verdict is recomputed from whatever events did land.
    """
    findings: list[VoiceFinding] = []

    speakers = assess_speakers(event.speaker_tag_counts)
    if speakers is not None:
        findings.append(speakers)

    silence = assess_silence(event.silence_gap_ms)
    if silence is not None:
        findings.append(silence)

    if not findings:
        return

    from app.db.session import tenant_session
    from app.modules.proctoring import collector, rules

    try:
        async with tenant_session(event.org_id, "system", None) as session:
            thresholds = rules.Thresholds.from_policy(
                await collector.policy_for_interview(session, event.interview_id)
            )
            counters = collector.SessionCounters()
            await counters.prime(session, event.interview_id)

            for finding in findings:
                await collector.record(
                    session,
                    org_id=event.org_id,
                    interview_id=event.interview_id,
                    event_type=finding.event_type,
                    thresholds=thresholds,
                    counters=counters,
                    payload={"note": finding.note, "source": "audio"},
                    offset_ms=event.offset_ms,
                    # Server-derived: the severity comes from the rules that
                    # detected it, not from the browser, which is the whole
                    # reason a client may not claim these event types.
                    severity=finding.severity,
                )
        log.info(
            "proctor.voice_signals_recorded",
            interview_id=str(event.interview_id),
            findings=[f.event_type.value for f in findings],
        )
    except Exception:  # noqa: BLE001 - never break a call over a signal
        log.warning(
            "proctor.voice_signal_failed",
            interview_id=str(event.interview_id),
            exc_info=True,
        )


def register() -> None:
    """Wire this module to the bus. Called once from the app lifespan.

    UNTIL THIS EXISTED, NOTHING IMPORTED THIS MODULE. The detection logic was
    written and unit-tested in the proctoring slice and then never connected, so
    ``SECOND_SPEAKER`` and ``ANOMALOUS_SILENCE`` could not fire at all -- a
    recruiter's proctoring report showed only what the browser volunteered.
    """
    subscribe(VoiceSignalObserved, _on_voice_signal)
