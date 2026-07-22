"""Pipecat Smart Turn / Silero VAD configuration (stop_secs=0.2).

Turn detection is the single most felt setting in the product. Too eager and the
interviewer cuts off a candidate who paused to think, which reads as rude and
loses the answer. Too patient and every exchange carries a beat of dead air,
which reads as slow.

0.2 seconds is deliberately short, and it works because it is not the only
signal: the ASR's own end-of-utterance logic (stop_history in nvidia/stt.py)
runs alongside it, so VAD proposes and the ASR confirms. A single 0.2s gap in
speech does not end the turn on its own.

Silero runs on onnxruntime, on the CPU, in about a millisecond per frame -- it
is not a meaningful part of the turn budget.
"""

from __future__ import annotations

import structlog
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor

from app.core.config import settings

log = structlog.get_logger(__name__)

# How confident Silero must be that this frame is speech. Lower would let a
# cough or a keyboard start a turn.
CONFIDENCE = 0.7
# Speech must persist this long before a turn starts, which rejects clicks.
START_SECS = 0.2
# Below this the frame is treated as silence regardless of Silero's opinion.
# Guards against a hissy microphone reading as continuous speech.
MIN_VOLUME = 0.6


def params() -> VADParams:
    return VADParams(
        confidence=CONFIDENCE,
        start_secs=START_SECS,
        stop_secs=settings.vad_stop_secs,
        min_volume=MIN_VOLUME,
    )


def analyzer() -> SileroVADAnalyzer:
    return SileroVADAnalyzer(params=params())


def build() -> VADProcessor:
    """The VAD processor for the pipeline.

    In pipecat 1.3 VAD is a pipeline processor rather than a transport
    parameter, so it sits explicitly between the transport input and the STT
    service where its ordering is visible.
    """
    log.info("voice.vad_configured", stop_secs=settings.vad_stop_secs)
    return VADProcessor(vad_analyzer=analyzer())
