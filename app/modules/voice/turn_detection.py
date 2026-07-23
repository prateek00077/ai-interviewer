"""Silero VAD configuration: when the candidate has stopped talking.

Turn detection is the single most felt setting in the product. Too eager and the
interviewer cuts off a candidate who paused to think, which reads as rude and
loses the answer. Too patient and every exchange carries a beat of dead air,
which reads as slow.

IT WAS 0.2 SECONDS AND THAT WAS WRONG. The theory was that VAD proposes and the
ASR confirms, so a short gap is safe. In practice a candidate thinking mid-answer
-- "we sharded on... tenant id" -- had that pause read as the end of their turn,
and the interviewer answered half a sentence. An interview is not a chat
assistant: the whole point is that people stop to think before saying something
considered, and the setting has to make room for that.

0.8s is pipecat's own default. The cost is a beat of dead air; the cost of the
old value was the answer.

This is THE setting for end-of-turn. The ASR has its own stop_history, but that
is measured in frames and is already ~3s -- far more patient than this -- so it
is not what cuts a candidate off. Leave it alone and tune here.

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
# Guards against a hissy microphone reading as continuous speech -- but set too
# high it does the opposite, and a softly-spoken candidate on a laptop mic is
# never heard at all. Configurable for exactly that reason.


def params() -> VADParams:
    return VADParams(
        confidence=CONFIDENCE,
        start_secs=START_SECS,
        stop_secs=settings.vad_stop_secs,
        min_volume=settings.vad_min_volume,
    )


def analyzer() -> SileroVADAnalyzer:
    return SileroVADAnalyzer(params=params())


def build() -> VADProcessor:
    """The VAD processor for the pipeline.

    In pipecat 1.3 VAD is a pipeline processor rather than a transport
    parameter, so it sits explicitly between the transport input and the STT
    service where its ordering is visible.
    """
    log.info(
        "voice.vad_configured",
        stop_secs=settings.vad_stop_secs,
        min_volume=settings.vad_min_volume,
    )
    return VADProcessor(vad_analyzer=analyzer())
