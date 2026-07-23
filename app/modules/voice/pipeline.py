"""Builds the Pipecat Pipeline (STT -> LLM -> TTS).

The order below is the turn budget, laid out left to right:

    transport in -> VAD -> STT -> user aggregator -> LLM -> TTS -> transport out
                                                        |
                                                  assistant aggregator

VAD sits before STT so end-of-turn is decided on raw audio rather than waiting
for a transcript. The two context aggregators bracket the LLM: the user side
appends the candidate's transcribed turn before inference, the assistant side
appends the reply after it, which is what gives the model memory of the
conversation without anyone maintaining a message list by hand.

The audio buffer is a passive tap at the end. It records both directions for the
offline transcript pass and the confidence signals; it pushes frames through
untouched, so it costs nothing on the turn budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from app.core.config import settings
from app.modules.voice import turn_detection
from app.modules.voice.nvidia import llm as llm_service
from app.modules.voice.nvidia import stt as stt_service
from app.modules.voice.nvidia import tts as tts_service
from app.modules.voice.observers import TranscriptObserver

log = structlog.get_logger(__name__)

# Both directions of the call, mixed to one track. Recorded at the ASR rate
# rather than the TTS rate: the offline pass re-transcribes this, and 16k is
# what the ASR wants anyway.
RECORDING_SAMPLE_RATE = stt_service.SAMPLE_RATE
RECORDING_CHANNELS = 1


@dataclass
class BuiltPipeline:
    """A pipeline and the handles the session manager needs to drive it."""

    task: PipelineTask
    audio_buffer: AudioBufferProcessor
    observer: TranscriptObserver
    tts: object
    context: LLMContext
    # Kept so the session manager can hang the opening greeting off
    # ``on_client_connected``. See session_manager._open_the_conversation.
    transport: SmallWebRTCTransport


def build(
    *,
    transport: SmallWebRTCTransport,
    messages: list[dict[str, str]],
    observer: TranscriptObserver,
) -> BuiltPipeline:
    """Assemble the live pipeline.

    ``messages`` is the server-assembled system prompt and opening instruction.
    It goes straight into the LLM context and is never serialised anywhere the
    client can reach.
    """
    stt = stt_service.build()
    llm = llm_service.build()
    tts = tts_service.build()
    vad = turn_detection.build()

    context = LLMContext(messages=messages)  # type: ignore[arg-type]
    aggregators = LLMContextAggregatorPair(context)

    audio_buffer = AudioBufferProcessor(
        sample_rate=RECORDING_SAMPLE_RATE,
        num_channels=RECORDING_CHANNELS,
    )

    pipeline = Pipeline(
        [
            transport.input(),
            vad,
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            # After transport.output() so it captures what was actually sent,
            # not what was queued.
            audio_buffer,
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[observer],
        # Idle = nobody has spoken, in either direction, for this long.
        idle_timeout_secs=settings.voice_idle_nudge_secs,
        # PIPECAT'S DEFAULT IS TO CANCEL THE PIPELINE ON IDLE, after 300s. That
        # is wrong for an interview twice over: five minutes of a candidate
        # thinking is not abandonment, and killing the call is not the right
        # response to silence -- asking "are you still there?" is. The session
        # manager handles the event and decides when to give up.
        cancel_on_idle_timeout=False,
        cancel_runner_on_idle_timeout=False,
    )

    log.info("voice.pipeline_built", processors=len(pipeline.processors))
    return BuiltPipeline(
        task=task,
        audio_buffer=audio_buffer,
        observer=observer,
        tts=tts,
        context=context,
        transport=transport,
    )
