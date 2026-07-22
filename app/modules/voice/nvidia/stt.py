"""NvidiaSTTService wrapper (gRPC, grpc.nvcf.nvidia.com:443 + function-id).

WHY THIS FILE EXISTS AT ALL, given pipecat already ships NvidiaSTTService:
its default ``model_name`` is ``nemotron-asr-streaming``, and the NVCF function
it also ships as the default rejects that name with INVALID_ARGUMENT. Verified
directly -- streaming against the same function id succeeds with
``cache-aware-parakeet-rnnt-en-US-asr-streaming-sortformer`` and fails with
``nemotron-asr-streaming``, so pipecat's two defaults disagree with each other.

Constructing the service from those defaults therefore produces a pipeline that
builds fine and dies on the first frame of audio. The model name comes from
``config/services.cloud.yaml``, which scripts/check_nim.py verifies against the
live endpoint.

The model is "sortformer", meaning speaker diarization is built into the ASR.
Phase 6's second-speaker detection reads that off this stream rather than
running a separate diarization pass.
"""

from __future__ import annotations

import structlog
from pipecat.services.nvidia.stt import NvidiaSTTService

from app.core.config import settings
from app.modules.voice.nvidia.catalog import ServiceSpec, get_service

log = structlog.get_logger(__name__)

# Riva wants 16k mono PCM. Resampling happens in the transport, not here.
SAMPLE_RATE = 16_000

# ASR frames of trailing silence before end-of-utterance is declared. Part of
# the turn budget: too low and the interviewer talks over a candidate who paused
# to think, too high and every turn carries dead air.
STOP_HISTORY_FRAMES = 320

# One candidate. Anything beyond that is the proctoring signal, not a
# participant, so there is no reason to model more of them.
MAX_SPEAKERS = 2


def build(spec: ServiceSpec | None = None) -> NvidiaSTTService:
    """The configured ASR service.

    Raises rather than degrading if the key is missing: a pipeline that starts
    without ASR looks alive and transcribes nothing.
    """
    spec = spec or get_service("stt")
    api_key = settings.nvidia_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required to start a voice session.")

    log.info("voice.stt_configured", model=spec.model, server=spec.server)
    return NvidiaSTTService(
        api_key=api_key,
        server=spec.grpc_server,
        use_ssl=spec.use_ssl,
        # Both halves together. Passing only the function id would leave
        # pipecat's mismatched default model name in place.
        model_function_map={"function_id": spec.function_id, "model_name": spec.model},
        sample_rate=SAMPLE_RATE,
        stop_history=STOP_HISTORY_FRAMES,
        settings=NvidiaSTTService.Settings(
            language=spec.option("language", "en-US"),
            # Punctuation is not cosmetic here: the transcript is fed to the
            # scorer as prose, and an unpunctuated wall of text scores worse.
            automatic_punctuation=True,
            # The model is sortformer, so diarization costs nothing extra and is
            # what Phase 6 reads to detect a second voice in the room.
            speaker_diarization=True,
            diarization_max_speakers=MAX_SPEAKERS,
        ),
    )
