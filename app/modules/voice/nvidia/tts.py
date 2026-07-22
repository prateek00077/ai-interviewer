"""NvidiaTTSService wrapper (gRPC, Magpie TTS Multilingual).

Magpie is the slowest thing to wake up in the pipeline -- a cold function
answered its first byte in 687ms during the Phase 0 probe, against ~70ms warm.
That is why prewarm.py exists and why session start gates on it: a candidate
should not hear silence for the first second of their interview.

Output is 44.1kHz because that is what Magpie synthesises natively. Asking for
16k would make it resample twice, once down here and once up in the browser.
"""

from __future__ import annotations

import structlog
from pipecat.services.nvidia.tts import NvidiaTTSService

from app.core.config import settings
from app.modules.voice.nvidia.catalog import ServiceSpec, get_service

log = structlog.get_logger(__name__)

SAMPLE_RATE = 44_100


def build(spec: ServiceSpec | None = None) -> NvidiaTTSService:
    spec = spec or get_service("tts")
    api_key = settings.nvidia_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required to start a voice session.")

    voice = spec.option("voice")
    log.info("voice.tts_configured", model=spec.model, voice=voice)
    return NvidiaTTSService(
        api_key=api_key,
        server=spec.grpc_server,
        use_ssl=spec.use_ssl,
        model_function_map={"function_id": spec.function_id, "model_name": spec.model},
        sample_rate=SAMPLE_RATE,
        settings=NvidiaTTSService.Settings(voice=voice),
    )
