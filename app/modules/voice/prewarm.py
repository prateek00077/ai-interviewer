"""Gate session start on Magpie TTS warm-up (cold starts are slow).

MEASURED: a cold Magpie function returned its first audio byte 687ms after the
request; warm it is around 70ms. That difference lands entirely on the
candidate's first impression -- they connect, say hello, and hear nothing for
most of a second while the function spins up.

WHY THIS GOES STRAIGHT TO RIVA rather than through pipecat's TTS service: what
is cold is the NVCF function on NVIDIA's side, not our client object. Pipecat's
service additionally refuses to synthesise until the pipeline has started it
("TTS service not initialized"), so warming through it would mean starting the
pipeline first -- which is the thing we are trying to warm up *before*. A direct
Riva call hits the same function id and leaves it warm for the pipeline that
follows.

A prewarm that fails or times out does NOT block the session. Starting warm is
better; starting late is worse than starting cold, and refusing to start at all
because an optimisation failed would strand a candidate.
"""

from __future__ import annotations

import asyncio
import time

import structlog

from app.core.config import settings
from app.modules.voice.nvidia.catalog import ServiceSpec, get_service

log = structlog.get_logger(__name__)

# Short and unremarkable. Long enough to force the function to actually
# synthesise, short enough not to waste a second of wall clock.
WARMUP_TEXT = "Hello."
WARMUP_SAMPLE_RATE = 44_100


def _synthesize_once(spec: ServiceSpec, api_key: str) -> int:
    """Blocking. Returns bytes of audio produced."""
    import riva.client

    metadata = [["authorization", f"Bearer {api_key}"]]
    if spec.function_id:
        metadata.append(["function-id", spec.function_id])

    auth = riva.client.Auth(
        uri=spec.grpc_server, use_ssl=spec.use_ssl, metadata_args=metadata
    )
    service = riva.client.SpeechSynthesisService(auth)

    produced = 0
    for chunk in service.synthesize_online(
        text=WARMUP_TEXT,
        voice_name=spec.option("voice"),
        language_code=spec.option("language", "en-US"),
        sample_rate_hz=WARMUP_SAMPLE_RATE,
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
    ):
        produced += len(chunk.audio)
        # One chunk proves the function is up. Draining the rest would only
        # spend wall clock the candidate is waiting through.
        break
    return produced


async def warm_tts(spec: ServiceSpec | None = None, timeout_secs: float | None = None) -> float:
    """Force a first synthesis. Returns how long it took, in milliseconds.

    Returns -1.0 if the warm-up failed or timed out, which is a signal for the
    log rather than an error for the caller.
    """
    spec = spec or get_service("tts")
    api_key = settings.nvidia_api_key.get_secret_value()
    if not api_key:
        return -1.0

    timeout = timeout_secs or settings.connect_prewarm_timeout_secs
    started = time.perf_counter()
    try:
        # The Riva client is blocking; a thread keeps it off the event loop that
        # is about to run a live call.
        await asyncio.wait_for(
            asyncio.to_thread(_synthesize_once, spec, api_key), timeout=timeout
        )
    except TimeoutError:
        log.warning("voice.prewarm_timeout", timeout_secs=timeout)
        return -1.0
    except Exception as exc:  # noqa: BLE001 - never block a session on this
        log.warning("voice.prewarm_failed", error=str(exc)[:200])
        return -1.0

    elapsed_ms = (time.perf_counter() - started) * 1000
    log.info("voice.prewarmed", ms=round(elapsed_ms))
    return elapsed_ms
