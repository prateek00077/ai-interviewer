"""Verify NVIDIA_API_KEY against all three NIM services (REST LLM, gRPC ASR, gRPC TTS).

Run this FIRST - it is the highest-risk assumption in the architecture.

    python scripts/check_nim.py

The split is the thing worth proving: the LLM is ordinary OpenAI-shaped REST on
integrate.api.nvidia.com, while ASR and TTS are Riva gRPC on grpc.nvcf.nvidia.com:443
routed by a `function-id` metadata header. One key, three very different call shapes.

Exits non-zero if any configured service fails, so CI can gate on it.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.modules.voice.nvidia.catalog import ServiceSpec, get_service  # noqa: E402

SAMPLE_RATE = 16_000
PROBE_TEXT = "Tell me about a project you are proud of."


@dataclass
class Result:
    service: str
    ok: bool
    detail: str
    ms: float = 0.0
    endpoint: str = ""


def _api_key() -> str:
    return settings.nvidia_api_key.get_secret_value()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


# The shared hosted endpoint sheds load with 503 "Worker local total request limit
# reached". That is capacity, not a broken key, and it must not read as a failure.
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
_MAX_ATTEMPTS = 4


async def _post_with_retry(
    client: httpx.AsyncClient, url: str, body: dict[str, object]
) -> httpx.Response:
    for attempt in range(_MAX_ATTEMPTS):
        resp = await client.post(
            url, headers={"Authorization": f"Bearer {_api_key()}"}, json=body
        )
        if resp.status_code not in _RETRY_STATUSES or attempt == _MAX_ATTEMPTS - 1:
            return resp
        await asyncio.sleep(2**attempt)
    raise AssertionError("unreachable")


# --- REST (LLM, embeddings) -------------------------------------------------


async def check_rest_llm(spec: ServiceSpec) -> Result:
    started = time.perf_counter()
    body: dict[str, object] = {
        "model": spec.model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "max_tokens": 16,
        "stream": False,
    }
    # Nemotron 3 reasons by default and would spend seconds thinking before the
    # first token. Off is not an optimisation here, it is a requirement.
    if spec.option("params", {}).get("enable_thinking") is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    try:
        async with httpx.AsyncClient(timeout=settings.nim_request_timeout_secs) as client:
            resp = await _post_with_retry(client, f"{spec.base_url}/chat/completions", body)
        if resp.status_code != 200:
            detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return Result("llm", False, detail, _elapsed_ms(started))
        text = resp.json()["choices"][0]["message"]["content"]
        return Result("llm", True, f"replied {text.strip()[:40]!r}", _elapsed_ms(started))
    except Exception as exc:  # noqa: BLE001 - this script reports failures, never raises
        return Result("llm", False, f"{type(exc).__name__}: {exc}", _elapsed_ms(started))


async def check_rest_embeddings(spec: ServiceSpec) -> Result:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=settings.nim_request_timeout_secs) as client:
            resp = await _post_with_retry(
                client,
                f"{spec.base_url}/embeddings",
                {
                    "model": spec.model,
                    "input": [PROBE_TEXT],
                    # nv-embedqa distinguishes the two sides of a retrieval pair;
                    # omitting input_type is a 400, not a default.
                    "input_type": "query",
                    "encoding_format": "float",
                },
            )
        if resp.status_code != 200:
            detail = f"HTTP {resp.status_code}: {resp.text[:200]}"
            return Result("embeddings", False, detail, _elapsed_ms(started))
        dims = len(resp.json()["data"][0]["embedding"])
        return Result("embeddings", True, f"{dims} dimensions", _elapsed_ms(started))
    except Exception as exc:  # noqa: BLE001
        return Result("embeddings", False, f"{type(exc).__name__}: {exc}", _elapsed_ms(started))


# --- gRPC (ASR, TTS) --------------------------------------------------------


def _riva_auth(spec: ServiceSpec):
    """Riva client + the NVCF routing header.

    ``function-id`` is what selects the model on NVCF. Self-hosted NIMs have no
    function id and reject the header, hence the conditional.
    """
    import riva.client  # type: ignore[import-not-found]

    metadata = [["authorization", f"Bearer {_api_key()}"]]
    if spec.function_id:
        metadata.append(["function-id", spec.function_id])
    return riva.client, riva.client.Auth(
        uri=spec.server, use_ssl=spec.use_ssl, metadata_args=metadata
    )


def check_grpc_stt(spec: ServiceSpec) -> Result:
    """Streaming, not offline.

    The hosted ASR function is online-only. ``offline_recognize`` against it fails
    with INVALID_ARGUMENT, which reads like a bad model name and is not one -- so
    the probe exercises the same streaming path the live interview will use.
    """
    started = time.perf_counter()
    try:
        riva_client, auth = _riva_auth(spec)
        service = riva_client.ASRService(auth)
        config = riva_client.RecognitionConfig(
            encoding=riva_client.AudioEncoding.LINEAR_PCM,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=spec.option("language", "en-US"),
            max_alternatives=1,
            enable_automatic_punctuation=True,
        )
        if spec.model:
            config.model = spec.model
        streaming_config = riva_client.StreamingRecognitionConfig(
            config=config, interim_results=True
        )
        # 100ms frames of silence: the shape a live mic feed arrives in.
        frames = [b"\x00\x00" * (SAMPLE_RATE // 10) for _ in range(10)]
        responses = sum(
            1
            for _ in service.streaming_response_generator(
                audio_chunks=iter(frames), streaming_config=streaming_config
            )
        )
        # Silence transcribes to nothing; a clean round trip is the whole assertion.
        detail = f"stream accepted 1s, {responses} responses"
        return Result("stt", True, detail, _elapsed_ms(started))
    except ImportError:
        return Result("stt", False, "nvidia-riva-client not installed")
    except Exception as exc:  # noqa: BLE001
        return Result("stt", False, f"{type(exc).__name__}: {exc}", _elapsed_ms(started))


def check_grpc_tts(spec: ServiceSpec) -> Result:
    started = time.perf_counter()
    try:
        riva_client, auth = _riva_auth(spec)
        service = riva_client.SpeechSynthesisService(auth)
        first_chunk_ms = 0.0
        audio_bytes = 0
        for chunk in service.synthesize_online(
            text=PROBE_TEXT,
            voice_name=spec.option("voice"),
            language_code=spec.option("language", "en-US"),
            sample_rate_hz=44_100,
            encoding=riva_client.AudioEncoding.LINEAR_PCM,
        ):
            if not first_chunk_ms:
                first_chunk_ms = _elapsed_ms(started)
            audio_bytes += len(chunk.audio)
        if not audio_bytes:
            return Result("tts", False, "stream returned no audio", _elapsed_ms(started))
        return Result(
            "tts", True, f"{audio_bytes} bytes, TTFB {first_chunk_ms:.0f}ms", _elapsed_ms(started)
        )
    except ImportError:
        return Result("tts", False, "nvidia-riva-client not installed")
    except Exception as exc:  # noqa: BLE001
        return Result("tts", False, f"{type(exc).__name__}: {exc}", _elapsed_ms(started))


# --- Driver -----------------------------------------------------------------

CHECKS = {
    "llm": check_rest_llm,
    "embeddings": check_rest_embeddings,
    "stt": check_grpc_stt,
    "tts": check_grpc_tts,
}


async def main() -> int:
    if not _api_key():
        print("NVIDIA_API_KEY is not set. Add it to .env (see .env.example).")
        return 2

    print(f"profile={settings.nim_profile}\n")
    results: list[Result] = []

    for name, check in CHECKS.items():
        try:
            spec = get_service(name)  # type: ignore[arg-type]
        except KeyError as exc:
            results.append(Result(name, False, str(exc)))
            continue

        print(f"checking {name:<11} {spec.model or '-':<34} {spec.endpoint}")
        # The Riva clients are blocking; a thread keeps them off the event loop.
        result = (
            await check(spec)
            if asyncio.iscoroutinefunction(check)
            else await asyncio.to_thread(check, spec)
        )
        result.endpoint = spec.endpoint
        results.append(result)

    print()
    for r in results:
        mark = "OK  " if r.ok else "FAIL"
        print(f"  [{mark}] {r.service:<11} {r.ms:7.0f}ms  {r.detail}")

    failed = [r.service for r in results if not r.ok]
    print()
    if failed:
        print(f"{len(failed)} service(s) failed: {', '.join(failed)}")
        return 1
    print("all NIM services reachable")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
