"""Non-Pipecat NIM calls: planning, scoring, embeddings.

Pipecat owns the live conversation. This module owns everything else that talks
to a model: embedding resume chunks now, generating question plans and scoring
transcripts later.

Retries are not optional here. The shared hosted endpoint sheds load with
``503 Worker local total request limit reached`` -- observed during Phase 0 --
and that is capacity, not a broken request. Retrying on 5xx and 429 with
exponential backoff turns a routine shed into a slower call rather than a failed
interview. 4xx other than 429 is our bug and is raised immediately; retrying it
just burns quota.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx
import structlog

from app.core.config import settings
from app.core.exceptions import AppError
from app.modules.voice.nvidia.catalog import ServiceSpec, get_service

log = structlog.get_logger(__name__)

RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_ATTEMPTS = 4
BASE_BACKOFF_SECS = 1.0
# nv-embedqa rejects oversized batches outright rather than truncating.
MAX_EMBED_BATCH = 32


class NimError(AppError):
    status_code = 502
    code = "model_unavailable"
    message = "The model service is unavailable."


def _headers() -> dict[str, str]:
    key = settings.nvidia_api_key.get_secret_value()
    if not key:
        raise NimError("NVIDIA_API_KEY is not configured.")
    return {"Authorization": f"Bearer {key}", "Accept": "application/json"}


async def _post(
    client: httpx.AsyncClient, url: str, body: dict[str, Any], *, what: str
) -> dict[str, Any]:
    """POST with backoff on transient failures. Returns the decoded JSON body."""
    last_detail = ""
    for attempt in range(MAX_ATTEMPTS):
        try:
            response = await client.post(url, headers=_headers(), json=body)
        except httpx.HTTPError as exc:
            # Connect timeouts and resets are transient by nature.
            last_detail = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                return response.json()
            last_detail = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRY_STATUSES:
                log.error("nim_request_failed", what=what, detail=last_detail)
                raise NimError()

        if attempt < MAX_ATTEMPTS - 1:
            # Jitter, so a burst of workers retrying a shed does not re-converge
            # on the same instant and shed again.
            delay = BASE_BACKOFF_SECS * (2**attempt) * (0.5 + random.random())
            log.warning(
                "nim_retry",
                what=what,
                attempt=attempt + 1,
                detail=last_detail,
                delay=round(delay, 2),
            )
            await asyncio.sleep(delay)

    log.error("nim_exhausted", what=what, attempts=MAX_ATTEMPTS, detail=last_detail)
    raise NimError()


# --- Embeddings -------------------------------------------------------------


async def embed(
    texts: list[str], *, input_type: str = "passage", spec: ServiceSpec | None = None
) -> list[list[float]]:
    """Embed texts, batching as needed. Order of the result matches the input.

    ``input_type`` is not decoration: nv-embedqa is an asymmetric model trained
    with distinct query and passage encodings. Embedding a stored chunk as
    "query" puts it in the wrong space and quietly degrades every later search,
    with no error to notice.
    """
    if not texts:
        return []

    spec = spec or get_service("embeddings")
    vectors: list[list[float]] = []

    async with httpx.AsyncClient(timeout=settings.nim_request_timeout_secs) as client:
        for start in range(0, len(texts), MAX_EMBED_BATCH):
            batch = texts[start : start + MAX_EMBED_BATCH]
            payload = await _post(
                client,
                f"{spec.base_url}/embeddings",
                {
                    "model": spec.model,
                    "input": batch,
                    "input_type": input_type,
                    "encoding_format": "float",
                },
                what="embeddings",
            )
            # The API is documented to preserve order, but it also returns an
            # explicit index. Sorting by it costs nothing and removes the
            # assumption entirely.
            items = sorted(payload["data"], key=lambda d: d.get("index", 0))
            if len(items) != len(batch):
                raise NimError(f"Embedding count mismatch: sent {len(batch)}, got {len(items)}")
            vectors.extend(item["embedding"] for item in items)

    return vectors


async def embed_one(text: str, *, input_type: str = "query") -> list[float]:
    """A single vector. Defaults to the query side, which is the common caller."""
    return (await embed([text], input_type=input_type))[0]


# --- Chat -------------------------------------------------------------------


async def complete(
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.2,
    spec: ServiceSpec | None = None,
) -> str:
    """One non-streaming completion. Returns the assistant's text.

    Streaming belongs in the live pipeline, where Pipecat handles it. Callers
    here -- plan generation, scoring -- consume a whole document and gain nothing
    from tokens arriving early.
    """
    spec = spec or get_service("llm")
    body: dict[str, Any] = {
        "model": spec.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    params = spec.option("params", {}) or {}
    if "repetition_penalty" in params:
        body["repetition_penalty"] = params["repetition_penalty"]
    if params.get("enable_thinking") is False:
        # Nemotron 3 reasons by default. Left on, it spends seconds and tokens
        # thinking before the first output token.
        body["chat_template_kwargs"] = {"enable_thinking": False}

    async with httpx.AsyncClient(timeout=settings.nim_request_timeout_secs) as client:
        payload = await _post(client, f"{spec.base_url}/chat/completions", body, what="chat")

    return payload["choices"][0]["message"]["content"]
