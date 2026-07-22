"""NvidiaLLMService wrapper (REST, integrate.api.nvidia.com/v1).

THINKING MUST BE OFF. Nemotron 3 reasons before answering by default, which adds
two to four seconds before the first token -- inside a 1.5s turn budget that is
not a latency regression, it is a broken product. The flag rides in
``extra_body.chat_template_kwargs``, not as a top-level parameter, so it is easy
to set somewhere that silently does nothing.

The service is OpenAI-shaped underneath, so pipecat's streaming and context
aggregation work unchanged.
"""

from __future__ import annotations

import structlog
from pipecat.services.nvidia.llm import NvidiaLLMService

from app.core.config import settings
from app.modules.voice.nvidia.catalog import ServiceSpec, get_service

log = structlog.get_logger(__name__)

# Answers are spoken aloud. A cap here is what stops the interviewer delivering
# a monologue while the candidate waits to talk.
MAX_TOKENS = 400
TEMPERATURE = 0.4


def build(spec: ServiceSpec | None = None) -> NvidiaLLMService:
    spec = spec or get_service("llm")
    api_key = settings.nvidia_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is required to start a voice session.")

    params = spec.option("params", {}) or {}
    extra_body: dict = {}
    if params.get("enable_thinking") is False:
        extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if "repetition_penalty" in params:
        extra_body["repetition_penalty"] = params["repetition_penalty"]

    log.info("voice.llm_configured", model=spec.model, thinking_off=bool(extra_body))
    return NvidiaLLMService(
        api_key=api_key,
        base_url=spec.rest_base_url,
        settings=NvidiaLLMService.Settings(
            model=spec.model,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra=extra_body,
        ),
    )
