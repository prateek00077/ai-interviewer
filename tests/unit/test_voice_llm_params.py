"""How NVIDIA-specific request fields reach the model.

Pipecat builds its request as ``params.update(settings.extra)`` and then calls
``client.chat.completions.create(**params)``. So anything in ``extra`` becomes a
TOP-LEVEL argument to the OpenAI SDK -- and the SDK rejects arguments it does
not know, before a request is ever built.

OBSERVED in a live interview: ``chat_template_kwargs`` was passed in ``extra``,
every LLM call raised ``AsyncCompletions.create() got an unexpected keyword
argument 'chat_template_kwargs'``, and the interviewer greeted nobody. The
WebRTC call connected, the pipeline reported ready, and it sat silent.
"""

import inspect

import pytest

from app.modules.voice.nvidia import llm as llm_service


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(
        llm_service.settings, "nvidia_api_key", type("S", (), {"get_secret_value": lambda _: "k"})()
    )
    return llm_service.build()


def test_nvidia_fields_are_nested_under_extra_body(service):
    """The whole bug in one assertion."""
    extra = service._settings.extra
    assert set(extra) == {"extra_body"}, (
        f"these become top-level SDK kwargs and will raise TypeError: {sorted(extra)}"
    )
    assert "chat_template_kwargs" in extra["extra_body"]


def test_thinking_is_off(service):
    """Nemotron 3 reasons before answering by default, adding seconds before the
    first token. Inside a turn budget that is a broken product, not a slow one."""
    body = service._settings.extra["extra_body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_every_top_level_param_is_one_the_sdk_accepts(service):
    """The check that would have caught it without a live call.

    Builds the request exactly as pipecat does and compares the keys against the
    SDK's own signature.
    """
    from openai.resources.chat.completions import AsyncCompletions

    params = service.build_chat_completion_params(
        {"messages": [{"role": "user", "content": "hello"}]}
    )
    accepted = set(inspect.signature(AsyncCompletions.create).parameters)
    unknown = sorted(set(params) - accepted)
    assert not unknown, f"the OpenAI SDK will reject these: {unknown}"


def test_the_spoken_answer_is_capped(service):
    """Answers are read aloud; without a cap the interviewer monologues while
    the candidate waits to talk."""
    assert 0 < service._settings.max_tokens <= 600
