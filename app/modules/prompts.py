"""Prompt templates loaded from config/prompts/*.yaml.

Prompts live in YAML rather than in Python string literals for one practical
reason: they are edited far more often than the code around them, usually by
someone tuning interview quality rather than changing behaviour. A prompt change
should not be a code diff full of escaping.

Rendering uses ``str.format``, so a literal brace in a template must be doubled.
That is deliberate -- the JSON shape examples in these prompts are full of
braces, and a templating engine that treated them as syntax would be worse.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROMPT_DIR = Path(__file__).resolve().parents[2] / "config" / "prompts"


@lru_cache(maxsize=16)
def load(name: str) -> dict[str, str]:
    """Load ``config/prompts/{name}.yaml``. Cached: these do not change at runtime."""
    path = PROMPT_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no prompt template at {path}")

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict) or "system" not in data:
        raise ValueError(f"{path}: expected a mapping with at least a 'system' key")
    return data


def render(name: str, **values: Any) -> list[dict[str, str]]:
    """Template -> chat messages, ready for the NIM client.

    A missing placeholder raises KeyError here rather than silently sending a
    prompt with a hole in it, which would produce plausible-looking garbage.
    """
    template = load(name)
    messages = [{"role": "system", "content": template["system"].strip()}]
    if "user" in template:
        messages.append({"role": "user", "content": template["user"].format(**values).strip()})
    return messages


def reset_cache() -> None:
    """For tests, and for editing a prompt without restarting."""
    load.cache_clear()
