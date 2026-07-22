"""Loads config/services.{cloud,local}.yaml and merges them.

A reachable local NIM entry shadows its cloud counterpart, so hosted-vs-self-hosted
is a config flip with no code branch.

The merge is per *service*, not per key: a self-hosted ASR entry replaces the cloud
ASR entry wholesale rather than inheriting stray fields like ``function_id`` from
it, which would otherwise be sent to a local server that has no idea what it means.

Reachability is probed with a plain TCP connect, not an API call. Probing properly
would mean speaking gRPC to one service and HTTP to another before we know which
we are talking to; a closed port is the only signal we need to decide "not running".
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[4] / "config"
CLOUD_FILE = CONFIG_DIR / "services.cloud.yaml"
LOCAL_FILE = CONFIG_DIR / "services.local.yaml"

ServiceName = Literal["llm", "stt", "tts", "embeddings"]
Transport = Literal["rest", "grpc"]

_PROBE_TIMEOUT_SECS = 0.5


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """One NIM service, resolved to either its hosted or self-hosted endpoint."""

    name: str
    provider: str
    transport: Transport
    model: str
    # REST only.
    base_url: str | None = None
    # gRPC only.
    server: str | None = None
    use_ssl: bool = True
    # NVCF routes by function id. Empty when self-hosted -- the server IS the function.
    function_id: str = ""
    # Free-form per-service extras (voice, language, synthesis_mode, params, ...).
    options: dict[str, Any] = field(default_factory=dict)
    # True when this entry came from services.local.yaml.
    self_hosted: bool = False

    @property
    def is_hosted(self) -> bool:
        return not self.self_hosted

    @property
    def endpoint(self) -> str:
        """A single human-readable address, for logs and health output."""
        return self.base_url or self.server or ""

    def option(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    @property
    def grpc_server(self) -> str:
        """The gRPC address, asserted non-null.

        ``_to_spec`` already rejects a grpc entry without one, but the field is
        Optional because REST specs have none. This narrows it for callers
        rather than making each one repeat the check.
        """
        if not self.server:
            raise ValueError(f"service {self.name!r} has no gRPC server configured")
        return self.server

    @property
    def rest_base_url(self) -> str:
        if not self.base_url:
            raise ValueError(f"service {self.name!r} has no REST base_url configured")
        return self.base_url


_KNOWN_KEYS = frozenset(
    {"provider", "transport", "base_url", "server", "use_ssl", "model", "function_id"}
)


def _to_spec(name: str, raw: dict[str, Any], *, self_hosted: bool) -> ServiceSpec:
    transport = raw.get("transport")
    if transport not in ("rest", "grpc"):
        raise ValueError(f"service {name!r}: transport must be 'rest' or 'grpc', got {transport!r}")

    if transport == "rest" and not raw.get("base_url"):
        raise ValueError(f"service {name!r}: rest transport requires base_url")
    if transport == "grpc" and not raw.get("server"):
        raise ValueError(f"service {name!r}: grpc transport requires server")

    return ServiceSpec(
        name=name,
        provider=raw.get("provider", "nvidia"),
        transport=transport,
        model=raw.get("model", ""),
        base_url=raw.get("base_url"),
        server=raw.get("server"),
        use_ssl=bool(raw.get("use_ssl", True)),
        function_id=raw.get("function_id") or "",
        options={k: v for k, v in raw.items() if k not in _KNOWN_KEYS},
        self_hosted=self_hosted,
    )


def _load(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping of service name -> config")
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def _host_port(spec: ServiceSpec) -> tuple[str, int] | None:
    if spec.transport == "grpc" and spec.server:
        host, _, port = spec.server.rpartition(":")
        return (host, int(port)) if host and port.isdigit() else None
    if spec.base_url:
        url = urlparse(spec.base_url)
        if url.hostname:
            return url.hostname, url.port or (443 if url.scheme == "https" else 80)
    return None


def is_reachable(spec: ServiceSpec, timeout: float = _PROBE_TIMEOUT_SECS) -> bool:
    """TCP-connect probe. False for a malformed address rather than raising."""
    target = _host_port(spec)
    if target is None:
        return False
    try:
        with socket.create_connection(target, timeout=timeout):
            return True
    except OSError:
        return False


def load_catalog(*, prefer_local: bool = True) -> dict[str, ServiceSpec]:
    """Resolve every service to exactly one endpoint.

    ``prefer_local=False`` forces the hosted endpoints even when a local NIM is up,
    which is what ``NIM_PROFILE=cloud`` means.
    """
    cloud = {n: _to_spec(n, raw, self_hosted=False) for n, raw in _load(CLOUD_FILE).items()}
    if not prefer_local:
        return cloud

    resolved = dict(cloud)
    for name, raw in _load(LOCAL_FILE).items():
        local = _to_spec(name, raw, self_hosted=True)
        # An unreachable local override is a stale comment, not an outage: fall
        # back to hosted rather than failing a session start.
        if is_reachable(local):
            resolved[name] = local
    return resolved


@lru_cache(maxsize=2)
def _cached_catalog(prefer_local: bool) -> dict[str, ServiceSpec]:
    return load_catalog(prefer_local=prefer_local)


def get_service(name: ServiceName, *, profile: str | None = None) -> ServiceSpec:
    """The resolved spec for one service. Cached: reachability is probed once."""
    if profile is None:
        from app.core.config import settings

        profile = settings.nim_profile

    catalog = _cached_catalog(profile == "local")
    if name not in catalog:
        raise KeyError(f"no NIM service configured named {name!r} (checked {CLOUD_FILE})")
    return catalog[name]


def reset_cache() -> None:
    """Re-probe on the next lookup. For tests and for a local NIM coming up late."""
    _cached_catalog.cache_clear()
