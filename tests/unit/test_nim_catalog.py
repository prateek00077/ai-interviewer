"""The cloud/local NIM catalog merge.

The behaviour worth pinning is the *shadowing* rule: a local entry replaces its
cloud counterpart wholesale only when the endpoint is actually reachable, and a
stale local entry must never take a service offline.
"""

from pathlib import Path

import pytest

from app.modules.voice.nvidia import catalog
from app.modules.voice.nvidia.catalog import ServiceSpec, load_catalog


@pytest.fixture(autouse=True)
def _clear_cache():
    catalog.reset_cache()
    yield
    catalog.reset_cache()


def _write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cloud: str, local: str = "") -> None:
    cloud_file = tmp_path / "services.cloud.yaml"
    local_file = tmp_path / "services.local.yaml"
    cloud_file.write_text(cloud)
    local_file.write_text(local)
    monkeypatch.setattr(catalog, "CLOUD_FILE", cloud_file)
    monkeypatch.setattr(catalog, "LOCAL_FILE", local_file)


CLOUD = """
llm:
  provider: nvidia
  transport: rest
  base_url: https://integrate.api.nvidia.com/v1
  model: nvidia/nemotron-3-nano-30b-a3b
tts:
  provider: nvidia
  transport: grpc
  server: grpc.nvcf.nvidia.com:443
  use_ssl: true
  model: magpie-tts-multilingual
  function_id: 877104f7-e885-42b9-8de8-f6e4c6303969
  voice: Magpie-Multilingual.EN-US.Aria
"""

LOCAL_TTS = """
tts:
  provider: nvidia
  transport: grpc
  server: localhost:50051
  use_ssl: false
  model: magpie-tts-multilingual
  voice: Magpie-Multilingual.EN-US.Aria
"""


def test_real_config_files_parse():
    """The shipped config/services.*.yaml must always be loadable."""
    resolved = load_catalog(prefer_local=False)
    assert {"llm", "stt", "tts", "embeddings"} <= set(resolved)
    assert resolved["stt"].transport == "grpc"
    assert resolved["stt"].function_id
    assert resolved["llm"].transport == "rest"


def test_unknown_keys_land_in_options(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, CLOUD)
    tts = load_catalog(prefer_local=False)["tts"]
    assert tts.option("voice") == "Magpie-Multilingual.EN-US.Aria"
    assert "model" not in tts.options  # known keys are promoted to fields, not duplicated


def test_reachable_local_entry_shadows_cloud(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, CLOUD, LOCAL_TTS)
    monkeypatch.setattr(catalog, "is_reachable", lambda spec, timeout=0.5: True)

    resolved = load_catalog(prefer_local=True)
    assert resolved["tts"].server == "localhost:50051"
    assert resolved["tts"].self_hosted is True
    # Wholesale replacement: the NVCF function id must NOT leak into a local entry,
    # which would send an unroutable header to a server that never asked for it.
    assert resolved["tts"].function_id == ""
    # Services with no local override are untouched.
    assert resolved["llm"].self_hosted is False


def test_unreachable_local_entry_falls_back_to_cloud(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, CLOUD, LOCAL_TTS)
    monkeypatch.setattr(catalog, "is_reachable", lambda spec, timeout=0.5: False)

    resolved = load_catalog(prefer_local=True)
    assert resolved["tts"].server == "grpc.nvcf.nvidia.com:443"
    assert resolved["tts"].self_hosted is False


def test_cloud_profile_ignores_a_running_local_nim(tmp_path, monkeypatch):
    _write(tmp_path, monkeypatch, CLOUD, LOCAL_TTS)
    monkeypatch.setattr(catalog, "is_reachable", lambda spec, timeout=0.5: True)

    assert load_catalog(prefer_local=False)["tts"].self_hosted is False


@pytest.mark.parametrize(
    "bad",
    [
        "llm:\n  transport: carrier-pigeon\n  model: x\n",
        "llm:\n  transport: rest\n  model: x\n",  # rest without base_url
        "stt:\n  transport: grpc\n  model: x\n",  # grpc without server
    ],
)
def test_malformed_entries_fail_loudly(tmp_path, monkeypatch, bad):
    _write(tmp_path, monkeypatch, bad)
    with pytest.raises(ValueError):
        load_catalog(prefer_local=False)


def test_is_reachable_is_false_for_a_malformed_address():
    spec = ServiceSpec(name="x", provider="nvidia", transport="grpc", model="m", server="no-port")
    assert catalog.is_reachable(spec) is False


def test_get_service_rejects_an_unconfigured_name():
    with pytest.raises(KeyError):
        catalog.get_service("telepathy", profile="cloud")  # type: ignore[arg-type]
