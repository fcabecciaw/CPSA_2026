"""The persisted default choice (``<cache dir>/default.json``)."""

from __future__ import annotations

import json
from pathlib import Path

from cpsa_tts.defaults import DefaultStore, apply_persisted
from cpsa_tts.engines import EngineRegistry


def test_store_roundtrip(tmp_path: Path) -> None:
    store = DefaultStore(tmp_path / "cache" / "default.json")
    assert store.read() is None
    path = store.write("edge", "it-IT-ElsaNeural")
    assert path.is_file()
    assert store.read() == {"engine": "edge", "voice": "it-IT-ElsaNeural"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["engine"] == "edge"
    assert "updated_at" in payload


def test_store_overwrites_and_leaves_no_temp_file(tmp_path: Path) -> None:
    store = DefaultStore(tmp_path / "default.json")
    store.write("kokoro", "if_sara")
    store.write("kokoro", "im_nicola")
    assert store.read() == {"engine": "kokoro", "voice": "im_nicola"}
    assert [child.name for child in tmp_path.iterdir()] == ["default.json"]


def test_malformed_default_file_is_ignored(tmp_path: Path) -> None:
    path = tmp_path / "default.json"
    path.write_text("{not json", encoding="utf-8")
    assert DefaultStore(path).read() is None
    path.write_text(json.dumps({"engine": "kokoro"}), encoding="utf-8")
    assert DefaultStore(path).read() is None
    path.write_text(json.dumps(["kokoro", "if_sara"]), encoding="utf-8")
    assert DefaultStore(path).read() is None


def test_apply_persisted_adopts_only_a_known_engine_and_voice(tmp_path: Path) -> None:
    store = DefaultStore(tmp_path / "default.json")
    registry = EngineRegistry(default_engine="kokoro", default_voice="if_sara")
    store.write("edge", "it-IT-ElsaNeural")
    assert apply_persisted(registry, store) == {"engine": "edge", "voice": "it-IT-ElsaNeural"}
    assert registry.defaults() == {"engine": "edge", "voice": "it-IT-ElsaNeural"}

    store.write("kokoro", "it-IT-ElsaNeural")  # a voice of another engine
    assert apply_persisted(registry, store) is None
    assert registry.defaults() == {"engine": "edge", "voice": "it-IT-ElsaNeural"}
