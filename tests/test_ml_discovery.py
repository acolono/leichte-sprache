"""Tests for tools/ml/discovery.py -- rule discovery and metadata loading."""

from pathlib import Path

from tools.ml.discovery import discover_trainable_rules, get_training_status, load_metadata


def test_discover_finds_abkuerzungen():
    rules = discover_trainable_rules()
    assert "abkuerzungen" in rules


def test_discover_returns_paths():
    rules = discover_trainable_rules()
    for value in rules.values():
        assert isinstance(value, Path)


def test_load_metadata_abkuerzungen():
    metadata = load_metadata("abkuerzungen")
    for key in ("model_type", "framework", "description"):
        assert key in metadata, f"metadata missing key: {key}"


def test_training_status_trained():
    # abkuerzungen has model/ directory with config.json
    rule_dir = Path(__file__).resolve().parent.parent / "regeln" / "abkuerzungen"
    status = get_training_status(rule_dir)
    assert status == "trained"


def test_training_status_untrained(tmp_path):
    status = get_training_status(tmp_path)
    assert status == "untrained"
