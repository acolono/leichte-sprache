"""Tests for the train_config.py contract (METADATA, train, evaluate)."""

import pytest


def test_metadata_exists():
    from regeln.abkuerzungen.train_config import METADATA

    assert isinstance(METADATA, dict)


def test_metadata_has_required_keys():
    from regeln.abkuerzungen.train_config import METADATA

    for key in ("model_type", "framework", "description"):
        assert key in METADATA, f"METADATA missing key: {key}"


def test_train_function_exists_and_callable():
    from regeln.abkuerzungen import train_config

    assert hasattr(train_config, "train")
    assert callable(train_config.train)


def test_evaluate_function_exists_and_callable():
    from regeln.abkuerzungen import train_config

    assert hasattr(train_config, "evaluate")
    assert callable(train_config.evaluate)


def test_existing_constants_preserved():
    from regeln.abkuerzungen import train_config

    assert hasattr(train_config, "DATA_FILE")
    assert hasattr(train_config, "SEED")
    assert hasattr(train_config, "MODEL_CHECKPOINT")
    assert train_config.SEED == 42
    assert train_config.MODEL_CHECKPOINT == "bert-base-german-cased"
