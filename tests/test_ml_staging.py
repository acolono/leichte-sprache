"""Tests for tools/ml/staging.py -- staging directory and promote workflow."""

import pytest

from tools.ml.staging import get_staging_dir, promote_model


def test_get_staging_dir_creates(tmp_path):
    (tmp_path / "model").mkdir()
    staging = get_staging_dir(tmp_path)
    assert staging == tmp_path / "model" / ".staging"
    assert staging.exists()


def test_promote_no_staging(tmp_path):
    (tmp_path / "model").mkdir()
    with pytest.raises(FileNotFoundError):
        promote_model(tmp_path)


def test_promote_empty_staging(tmp_path):
    staging = tmp_path / "model" / ".staging"
    staging.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        promote_model(tmp_path)


def test_promote_creates_backup(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    # Create an existing model file
    (model_dir / "config.json").write_text('{"key": "old"}')

    # Create staged file
    staging = model_dir / ".staging"
    staging.mkdir()
    (staging / "config.json").write_text('{"key": "new"}')

    backup_dir = promote_model(tmp_path)

    # Backup directory should exist under .backup/
    assert backup_dir.parent == model_dir / ".backup"
    assert (backup_dir / "config.json").exists()
    assert (backup_dir / "config.json").read_text() == '{"key": "old"}'


def test_promote_copies_staged(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"key": "old"}')

    staging = model_dir / ".staging"
    staging.mkdir()
    (staging / "config.json").write_text('{"key": "new"}')
    (staging / "weights.bin").write_bytes(b"staged_weights")

    promote_model(tmp_path)

    # Staged files should be in model root
    assert (model_dir / "config.json").read_text() == '{"key": "new"}'
    assert (model_dir / "weights.bin").read_bytes() == b"staged_weights"
