"""Staging directory and promote workflow for ML model training.

Training output goes to .staging/ inside the rule's model directory.
The promote workflow backs up the existing model before copying staged files.
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def get_staging_dir(rule_dir: Path) -> Path:
    """Get or create the staging directory for a rule's model.

    Args:
        rule_dir: Path to the rule directory (e.g. regeln/abkuerzungen/).

    Returns:
        Path to the staging directory (rule_dir/model/.staging/).
    """
    staging = rule_dir / "model" / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def promote_model(rule_dir: Path) -> Path:
    """Promote a staged model to production with automatic backup.

    Steps:
        1. Verify .staging/ exists and is not empty.
        2. Back up all non-hidden items from model/ to model/.backup/<timestamp>/.
        3. Copy all items from .staging/ to model/ root, overwriting existing files.

    Args:
        rule_dir: Path to the rule directory.

    Returns:
        Path to the backup directory.

    Raises:
        FileNotFoundError: If no staged model is found.
    """
    model_dir = rule_dir / "model"
    staging_dir = model_dir / ".staging"

    if not staging_dir.exists() or not any(staging_dir.iterdir()):
        raise FileNotFoundError(f"No staged model found at {staging_dir}")

    # Back up existing model (skip hidden dirs like .staging, .backup)
    backup_dir = model_dir / ".backup" / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    for item in model_dir.iterdir():
        if item.name.startswith("."):
            continue
        dest = backup_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Copy staged model to production (overwrite existing files)
    for item in staging_dir.iterdir():
        dest = model_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    return backup_dir
