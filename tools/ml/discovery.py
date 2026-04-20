"""Rule discovery via train_config.py convention.

Scans regeln/*/train_config.py to find rules that support CLI-driven training.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

REGELN_DIR = Path(__file__).resolve().parent.parent.parent / "regeln"


def discover_trainable_rules() -> dict[str, Path]:
    """Find all rules that have a train_config.py module.

    Returns:
        Dict mapping rule name to its directory Path, sorted by name.
    """
    trainable: dict[str, Path] = {}
    for rule_dir in sorted(REGELN_DIR.iterdir()):
        if (
            rule_dir.is_dir()
            and rule_dir.name != "__pycache__"
            and (rule_dir / "train_config.py").exists()
        ):
            trainable[rule_dir.name] = rule_dir
    return trainable


def load_metadata(rule_name: str) -> dict[str, Any]:
    """Load METADATA dict from a rule's train_config module.

    Only reads the METADATA attribute -- heavy imports inside functions
    within train_config.py are not triggered.

    Args:
        rule_name: Name of the rule directory (e.g. "abkuerzungen").

    Returns:
        METADATA dict, or empty dict if not defined.
    """
    module = importlib.import_module(f"regeln.{rule_name}.train_config")
    return getattr(module, "METADATA", {})


def get_training_status(rule_dir: Path) -> str:
    """Check whether a rule has a trained model on disk.

    Looks for common model artifacts (*.pt, *.pkl, *.safetensors, config.json)
    inside the rule's model/ directory.

    Args:
        rule_dir: Path to the rule directory.

    Returns:
        "trained" or "untrained".
    """
    model_dir = rule_dir / "model"
    if not model_dir.exists():
        return "untrained"

    has_model = any(
        model_dir.glob(pattern)
        for pattern in ["*.pt", "*.pkl", "*.safetensors", "config.json"]
    )
    return "trained" if has_model else "untrained"
