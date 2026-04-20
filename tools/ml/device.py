"""Centralized device detection and seed management for ML training."""

from __future__ import annotations


def detect_device(preference: str = "auto") -> str:
    """Detect the best available compute device.

    Args:
        preference: Device preference - "auto", "cpu", "cuda", or "mps".
            When "auto", selects CUDA > MPS > CPU based on availability.

    Returns:
        Device string: "cpu", "cuda", or "mps".

    Raises:
        ValueError: If preference is not a recognized device name.
        RuntimeError: If the requested device is not available on this hardware.
    """
    import torch

    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"

    validators = {
        "cuda": lambda: torch.cuda.is_available(),
        "mps": lambda: torch.backends.mps.is_available() and torch.backends.mps.is_built(),
        "cpu": lambda: True,
    }

    if preference not in validators:
        raise ValueError(
            f"Unknown device: {preference}. Choose from: auto, cpu, cuda, mps"
        )
    if not validators[preference]():
        raise RuntimeError(f"Device '{preference}' requested but not available")

    return preference


def set_seed(seed: int) -> None:
    """Set random seed across all frameworks for reproducibility.

    Sets seeds for: random, numpy, torch (CPU), torch.cuda, torch.mps.
    Also enables deterministic CuDNN behavior.

    Args:
        seed: Integer seed value.
    """
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
