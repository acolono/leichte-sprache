"""Tests for tools/ml/device.py -- device detection and seed management."""

import pytest
import torch

from tools.ml.device import detect_device, set_seed


def test_detect_device_auto():
    result = detect_device("auto")
    assert result in ("cpu", "cuda", "mps")


def test_detect_device_cpu():
    result = detect_device("cpu")
    assert result == "cpu"


def test_detect_device_invalid_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown device"):
        detect_device("invalid")


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_detect_device_unavailable_raises_runtimeerror():
    with pytest.raises(RuntimeError, match="not available"):
        detect_device("cuda")


def test_seed_reproducibility():
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.equal(a, b)
