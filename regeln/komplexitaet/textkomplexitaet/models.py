"""
Model loading and caching for TextKomplexitaet.

Uses BaseMLRule for thread-safe lazy loading of the complexity prediction model.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch

from regeln.base_ml_rule import BaseMLRule

logger = logging.getLogger(__name__)


class ComplexityModel(BaseMLRule):
    """Thread-safe lazy loading for the text complexity pipeline."""

    @classmethod
    def _load_model(cls):
        """Load the DistilBERT text complexity pipeline.

        Returns:
            A transformers text-classification pipeline.

        Raises:
            ImportError: If transformers is not installed.
        """
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        device = cls._detect_device()
        model_name = "MiriUll/distilbert-german-text-complexity"
        cache_dir = Path(__file__).parent / "data"

        logger.info("Loading model '%s' on device '%s'...", model_name, device)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, cache_dir=cache_dir
        )

        if device != "cpu":
            model = model.to(device)

        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            top_k=1,
        )

        logger.info("Model loaded successfully on %s", device)
        return pipe

    @staticmethod
    def _detect_device() -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def predict_complexity(text: str) -> Tuple[float, str]:
    """
    Predict complexity score for a text.

    Uses the ComplexityModel pipeline via BaseMLRule thread-safe loading.

    Args:
        text: Input text to analyze

    Returns:
        Tuple of (complexity_score, label)
        Score is on 1-7 scale where 1=simple, 7=complex
    """
    if not text or not text.strip():
        return 1.0, "simple"

    pipe = ComplexityModel.get_model()
    if pipe is None:
        return 4.0, "unknown"

    try:
        result = pipe(text[:512])  # Truncate to max length

        # Handle nested list from top_k parameter
        if isinstance(result, list):
            if len(result) > 0 and isinstance(result[0], list):
                result = result[0][0]  # [[{...}]] -> {...}
            elif len(result) > 0:
                result = result[0]

        # The model outputs labels like "LABEL_0" through "LABEL_6"
        # which correspond to complexity levels 1-7
        label = result.get("label", "LABEL_0")
        result.get("score", 0.0)

        # Extract numeric complexity level from label
        if label.startswith("LABEL_"):
            complexity_level = int(label.split("_")[1]) + 1
        else:
            # Try to parse as number
            try:
                complexity_level = float(label)
            except ValueError:
                complexity_level = 4.0  # Default to middle

        return float(complexity_level), label

    except Exception as e:
        logger.warning("Prediction error: %s", e)
        return 4.0, "unknown"  # Return middle value on error
