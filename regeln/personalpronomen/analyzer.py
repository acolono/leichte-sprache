"""High-level API for German personal pronoun analysis.

Thin wrapper over PronounPredictor providing convenience methods
for pronoun detection used by regel.py.
"""

from pathlib import Path
from typing import Dict, List

from .predictor import PronounPredictor


class PronounAnalyzer:
    """High-level API for analyzing German personal pronouns.

    This class provides a simple interface for pronoun analysis with
    convenience methods for common use cases.

    Example:
        >>> analyzer = PronounAnalyzer(model_path, label_encoders_path)
        >>> result = analyzer.analyze("Ich gehe zu ihr nach Hause.")
    """

    def __init__(
        self,
        model_path: Path,
        label_encoders_path: Path,
        tokenizer_name: str = "bert-base-german-cased",
    ):
        """Initialize the analyzer.

        Args:
            model_path: Path to model checkpoint (.pt file)
            label_encoders_path: Path to label encoders JSON
            tokenizer_name: Name of tokenizer to use
        """
        self.predictor = PronounPredictor(
            model_path=model_path,
            label_encoders_path=label_encoders_path,
            tokenizer_name=tokenizer_name,
        )

    def analyze(self, text: str) -> Dict:
        """Analyze pronouns in text.

        Args:
            text: Input text (can be a sentence or multiple sentences)

        Returns:
            Dictionary containing analysis results with sentences and pronouns
        """
        return self.predictor.predict_text(text)

    def find_pronouns(self, text: str) -> List[tuple]:
        """Find personal pronouns in text.

        Args:
            text: Input text

        Returns:
            List of (pronoun, position) tuples
        """
        return self.predictor.find_pronouns(text)

    def find_polite_forms(self, text: str) -> List[Dict]:
        """Find polite forms (Sie/Ihnen) in text.

        Args:
            text: Input text

        Returns:
            List of pronouns identified as polite forms
        """
        results = self.analyze(text)
        polite_forms = []

        for sent_result in results["sentences"]:
            for pred in sent_result["pronouns"]:
                if pred["features"]["polite"]["value"] == "Form":
                    polite_forms.append({
                        "token": pred["token"],
                        "position": pred["position"],
                        "sentence": sent_result["sentence"],
                        "confidence": pred["features"]["polite"]["confidence"],
                    })

        return polite_forms

    def find_reflexive_pronouns(self, text: str) -> List[Dict]:
        """Find reflexive pronouns (sich, mich, etc.) in text.

        Args:
            text: Input text

        Returns:
            List of pronouns identified as reflexive
        """
        results = self.analyze(text)
        reflexive = []

        for sent_result in results["sentences"]:
            for pred in sent_result["pronouns"]:
                if pred["features"]["reflex"]["value"] == "True":
                    reflexive.append({
                        "token": pred["token"],
                        "position": pred["position"],
                        "sentence": sent_result["sentence"],
                        "confidence": pred["features"]["reflex"]["confidence"],
                    })

        return reflexive
