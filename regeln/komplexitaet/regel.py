#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT-based complexity detection for Leichte Sprache.

This rule uses a pre-trained model for detecting complex words.
The implementation has been extracted into the `textkomplexitaet` package.

Usage:
    from regeln.regel_bert_komplexitaet import check_rule

    doc = nlp("Komplexer Text hier...")
    results = check_rule(doc)

The module uses the MiriUll/distilbert-german-text-complexity model,
which was specifically trained for German text complexity.
"""

import logging
from typing import List, Optional

import spacy.tokens

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

logger = logging.getLogger(__name__)

# Try to import the textkomplexitaet package (local in same folder for standalone)
try:
    from .textkomplexitaet import (
        DEFAULT_CONFIG,
        LENIENT_CONFIG,
        STRICT_CONFIG,
        ComplexityConfig,
        TextComplexityAnalyzer,
    )

    HAS_NEW_ANALYZER = True
except ImportError:
    # Fallback: try system-wide installation
    try:
        from textkomplexitaet import (
            DEFAULT_CONFIG,
            LENIENT_CONFIG,
            STRICT_CONFIG,
            ComplexityConfig,
            TextComplexityAnalyzer,
        )

        HAS_NEW_ANALYZER = True
    except ImportError:
        HAS_NEW_ANALYZER = False
        logger.warning(
            "textkomplexitaet Paket nicht gefunden. Bitte prüfen Sie die Installation."
        )


# Configuration presets (for backward compatibility)
STANDARD_CONFIG = DEFAULT_CONFIG if HAS_NEW_ANALYZER else None
STRICT_CONFIG_PRESET = STRICT_CONFIG if HAS_NEW_ANALYZER else None
LENIENT_CONFIG_PRESET = LENIENT_CONFIG if HAS_NEW_ANALYZER else None

# Current active configuration
CURRENT_CONFIG = STANDARD_CONFIG

def _create_analyzer() -> Optional["TextComplexityAnalyzer"]:
    """Create a new analyzer instance.

    The analyzer itself is lightweight (just holds config).
    Heavy model loading is handled by ComplexityModel via BaseMLRule.
    """
    if not HAS_NEW_ANALYZER:
        return None
    return TextComplexityAnalyzer(config=CURRENT_CONFIG)


def setze_konfiguration(neue_config) -> None:
    """
    Set the configuration for complexity analysis.

    Args:
        neue_config: ComplexityConfig instance
    """
    global CURRENT_CONFIG

    CURRENT_CONFIG = neue_config


def check_rule(doc: spacy.tokens.Doc) -> List[str]:
    """
    Main function for rule integration.

    Analyzes text for complex words and returns error messages.
    Uses the textkomplexitaet package for improved accuracy.

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of error messages for complex words with concrete suggestions
    """
    if not HAS_NEW_ANALYZER:
        return [
            "BERT-Komplexitätsanalyse nicht verfügbar: "
            "textkomplexitaet Paket nicht installiert."
        ]

    analyzer = _create_analyzer()
    if analyzer is None:
        return []

    try:
        # Analyze the document
        result = analyzer.analyze(doc.text, doc)

        # Convert to violation messages
        errors = []

        for word in result.complex_words:
            # Build informative message
            if word.suggestion:
                # Message with specific suggestion
                msg = (
                    f'{word.category or "Komplexes Wort"}: "{word.word}" '
                    f"({word.reason}). "
                    f'Besser: "{word.suggestion}"'
                )
            else:
                # Message without specific suggestion
                msg = (
                    f'Komplexes Wort: "{word.word}" '
                    f"({word.reason}). "
                    f"Prüfen Sie, ob ein einfacheres Wort möglich ist."
                )

            errors.append(msg)

        return errors

    except Exception as e:
        logger.error("Fehler bei BERT-Komplexitätsanalyse: %s", e)
        return [f"Fehler bei Komplexitätsanalyse: {str(e)}"]


def warmup() -> None:
    """Pre-load the model for faster first analysis."""
    if HAS_NEW_ANALYZER:
        from .textkomplexitaet.models import ComplexityModel

        ComplexityModel.get_model()


# For backward compatibility with old API
def setze_schwellenwerte(potentiell: float, komplex: float) -> None:
    """
    Set complexity thresholds (backward compatibility).

    Args:
        potentiell: Threshold for potentially complex words
        komplex: Threshold for definitely complex words
    """
    if HAS_NEW_ANALYZER:
        config = ComplexityConfig(simple_threshold=potentiell, medium_threshold=komplex)
        setze_konfiguration(config)


# Export for backward compatibility
DEBUG_CONFIG = (
    ComplexityConfig(
        simple_threshold=1.0, medium_threshold=2.0, word_ablation_threshold=0.1
    )
    if HAS_NEW_ANALYZER
    else None
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    """Test the rule standalone."""
    import spacy

    # Test text
    test_text = """
    Die Administration evaluiert die Implementation des neuen Konzepts.
    Das Haus ist groß. Es hat viele Zimmer.
    Der Mann geht nach Hause.
    Die komplexe Methodologie erfordert interdisziplinäre Zusammenarbeit.
    """

    logger.info("=== Test: regel_bert_komplexitaet ===\n")

    try:
        nlp = spacy.load("de_core_news_lg")
        doc = nlp(test_text.strip())

        logger.info("Text: %s...\n", test_text.strip()[:100])

        results = check_rule(doc)

        if results:
            logger.info(" %s komplexe Wörter gefunden:\n", len(results))
            for i, error in enumerate(results, 1):
                logger.error(" %s. %s\n", i, error)
        else:
            logger.info(" Keine komplexen Wörter gefunden.")

    except OSError as e:
        logger.error(" Fehler: %s", e)
        logger.error(" Bitte spaCy-Modell installieren: python -m spacy download de_core_news_lg")
