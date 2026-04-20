"""
Regel zur Erkennung komplexer Wörter basierend auf Korpus-Wahrscheinlichkeiten.
Vergleicht wordfreq-Wahrscheinlichkeiten mit Leichte-Sprache-Korpus-Wahrscheinlichkeiten.
"""

import json
from pathlib import Path
from typing import List

import spacy
from wordfreq import word_frequency

# Pfad zur Korpus-Wahrscheinlichkeiten-Datei
import logging

logger = logging.getLogger(__name__)
CORPUS_PROBS_FILE = Path(__file__).parent.parent / "corpus_probabilities.json"

# Schwellenwerte für verschiedene Komplexitäts-Kategorien
THRESHOLD_ABSTRAKT = 0.8  # Ratio > 5: Abstrakte/grammatisch komplexe Wörter
THRESHOLD_FREMDWORT = 0.6  # Ratio > 2: Potentielle Fremdwörter
MIN_CORPUS_COUNT = 1  # Minimum Vorkommen im Korpus für verlässliche Analyse


def _lade_korpus_wahrscheinlichkeiten() -> dict:
    """Lädt die vorberechneten Korpus-Wahrscheinlichkeiten."""
    corpus_file = Path(CORPUS_PROBS_FILE)
    if not corpus_file.exists():
        logger.error(" Korpus-Datei '%s' nicht gefunden!", CORPUS_PROBS_FILE)
        logger.error(" Bitte zuerst 'python process_corpus.py' ausführen.")
        return {}

    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(" Fehler beim Laden der Korpus-Wahrscheinlichkeiten: %s", e)
        return {}


def pruefe_regel(doc: spacy.tokens.Doc) -> List[str]:
    """
    Prüft Text auf komplexe Wörter durch Vergleich von Normal- und Korpus-Wahrscheinlichkeiten.

    Args:
        doc: spaCy Doc-Objekt mit verarbeitetem Text

    Returns:
        Liste von Fehlermeldungen mit Wahrscheinlichkeits-Ratios
    """
    fehler = []

    # Lade Korpus-Wahrscheinlichkeiten
    corpus_probs = _lade_korpus_wahrscheinlichkeiten()
    if not corpus_probs:
        return [
            "Korpus-Wahrscheinlichkeiten nicht verfügbar. Bitte process_corpus.py ausführen."
        ]

    # Analysiere jedes Wort im Text
    for token in doc:
        # Nur Vollwörter analysieren (keine Funktionswörter/Interpunktion)
        if not token.is_alpha or token.is_stop or len(token.lemma_) < 3:
            continue

        lemma = token.lemma_.lower()

        prob_normal = word_frequency(lemma, "de")
        prob_corpus = corpus_probs.get(lemma, 0)

        # Fall 1: Wort kommt gar nicht im LS-Korpus vor (sehr komplex!)
        if (
            prob_corpus == 0 and prob_normal > 0.0000001
        ):  # Nur relevante Normalsprache-Wörter
            fehler.append(
                f'Komplexes Wort "{token.text}" kommt nicht in Leichter Sprache vor. '
                f"Durch einfacheres Wort ersetzen oder erklären."
            )

        # Fall 2: Wort im Korpus, aber ungewöhnlich hohes Normal/Korpus-Verhältnis
        elif prob_corpus > 0 and prob_normal > 0:
            ratio = prob_normal / prob_corpus

            if ratio > THRESHOLD_ABSTRAKT:
                fehler.append(
                    f'Abstrakt/komplexes Wort "{token.text}" (Ratio: {ratio:.1f}). '
                    f"Möglicherweise durch einfacheres Wort ersetzen."
                )
            elif ratio > THRESHOLD_FREMDWORT:
                fehler.append(
                    f'Potentielles Fremdwort "{token.text}" (Ratio: {ratio:.1f}). '
                    f"Eventuell durch deutsches Wort ersetzen oder erklären."
                )

    return sorted(fehler)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Eigenständiges Testen der Regel
    nlp = spacy.load("de_core_news_lg")

    test_texts = [
        "Die Administration evaluiert das Konzept zur Implementierung der neuen Strategie.",
        "Das Team organisiert eine Besprechung für morgen.",
        "Der Mann kauft Brot im Laden.",
        "Die Philosophie der modernen Gesellschaft ist kompliziert.",
    ]

    logger.info("--- Test: Regel Komplexe Wörter ---")

    for i, text in enumerate(test_texts, 1):
        logger.info("\n%s. Text: '%s'", i, text)
        doc = nlp(text)
        ergebnisse = pruefe_regel(doc)

        if ergebnisse:
            logger.info(" %s Problem(e):", len(ergebnisse))
            for problem in ergebnisse:
                logger.info("  - %s", problem)
        else:
            logger.info(" Keine komplexen Wörter gefunden.")
