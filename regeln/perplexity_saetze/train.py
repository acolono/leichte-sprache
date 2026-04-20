#!/usr/bin/env python3
"""
Trainiert das Kneser-Ney n-Gramm Perplexity-Modell für die Satz-Komplexitätserkennung.

Verwendung:
    python -m regeln.perplexity_saetze.train --corpus mein_leichte_sprache_korpus.txt

Das Modell wird standardmäßig in diesem Verzeichnis als perplexity_model.pkl gespeichert.
"""

import pickle
from datetime import datetime
from pathlib import Path

import nltk
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

# Pfad zu diesem Modul
import logging

logger = logging.getLogger(__name__)
MODULE_DIR = Path(__file__).parent


def ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        logger.info(" Lade NLTK-Daten...")
        nltk.download("punkt")
        nltk.download("punkt_tab")


def load_corpus(corpus_path):
    """Lädt Korpus schnell."""
    logger.info(" Lade Korpus: %s", corpus_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        content = f.read()

    sentences = [line.strip() for line in content.split("\n") if line.strip()]
    logger.info(" %s Sätze geladen", len(sentences))
    return sentences


def tokenize_sentences(sentences):
    """Verbesserte Tokenisierung."""
    logger.info(" Tokenisiere...")
    tokenized = []

    for sentence in sentences:
        try:
            # NLTK Tokenisierung mit Filterung
            tokens = nltk.word_tokenize(sentence.lower(), language="german")
            filtered = [t for t in tokens if t.isalpha() or t in [".", ",", "!", "?"]]

            if len(filtered) >= 3:
                tokenized.append(filtered)
        except Exception:
            continue

    logger.info(" %s Sätze tokenisiert", len(tokenized))
    return tokenized


def train_model(tokenized_data, n_gram_size=3):
    """Trainiert Kneser-Ney Modell."""
    logger.info(" Trainiere %s-Gramm Kneser-Ney Modell...", n_gram_size)

    train_data, vocab = padded_everygram_pipeline(n_gram_size, tokenized_data)
    model = KneserNeyInterpolated(n_gram_size)
    model.fit(train_data, vocab)

    logger.info(" Modell trainiert, Vokabular: %s", len(model.vocab))
    return model


def save_model(model, output_path):
    """Speichert Modell."""
    logger.info(" Speichere Modell: %s", output_path)

    # Kalibrierte Schwellenwerte (basierend auf Tests)
    thresholds = {
        "THRESHOLD_KOMPLEX_SATZ": 15000.0,
        "THRESHOLD_SEHR_KOMPLEX_SATZ": 25000.0,
    }

    metadata = {
        "model": model,
        "thresholds": thresholds,
        "config": {
            "n_gram_size": 3,
            "smoothing": "kneser_ney",
            "timestamp": datetime.now().isoformat(),
        },
    }

    with open(output_path, "wb") as f:
        pickle.dump(metadata, f)

    logger.info(" Modell gespeichert!")
    return metadata


def demo_analysis(model):
    """Zeigt schnelle Demo."""
    logger.info("\n Schnelle Beispiele:")

    examples = [
        "der mann geht nach hause",
        "die administration evaluiert das konzept zur implementierung",
        "wir spielen im park",
    ]

    for sentence in examples:
        try:
            tokens = nltk.word_tokenize(sentence.lower(), language="german")
            tokens = [t for t in tokens if t.isalpha()]

            if len(tokens) >= 2:
                ppl = model.perplexity(tokens)
                complexity = (
                    "EINFACH"
                    if ppl < 1000
                    else "KOMPLEX"
                    if ppl < 15000
                    else "SEHR KOMPLEX"
                )
                logger.info(" '%s' → %.1f (%s)", sentence, ppl, complexity)
        except Exception:
            logger.error(" '%s' → Fehler", sentence)


def main():
    import argparse

    # Default output ist im selben Verzeichnis wie dieses Skript
    default_output = MODULE_DIR / "perplexity_model.pkl"

    parser = argparse.ArgumentParser(description="Perplexity Model Training")
    parser.add_argument(
        "--corpus",
        default="./mein_leichte_sprache_korpus.txt",
        help="Pfad zur Korpus-Datei (eine Zeile pro Satz)",
    )
    parser.add_argument(
        "--output",
        default=str(default_output),
        help=f"Ausgabepfad für das Modell (default: {default_output})",
    )
    args = parser.parse_args()

    logger.info(" Quick Perplexity Training")
    logger.info("=" * 40)

    # 1. Daten laden
    ensure_nltk_data()
    sentences = load_corpus(args.corpus)

    # 2. Tokenisierung
    tokenized = tokenize_sentences(sentences)

    # 3. Modell trainieren
    model = train_model(tokenized, n_gram_size=3)

    # 4. Modell speichern
    metadata = save_model(model, args.output)

    # 5. Demo
    demo_analysis(model)

    logger.info("\n Verwende diese Schwellenwerte in regel_perplexity_saetze.py:")
    logger.info(" THRESHOLD_KOMPLEX_SATZ = %s", metadata['thresholds']['THRESHOLD_KOMPLEX_SATZ'])
    logger.info(" THRESHOLD_SEHR_KOMPLEX_SATZ = %s", metadata['thresholds']['THRESHOLD_SEHR_KOMPLEX_SATZ'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
