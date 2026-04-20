"""
BERT-basierte Abkürzungserkennung mit trainiertem NER-Modell v2.0.

Diese Regel verwendet ein feinabgestimmtes BERT-Modell (bert-base-german-cased),
das auf deutschen Beispielen mit B-I-O Schema trainiert wurde, um Abkürzungen
mit hoher Präzision zu erkennen.

Modell v2.0 Neuerungen:
- B-I-O Tagging Schema (B-ABK, I-ABK, O) für Multi-Word Abbreviations
- Class Weighting (19.83x für I-ABK) zur Behandlung von Klassenimbalance
- 3-Label Klassifizierung statt 2-Label

Das Modell wurde auf SpaCy-tokenisierten Daten trainiert, aber BERT verwendet intern
WordPiece-Tokenisierung. Wir verwenden aggregation_strategy="simple" um Sub-Tokens
automatisch zu ganzen Wörtern zusammenzufassen.

Performance v2.0 (Test Set mit intelligentem Post-Processing):
- Precision: 97.00%
- Recall: 98.93%
- F1-Score: 97.95%
- Accuracy: 99.56%

Post-Processing-Features:
- Kontiguitäts-Aggregation: Heilt fragmentierte Tokens (d.h., CEO, AG & Co. KG)
- spaCy-basierte Satzzeichen-Validierung: Entfernt falsche Punkte am Satzende

Modell-Training: November 2025 (v2.0)
Modell-Pfad: bert_abbreviation_training/final_model/
Training-Zeit: 11.5 Minuten auf M1 Mac (3 Epochs)
Dataset: 11,877 Beispiele mit B-I-O Korrektur
"""

from pathlib import Path
from typing import Any, List

import spacy
from spacy.tokens import Doc

# Import Konfiguration
from .config import MIN_CONFIDENCE_THRESHOLD

import logging

from regeln.base_ml_rule import BaseMLRule

logger = logging.getLogger(__name__)


def _get_model_path() -> Path:
    """
    Ermittle den Pfad zum trainierten BERT-Modell.

    Returns:
        Path zum Modell-Verzeichnis
    """
    # Standalone-fähig: Modell liegt im selben Ordner wie regel.py
    model_path = Path(__file__).parent / "model"
    return model_path


class AbbreviationModel(BaseMLRule):
    """Thread-safe lazy loader for the abbreviation NER pipeline."""

    @classmethod
    def _load_model(cls) -> Any:
        """Load BERT NER pipeline and spaCy model for post-processing.

        Returns:
            Tuple of (ner_pipeline, spacy_nlp)

        Raises:
            OSError: If no spaCy model is available
            FileNotFoundError: If BERT model directory is missing
            ImportError: If transformers is not installed
        """
        # Lade spaCy Modell für Post-Processing (Satzzeichen-Validierung)
        try:
            spacy_nlp = spacy.load("de_core_news_lg")
        except OSError:
            # Fallback auf kleineres Modell
            spacy_nlp = spacy.load("de_core_news_sm")

        # Importiere Hugging Face Transformers
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        model_path = _get_model_path()

        # Prüfe ob Modell existiert
        if not model_path.exists():
            raise FileNotFoundError(f"BERT-Modell nicht gefunden: {model_path}")

        # 1. Lade Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        # 2. Lade Modell für Token-Klassifizierung
        model = AutoModelForTokenClassification.from_pretrained(str(model_path))

        # 3. Erstelle Pipeline für Token-Klassifizierung
        ner_pipeline = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=-1,
        )

        return (ner_pipeline, spacy_nlp)


def _heal_fragmented_predictions(
    raw_predictions: List[dict], original_text: str, spacy_nlp
) -> List[dict]:
    """
    Heilt fragmentierte Abkürzungsvorhersagen durch kontiguität-basierte Aggregation
    und spaCy-Validierung von Satzzeichen.

    Diese Funktion löst das Problem, dass der BERT-Tokenizer Abkürzungen fragmentiert:
    - 'd.h.' wird zu ['d', '.', 'h'] → heilt zu 'd.h.'
    - 'CEO' wird zu ['C', '##EO'] → heilt zu 'CEO'
    - 'AG.' (am Satzende) wird zu ['AG', '.'] → heilt zu 'AG' (ohne Punkt)

    Workflow:
    1. Kontiguitäts-Aggregation: Finde zusammenhängende ABK-Token-Blöcke
    2. Satzzeichen-Validierung: Nutze spaCy um falsche Satzzeichen am Ende zu entfernen

    Args:
        raw_predictions: Liste von Entity-Dicts mit 'word', 'start', 'end', 'score', 'entity_group'
        original_text: Original-Text für Validierung
        spacy_nlp: spaCy Modell für Satzzeichen-Validierung

    Returns:
        Liste von geheilten Entity-Dicts
    """
    if not raw_predictions:
        return []

    # Satzzeichen die am Ende einer Entität validiert werden müssen
    SENTENCE_PUNCTUATION = {".", "!", "?", ",", ";", ":"}

    healed_entities = []

    # SCHRITT 1: KONTIGUITÄTS-AGGREGATION
    # Sammle zusammenhängende Tokens basierend auf Character-Offsets

    i = 0
    while i < len(raw_predictions):
        current_entity = raw_predictions[i]

        # Starte einen neuen Block wenn ABK-Label gefunden
        if current_entity.get("entity_group", "").startswith(("ABK", "B-ABK", "I-ABK")):
            # Sammle alle aufeinanderfolgenden ABK-Tokens
            block_entities = [current_entity]
            block_start = current_entity["start"]
            block_end = current_entity["end"]

            # Schaue voraus und sammle weitere ABK-Tokens die direkt anschließen
            j = i + 1
            while j < len(raw_predictions):
                next_entity = raw_predictions[j]

                # Prüfe ob nächstes Token auch ABK ist UND direkt anschließt (keine Lücke)
                if (
                    next_entity.get("entity_group", "").startswith(
                        ("ABK", "B-ABK", "I-ABK")
                    )
                    and next_entity["start"] == block_end
                ):  # KRITISCH: Exakte Kontiguität!
                    block_entities.append(next_entity)
                    block_end = next_entity["end"]
                    j += 1
                else:
                    # Kontiguität gebrochen → Block ist komplett
                    break

            # Extrahiere den zusammengesetzten Text aus dem Original
            # WICHTIG: Verwende die Offsets um den EXAKTEN Text zu extrahieren
            merged_text = original_text[block_start:block_end]

            # Use mean confidence across all fragments for balanced scoring
            avg_confidence = sum(e["score"] for e in block_entities) / len(
                block_entities
            )

            # Debug: Prüfe ob extrahierter Text sinnvoll ist
            # Wenn der Text nur aus Whitespace/Sonderzeichen besteht, überspringe
            if not merged_text or not merged_text.strip():
                i = j if j > i + 1 else i + 1
                continue

            # SCHRITT 2: SATZZEICHEN-VALIDIERUNG MIT SPACY
            # Wenn Entität mit Satzzeichen endet, validiere ob es wirklich Teil der Abkürzung ist

            final_text = merged_text
            final_end = block_end

            if merged_text and merged_text[-1] in SENTENCE_PUNCTUATION:
                # Nutze spaCy um zu prüfen ob Satzzeichen Teil der Abkürzung ist
                try:
                    # Tokenisiere nur den extrahierten Text
                    spacy_doc = spacy_nlp(merged_text)

                    # Die Ausnahme-Regel:
                    # len > 1 → spaCy hat getrennt → Satzzeichen NICHT Teil der Abkürzung
                    # len == 1 → spaCy hat zusammengehalten → Satzzeichen IST Teil der Abkürzung

                    if len(spacy_doc) > 1:
                        # spaCy würde trennen → Satzzeichen entfernen
                        # Beispiel: "AG." → ['AG', '.'] → entferne '.'
                        final_text = merged_text[:-1]
                        final_end = block_end - 1

                        # Zusätzliche Sicherheit: Wenn nach Entfernung leer, überspringe
                        if not final_text.strip():
                            i = j  # Springe zum nächsten Token
                            continue

                    # else: len(spacy_doc) == 1 → behalte Satzzeichen
                    # Beispiel: "z.B." → ['z.B.'] → behalte 'z.B.'

                except Exception:
                    # Bei Fehler: Konservativ → entferne Satzzeichen
                    final_text = merged_text[:-1]
                    final_end = block_end - 1

                    if not final_text.strip():
                        i = j
                        continue

            # Erstelle geheilte Entität
            healed_entity = {
                "word": final_text,
                "start": block_start,
                "end": final_end,
                "score": avg_confidence,
                "entity_group": "ABK",  # Normalisiere Label
            }

            healed_entities.append(healed_entity)

            # Springe zum nächsten Token nach dem Block
            i = j
        else:
            # Kein ABK-Token → überspringe
            i += 1

    return healed_entities


def check_rule(doc: Doc) -> List[str]:
    """
    Erkenne Abkürzungen mit dem trainierten BERT-NER-Modell.

    Diese Funktion verwendet ein feinabgestimmtes BERT-Modell, um deutsche
    Abkürzungen im Text zu identifizieren. Das Modell wurde auf einem
    Datensatz deutscher Abkürzungen trainiert.

    Workflow:
    1. Lade BERT-Modell (nur beim ersten Aufruf)
    2. Wende Pipeline auf Rohtext an
    3. Filtere Predictions nach Abkürzungs-Label (ABK, B-ABK)
    4. Heile fragmentierte Predictions mit Post-Processing
    5. Erstelle Verletzungsmeldungen

    Args:
        doc: SpaCy Doc-Objekt mit dem zu prüfenden Text

    Returns:
        Liste von Verletzungsmeldungen (eine pro erkannter Abkürzung)

    Beispiel:
        >>> import spacy
        >>> nlp = spacy.load("de_core_news_lg")
        >>> doc = nlp("Die GmbH arbeitet mit ca. 50 Mitarbeitern.")
        >>> violations = check_rule(doc)
        >>> print(violations)
        ['Abkürzung erkannt: "GmbH" (Position: 4-8, Konfidenz: 99.5%)',
         'Abkürzung erkannt: "ca" (Position: 23-25, Konfidenz: 100.0%)']
    """

    # Lade Modell via BaseMLRule (thread-sicher, lazy, gecacht)
    result = AbbreviationModel.get_model()

    # Falls Modell nicht geladen werden konnte, gebe Warnung zurück
    if result is None:
        error = AbbreviationModel.get_error() or "Model unavailable"
        return [f"BERT-Abkuerzungserkennung nicht verfuegbar: {error}"]

    ner_pipeline, spacy_nlp = result

    # Prüfe ob Text leer ist
    if len(doc.text.strip()) == 0:
        return []

    violations = []

    try:
        # Wende BERT-NER-Modell auf Rohtext an
        # Pipeline führt automatisch folgende Schritte aus:
        # 1. Tokenisierung mit BERT WordPiece-Tokenizer
        # 2. Token-Klassifizierung mit BERT-Modell
        # 3. Aggregation von Sub-Tokens zu ganzen Wörtern (aggregation_strategy="simple")
        raw_predictions = ner_pipeline(doc.text)

        # Filtere nur Entities mit Abkürzungs-Label
        # Mit aggregation_strategy="simple" bekommen wir "entity_group"
        # v2.0: Unterstützt B-ABK und I-ABK Labels (B-I-O Schema)
        abbreviation_entities = [
            entity
            for entity in raw_predictions
            if entity.get("entity_group", "").startswith(("ABK", "B-ABK", "I-ABK"))
        ]

        # POST-PROCESSING: Heile fragmentierte Predictions
        # HINWEIS: Diese Funktion kombiniert fragmentierte Tokens (z.B. 'd', '.', 'h' → 'd.h.')
        # und entfernt falsche Satzzeichen am Ende (z.B. 'AG.' → 'AG')
        healed_entities = _heal_fragmented_predictions(
            raw_predictions=abbreviation_entities,
            original_text=doc.text,
            spacy_nlp=spacy_nlp,
        )

        # POST-PROCESSING: Filter false positives (embedded words, named entities)
        filtered_entities = []
        for entity in healed_entities:
            word = entity["word"]
            start_pos = entity["start"]
            end_pos = entity["end"]

            # Skip if embedded in a larger word (not at word boundary)
            if start_pos > 0 and doc.text[start_pos - 1].isalpha():
                continue
            if end_pos < len(doc.text) and doc.text[end_pos].isalpha():
                continue

            # Skip fragments from named entities that don't look like real abbreviations
            # Real abbreviations typically: end with ".", are all caps, or contain periods
            looks_like_abbreviation = (
                word.endswith(".")
                or word.isupper()
                or "." in word
                or word in ("GmbH", "AG", "KG", "OHG", "UG")  # Common company forms
            )

            if not looks_like_abbreviation:
                # Only then check if part of a named entity
                is_named_entity = False
                for token in doc:
                    if token.ent_type_ in (
                        "GPE",
                        "LOC",
                        "PER",
                        "ORG",
                    ):  # Named entities
                        token_start = token.idx
                        token_end = token.idx + len(token.text)
                        if (start_pos >= token_start and start_pos < token_end) or (
                            end_pos > token_start and end_pos <= token_end
                        ):
                            is_named_entity = True
                            break
                if is_named_entity:
                    continue

            filtered_entities.append(entity)

        # Verarbeite geheilte Entities
        for entity in filtered_entities:
            # Extrahiere Informationen
            abbreviation = entity["word"]
            confidence = entity["score"]  # Wert zwischen 0 und 1
            start_pos = entity["start"]
            end_pos = entity["end"]

            # FILTER: Nur Abkürzungen mit ausreichend hoher Konfidenz melden
            # Dies reduziert False Positives wie "Nachrichten", "Seil", etc.
            if confidence < MIN_CONFIDENCE_THRESHOLD:
                continue

            confidence_percent = confidence * 100  # Für Anzeige in Prozent

            # Finde Token-Position im SpaCy Doc (für bessere Integration)
            token_idx = None
            for i, token in enumerate(doc):
                if token.idx >= start_pos and token.idx < end_pos:
                    token_idx = i
                    break

            # Erstelle Verletzungsmeldung
            if token_idx is not None:
                violation = (
                    f'Abkürzung erkannt: "{abbreviation}" '
                    f"(Token-Position: {token_idx}, "
                    f"Text-Position: {start_pos}-{end_pos}, "
                    f"Konfidenz: {confidence_percent:.1f}%)"
                )
            else:
                # Fallback wenn Token nicht gefunden
                violation = (
                    f'Abkürzung erkannt: "{abbreviation}" '
                    f"(Text-Position: {start_pos}-{end_pos}, "
                    f"Konfidenz: {confidence_percent:.1f}%)"
                )

            violations.append(violation)

    except Exception as e:
        # Fehlerbehandlung: Gebe Fehlermeldung zurück statt abzubrechen
        violations.append(f"Fehler bei BERT-Abkuerzungserkennung: {str(e)}")

    return violations


# Metadaten für die Regel (optional, aber hilfreich für Dokumentation)
RULE_INFO = {
    "name": "BERT-basierte Abkürzungserkennung v2.0",
    "beschreibung": (
        "Erkennt deutsche Abkürzungen mit einem trainierten BERT-NER-Modell v2.0. "
        "Verwendet B-I-O Tagging Schema für Multi-Word Abbreviations. "
        "Intelligentes Post-Processing heilt fragmentierte Predictions und validiert Satzzeichen."
    ),
    "kategorie": "Lexikalisch",
    "schweregrad": "mittel",
    "modell": "bert-base-german-cased (feinabgestimmt v2.0 mit B-I-O)",
    "training_datum": "2025-11-12",
    "modell_version": "2.0",
    "performance": {
        "precision": 0.9700,
        "recall": 0.9893,
        "f1_score": 0.9795,
        "accuracy": 0.9956,
    },
    "training_info": {
        "dataset_size": 11877,
        "training_time_minutes": 11.5,
        "epochs": 3,
        "labels": ["B-ABK", "I-ABK", "O"],
        "class_weights": [3.14, 19.83, 0.38],
    },
    "beispiele": [
        "Die GmbH arbeitet mit ca. 50 Mitarbeitern.",
        "Prof. Dr. Schmidt ist der CEO.",
        "Die IT-Abt. plant die Server-Umstellung.",
        "Die Müller AG & Co. KG hat ihren Sitz in München.",
    ],
}


def get_info() -> dict:
    """
    Gebe Informationen über diese Regel zurück.

    Returns:
        Dictionary mit Regel-Metadaten
    """
    return RULE_INFO


# Test-Funktion (nur für Entwicklung)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Teste die Regel
    logger.info("Teste BERT-Abkürzungserkennung...")
    logger.info("=" * 80)

    import spacy

    # Lade SpaCy Modell
    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        logger.error("Fehler: de_core_news_lg nicht gefunden. Bitte installieren:")
        logger.info("python -m spacy download de_core_news_lg")
        exit(1)

    # Test-Texte mit verschiedenen Abkürzungen
    test_texte = [
        "Die Weber GmbH arbeitet mit ca. 50 Mitarbeitern.",
        "Prof. Dr. Schmidt ist der CEO der IT-Abt.",
        "Die Müller AG & Co. KG hat ihren Sitz in München.",
        "Tel.: 0123/456789, E-Mail: info@firma.de",
        "Die MwSt. beträgt 19%, d.h. der Preis ist inkl. Steuern.",
        "Er studiert an der TU München (Technische Universität) und braucht Geld von der EZB.",
    ]

    for text in test_texte:
        logger.info("\nText: %s", text)
        doc = nlp(text)
        violations = check_rule(doc)

        if violations:
            for v in violations:
                logger.info(" → %s", v)
        else:
            logger.info(" → Keine Abkürzungen erkannt")

    logger.info("\n" + "=" * 80)
    logger.info("Test abgeschlossen!")
