#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT-based number word detection for Leichte Sprache.

This rule uses a fine-tuned BERT token-classification model
(based on dbmdz/bert-base-german-cased) for precise detection of
written-out numbers in German texts.

The model detects 4 error types:
- BAD_WORD_NUM: Number written as word ("zwei", "acht")
- BAD_YEAR: Year written as word ("neunzehnhundert")
- BAD_PERCENT: Percentage written as word ("fuenfzig Prozent")
- BAD_COMPLEX_NUM: Complex number written as word ("dreiundzwanzig")

Usage:
    from regeln.zahlwoerter.regel import check_rule
    import spacy

    nlp = spacy.load("de_core_news_lg")
    doc = nlp("Das Kind ist acht Jahre alt.")
    errors = check_rule(doc)
    # ['Zahlwort "acht" sollte als Ziffer geschrieben werden. Text-Position: 12-16']
"""

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from spacy.tokens import Doc
from transformers import pipeline

# Import configuration
from . import config

# Suppress TensorFlow/Transformers warnings
import logging

from regeln.base_ml_rule import BaseMLRule

logger = logging.getLogger(__name__)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# RULE-BASED FILTER: Number words 0-50 with inflected forms
# ============================================================================

# Number words 0-50 with all inflected forms (without "elf", "acht", "ein/eine/einer")
# Exceptions are defined in config.NUMBER_WORDS_0_50_EXCEPTIONS
ZAHLWOERTER_0_50_MIT_FLEXIONEN = {
    # 0
    "null": ["null"],
    # 1 - wird komplett von BERT behandelt (Artikel vs. Zahl)
    # "ein": [...],  # Auskommentiert
    # 2
    "zwei": [
        "zwei",
        "zweier",
        "zweien",
        "zweite",
        "zweiten",
        "zweiter",
        "zweites",
        "zweimal",
        "zweitens",
    ],
    # 3
    "drei": [
        "drei",
        "dreier",
        "dreien",
        "dritte",
        "dritten",
        "dritter",
        "drittes",
        "dreimal",
        "drittens",
    ],
    # 4
    "vier": [
        "vier",
        "vierer",
        "vieren",
        "vierte",
        "vierten",
        "vierter",
        "viertes",
        "viermal",
        "viertens",
    ],
    # 5
    "fünf": [
        "fünf",
        "fünfer",
        "fünfen",
        "fünfte",
        "fünften",
        "fünfter",
        "fünftes",
        "fünfmal",
        "fünftens",
    ],
    # 6
    "sechs": [
        "sechs",
        "sechser",
        "sechsen",
        "sechste",
        "sechsten",
        "sechster",
        "sechstes",
        "sechsmal",
        "sechstens",
    ],
    # 7
    "sieben": [
        "sieben",
        "siebener",
        "siebenen",
        "siebte",
        "siebten",
        "siebter",
        "siebtes",
        "siebenmal",
        "siebtens",
    ],
    # 8 - mehrdeutig (Redewendungen), wird vom BERT behandelt
    # "acht": [...],  # In config.NUMBER_WORDS_0_50_EXCEPTIONS
    # 9
    "neun": [
        "neun",
        "neuner",
        "neunen",
        "neunte",
        "neunten",
        "neunter",
        "neuntes",
        "neunmal",
        "neuntens",
    ],
    # 10
    "zehn": [
        "zehn",
        "zehner",
        "zehnen",
        "zehnte",
        "zehnten",
        "zehnter",
        "zehntes",
        "zehnmal",
        "zehntens",
    ],
    # 11 - mehrdeutig (Fußball, Fantasy), wird vom BERT behandelt
    # "elf": [...],  # In config.NUMBER_WORDS_0_50_EXCEPTIONS
    # 12
    "zwölf": ["zwölf", "zwölfte", "zwölften", "zwölfter", "zwölftes", "zwölfmal"],
    # 13-19
    "dreizehn": ["dreizehn", "dreizehnte", "dreizehnten", "dreizehnter", "dreizehntes"],
    "vierzehn": ["vierzehn", "vierzehnte", "vierzehnten", "vierzehnter", "vierzehntes"],
    "fünfzehn": ["fünfzehn", "fünfzehnte", "fünfzehnten", "fünfzehnter", "fünfzehntes"],
    "sechzehn": ["sechzehn", "sechzehnte", "sechzehnten", "sechzehnter", "sechzehntes"],
    "siebzehn": ["siebzehn", "siebzehnte", "siebzehnten", "siebzehnter", "siebzehntes"],
    "achtzehn": ["achtzehn", "achtzehnte", "achtzehnten", "achtzehnter", "achtzehntes"],
    "neunzehn": ["neunzehn", "neunzehnte", "neunzehnten", "neunzehnter", "neunzehntes"],
    # 20-29
    "zwanzig": ["zwanzig", "zwanzigste", "zwanzigsten", "zwanzigster", "zwanzigstes"],
    "einundzwanzig": [
        "einundzwanzig",
        "einundzwanzigste",
        "einundzwanzigsten",
        "einundzwanzigster",
        "einundzwanzigstes",
    ],
    "zweiundzwanzig": [
        "zweiundzwanzig",
        "zweiundzwanzigste",
        "zweiundzwanzigsten",
        "zweiundzwanzigster",
        "zweiundzwanzigstes",
    ],
    "dreiundzwanzig": [
        "dreiundzwanzig",
        "dreiundzwanzigste",
        "dreiundzwanzigsten",
        "dreiundzwanzigster",
        "dreiundzwanzigstes",
    ],
    "vierundzwanzig": [
        "vierundzwanzig",
        "vierundzwanzigste",
        "vierundzwanzigsten",
        "vierundzwanzigster",
        "vierundzwanzigstes",
    ],
    "fünfundzwanzig": [
        "fünfundzwanzig",
        "fünfundzwanzigste",
        "fünfundzwanzigsten",
        "fünfundzwanzigster",
        "fünfundzwanzigstes",
    ],
    "sechsundzwanzig": [
        "sechsundzwanzig",
        "sechsundzwanzigste",
        "sechsundzwanzigsten",
        "sechsundzwanzigster",
        "sechsundzwanzigstes",
    ],
    "siebenundzwanzig": [
        "siebenundzwanzig",
        "siebenundzwanzigste",
        "siebenundzwanzigsten",
        "siebenundzwanzigster",
        "siebenundzwanzigstes",
    ],
    "achtundzwanzig": [
        "achtundzwanzig",
        "achtundzwanzigste",
        "achtundzwanzigsten",
        "achtundzwanzigster",
        "achtundzwanzigstes",
    ],
    "neunundzwanzig": [
        "neunundzwanzig",
        "neunundzwanzigste",
        "neunundzwanzigsten",
        "neunundzwanzigster",
        "neunundzwanzigstes",
    ],
    # 30-39
    "dreißig": ["dreißig", "dreißigste", "dreißigsten", "dreißigster", "dreißigstes"],
    "einunddreißig": [
        "einunddreißig",
        "einunddreißigste",
        "einunddreißigsten",
        "einunddreißigster",
        "einunddreißigstes",
    ],
    "zweiunddreißig": [
        "zweiunddreißig",
        "zweiunddreißigste",
        "zweiunddreißigsten",
        "zweiunddreißigster",
        "zweiunddreißigstes",
    ],
    "dreiunddreißig": [
        "dreiunddreißig",
        "dreiunddreißigste",
        "dreiunddreißigsten",
        "dreiunddreißigster",
        "dreiunddreißigstes",
    ],
    "vierunddreißig": [
        "vierunddreißig",
        "vierunddreißigste",
        "vierunddreißigsten",
        "vierunddreißigster",
        "vierunddreißigstes",
    ],
    "fünfunddreißig": [
        "fünfunddreißig",
        "fünfunddreißigste",
        "fünfunddreißigsten",
        "fünfunddreißigster",
        "fünfunddreißigstes",
    ],
    "sechsunddreißig": [
        "sechsunddreißig",
        "sechsunddreißigste",
        "sechsunddreißigsten",
        "sechsunddreißigster",
        "sechsunddreißigstes",
    ],
    "siebenunddreißig": [
        "siebenunddreißig",
        "siebenunddreißigste",
        "siebenunddreißigsten",
        "siebenunddreißigster",
        "siebenunddreißigstes",
    ],
    "achtunddreißig": [
        "achtunddreißig",
        "achtunddreißigste",
        "achtunddreißigsten",
        "achtunddreißigster",
        "achtunddreißigstes",
    ],
    "neununddreißig": [
        "neununddreißig",
        "neununddreißigste",
        "neununddreißigsten",
        "neununddreißigster",
        "neununddreißigstes",
    ],
    # 40-49
    "vierzig": ["vierzig", "vierzigste", "vierzigsten", "vierzigster", "vierzigstes"],
    "einundvierzig": [
        "einundvierzig",
        "einundvierzigste",
        "einundvierzigsten",
        "einundvierzigster",
        "einundvierzigstes",
    ],
    "zweiundvierzig": [
        "zweiundvierzig",
        "zweiundvierzigste",
        "zweiundvierzigsten",
        "zweiundvierzigster",
        "zweiundvierzigstes",
    ],
    "dreiundvierzig": [
        "dreiundvierzig",
        "dreiundvierzigste",
        "dreiundvierzigsten",
        "dreiundvierzigster",
        "dreiundvierzigstes",
    ],
    "vierundvierzig": [
        "vierundvierzig",
        "vierundvierzigste",
        "vierundvierzigsten",
        "vierundvierzigster",
        "vierundvierzigstes",
    ],
    "fünfundvierzig": [
        "fünfundvierzig",
        "fünfundvierzigste",
        "fünfundvierzigsten",
        "fünfundvierzigster",
        "fünfundvierzigstes",
    ],
    "sechsundvierzig": [
        "sechsundvierzig",
        "sechsundvierzigste",
        "sechsundvierzigsten",
        "sechsundvierzigster",
        "sechsundvierzigstes",
    ],
    "siebenundvierzig": [
        "siebenundvierzig",
        "siebenundvierzigste",
        "siebenundvierzigsten",
        "siebenundvierzigster",
        "siebenundvierzigstes",
    ],
    "achtundvierzig": [
        "achtundvierzig",
        "achtundvierzigste",
        "achtundvierzigsten",
        "achtundvierzigster",
        "achtundvierzigstes",
    ],
    "neunundvierzig": [
        "neunundvierzig",
        "neunundvierzigste",
        "neunundvierzigsten",
        "neunundvierzigster",
        "neunundvierzigstes",
    ],
    # 50
    "fünfzig": ["fünfzig", "fünfzigste", "fünfzigsten", "fünfzigster", "fünfzigstes"],
}


# Flatten to case-insensitive set for fast O(1) lookup
def _build_lookup_set() -> set:
    """
    Builds the lookup set with all inflected forms (without exceptions).
    Created lazily on first call.
    """
    lookup = set()
    for flexionen_liste in ZAHLWOERTER_0_50_MIT_FLEXIONEN.values():
        for wort in flexionen_liste:
            lookup.add(wort.lower())

    # Remove ambiguous words from lookup (defined in config.NUMBER_WORDS_0_50_EXCEPTIONS)
    # These words are only handled by the BERT model
    for ausnahme in config.NUMBER_WORDS_0_50_EXCEPTIONS:
        # Remove all inflected forms of the exception
        if ausnahme in ZAHLWOERTER_0_50_MIT_FLEXIONEN:
            for flexion in ZAHLWOERTER_0_50_MIT_FLEXIONEN[ausnahme]:
                lookup.discard(flexion.lower())

    return lookup


# Lazy-initialized lookup set
_zahlwoerter_lookup_cache: Optional[set] = None


def _get_zahlwoerter_lookup() -> set:
    """Returns the lookup set (lazy-initialized)."""
    global _zahlwoerter_lookup_cache
    if _zahlwoerter_lookup_cache is None:
        _zahlwoerter_lookup_cache = _build_lookup_set()
    return _zahlwoerter_lookup_cache


# ============================================================================
# ML MODEL: Thread-safe lazy loading via BaseMLRule
# ============================================================================


def _get_device_id() -> int:
    """
    Determines the device ID for pipeline loading.

    Returns:
        int: 0 for GPU/MPS, -1 for CPU
    """
    if config.DEVICE_PREFERENCE == "cpu":
        return -1

    # Auto-detection or explicit device
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        return 0

    return -1


class NumberWordsModel(BaseMLRule):
    """Thread-safe lazy loader for the number word BERT pipeline."""

    @classmethod
    def _load_model(cls) -> Any:
        """Load BERT pipeline from local model, HuggingFace, or legacy paths.

        Tries in order:
        1. Local model directory (regeln/zahlwoerter/model/) -- priority after retraining
        2. Hugging Face Model Hub
        3. Legacy local paths from config.LOCAL_MODEL_PATHS

        Returns:
            Pipeline object for token-classification

        Raises:
            RuntimeError: If no source provides a working pipeline
        """
        device_id = _get_device_id()

        # Attempt 1: Local model directory (takes priority after retraining)
        local_model_dir = Path(__file__).parent / "model"
        if local_model_dir.exists():
            # Check for actual model files, not just .staging/.backup dirs
            has_model = any(
                (local_model_dir / f).exists()
                for f in ("model.safetensors", "pytorch_model.bin", "config.json")
            )
            if has_model:
                try:
                    logger.info(" Loading BERT pipeline from local model: %s", local_model_dir)
                    nlp = pipeline(
                        "token-classification",
                        model=str(local_model_dir),
                        tokenizer=str(local_model_dir),
                        aggregation_strategy="simple",
                        device=device_id,
                    )
                    logger.info(" BERT pipeline loaded from local model: %s", local_model_dir)
                    return nlp
                except Exception as e:
                    logger.info(" Local model load failed: %s", e)

        # Attempt 2: Hugging Face
        model_id = config.HUGGINGFACE_MODEL_ID
        if model_id != "YOUR-USERNAME/leichte-sprache-zahlwoerter":
            try:
                logger.info(" Lade BERT-Pipeline von Hugging Face: %s", model_id)
                device_name = "GPU/MPS" if device_id == 0 else "CPU"
                logger.info(" Device: %s", device_name)

                nlp = pipeline(
                    "token-classification",
                    model=model_id,
                    tokenizer=model_id,
                    aggregation_strategy="simple",
                    device=device_id,
                )

                logger.info(" BERT-Pipeline erfolgreich geladen von Hugging Face")
                return nlp
            except Exception as e:
                logger.info(" Hugging Face Download fehlgeschlagen: %s", e)

        # Attempt 3: Legacy local paths (fallback)
        for local_path_str in config.LOCAL_MODEL_PATHS:
            local_path = Path(local_path_str).expanduser().resolve()

            if not local_path.exists():
                continue

            try:
                logger.info(" Versuche lokale Pipeline zu laden: %s", local_path)
                device_name = "GPU/MPS" if device_id == 0 else "CPU"
                logger.info(" Device: %s", device_name)

                nlp = pipeline(
                    "token-classification",
                    model=str(local_path),
                    tokenizer=str(local_path),
                    aggregation_strategy="simple",
                    device=device_id,
                )

                logger.info(" BERT-Pipeline erfolgreich geladen von: %s", local_path)
                return nlp
            except Exception as e:
                logger.error(" Fehler beim Laden von %s: %s", local_path, e)
                continue

        # All sources failed
        raise RuntimeError(
            "BERT-Pipeline konnte nicht geladen werden!\n"
            "\n"
            "Moegliche Loesungen:\n"
            "1. Hugging Face Upload durchfuehren (siehe HUGGINGFACE_UPLOAD_ANLEITUNG.md)\n"
            "2. Model ID in config.py anpassen: HUGGINGFACE_MODEL_ID = 'username/model-name'\n"
            "3. Lokales Modell unter einem der folgenden Pfade platzieren:\n"
            + "\n".join(f"   - {p}" for p in config.LOCAL_MODEL_PATHS)
        )


# ============================================================================
# POST-PROCESSING FILTER: Intelligent cleanup of BERT results
# ============================================================================


def _clean_bert_output(
    results: List[Dict[str, Any]], text: str
) -> List[Dict[str, Any]]:
    """
    Intelligent filter for BERT pipeline results.
    Removes artifacts, fragments, and uncertain matches.

    Filter steps:
    1. Score filter: discard uncertain (< CONFIDENCE_THRESHOLD)
    2. Label filter: ignore "O"
    3. Hash filter: remove tokenizer markers (##)
    4. Ghost filter: blacklist check for suffixes (iges, sten, ten, etc.)
    5. Single-char filter: single characters (except digits)
    6. Hyphen context check: prevents "50-jaehriges" -> "iges"

    Args:
        results: Raw results from BERT pipeline
        text: Original text for context checks

    Returns:
        Filtered list of detections
    """
    filtered = []

    for entity in results:
        word = entity["word"]
        score = entity["score"]
        group = entity["entity_group"]
        start = entity["start"]
        entity["end"]

        # 1. SCORE FILTER: Discard uncertain results
        if score < config.CONFIDENCE_THRESHOLD:
            continue

        # 2. LABEL FILTER: Ignore "O"
        if group == "O":
            continue

        # 3. HASH FILTER: Remove internal tokenizer markers (e.g. ##iges)
        # Even with aggregation_strategy="simple", remnants may remain
        clean_word = word.replace("##", "")

        # 4. GHOST FILTER: Suffix blacklist
        # If the detected word is just a suffix (e.g. "iges"), discard it
        if clean_word.lower() in config.GHOST_ARTIFACTS:
            continue

        # 5. SINGLE-CHAR-FILTER
        # Einzelne Buchstaben sind meist Rauschen (außer es sind Ziffern)
        if len(clean_word) < config.MIN_WORD_LENGTH and not clean_word.isdigit():
            continue

        # 6. KONTEXT-CHECK: Bindestrich-Suffix-Filter
        # Prüfen, ob direkt vor dem Fundstück ein Bindestrich steht (z.B. bei 50-jähriges)
        # Wenn ja, und das Fundstück ist ein Suffix, verwerfen wir es
        if start > 0 and text[start - 1] == "-":
            # Das ist oft ein Indikator für [Ziffer]-[Suffix] → Suffix verwerfen
            # da wir Ziffern-Komposita (50-jähriges) eigentlich erlauben wollen
            continue

        # Wenn alle Tests bestanden: Eintrag übernehmen
        # Wir speichern das bereinigte Wort (ohne ##)
        entity["word"] = clean_word
        filtered.append(entity)

    return filtered


# ============================================================================
# HYBRID FILTER: Regelbasiert + BERT
# ============================================================================


def _rule_based_filter(doc: Doc) -> List[Dict[str, Any]]:
    """
    Regelbasierter Filter für eindeutige Zahlwörter (0-50).

    Erkennt:
    - Alle Flexionsformen von 0-50
    - Case-insensitive

    Ausnahmen:
    - "elf", "acht" (mehrdeutig, in config.NUMBER_WORDS_0_50_EXCEPTIONS)
    - "ein/eine/einer" (nicht in ZAHLWOERTER_0_50_MIT_FLEXIONEN)
    - Komposita (automatisch durch exaktes Token-Matching)

    Returns:
        Liste von Dicts mit: {
            'word': str,      # Erkanntes Wort (Original-Schreibweise)
            'start': int,     # Start-Position im Text
            'end': int,       # End-Position im Text
            'source': 'rule'  # Quelle: regelbasiert
        }
    """
    if not config.ENABLE_RULE_BASED_FILTER:
        return []

    errors = []
    lookup = _get_zahlwoerter_lookup()

    for token in doc:
        # Skip punctuation and whitespace
        if token.is_punct or token.is_space:
            continue

        token_lower = token.text.lower()

        # Check if token exactly matches a number word (case-insensitive)
        # Compounds are automatically excluded since "Zweifamilienhaus" != "zwei"
        if token_lower in lookup:
            errors.append(
                {
                    "word": token.text,  # Original-Schreibweise erhalten
                    "start": token.idx,
                    "end": token.idx + len(token.text),
                    "source": "rule",
                }
            )

    return errors


def _bert_analysis(doc: Doc) -> List[Dict[str, Any]]:
    """
    BERT-basierte Analyse mit Pipeline und Post-Processing.

    Verwendet transformers.pipeline mit aggregation_strategy="simple" für
    automatische Subword-Token-Aggregation + intelligente Filter zur
    Vermeidung von Ghost-Artifacts und False Positives.

    Erkennt:
    - Zahlen über 50
    - Jahreszahlen (z.B. "neunzehnhundertfünfundachtzig")
    - Prozentangaben (z.B. "fünfzig Prozent")
    - Komplexe Zahlen (z.B. "dreiundzwanzig")
    - Kontextabhängige Fälle (z.B. "ein" als Zahlwort vs. Artikel)

    Post-Processing Filter:
    - Score-Filter (< 0.60)
    - Ghost-Artifacts-Filter (iges, sten, ten, etc.)
    - Bindestrich-Kontext-Check (50-jähriges → kein "iges")
    - Single-Char-Filter
    - Hash-Filter (##)

    Returns:
        Liste von Dicts mit: {
            'word': str,         # Erkanntes Wort (bereinigt)
            'start': int,        # Start-Position
            'end': int,          # End-Position
            'entity_group': str, # BERT-Label (BAD_WORD_NUM, etc.) - verwendet von Pipeline
            'score': float,      # Confidence-Score
            'source': 'bert'     # Quelle: BERT-Pipeline
        }
    """
    if not config.ENABLE_BERT_MODEL:
        return []

    # Pipeline laden via BaseMLRule (thread-sicher, lazy, gecacht)
    nlp = NumberWordsModel.get_model()

    if nlp is None:
        return []

    text = doc.text

    if not text.strip():
        return []

    try:
        # Pipeline-Inference (Subword-Aggregation automatisch)
        raw_results = nlp(text)

        # Post-Processing: Intelligente Filterung
        clean_results = _clean_bert_output(raw_results, text)

        # Konvertiere Pipeline-Format zu internem Format + füge 'source' hinzu
        filtered_errors = []
        for entity in clean_results:
            filtered_errors.append(
                {
                    "word": entity["word"],
                    "start": entity["start"],
                    "end": entity["end"],
                    "label": entity[
                        "entity_group"
                    ],  # Pipeline verwendet 'entity_group'
                    "score": entity["score"],
                    "source": "bert",
                }
            )

        return filtered_errors

    except Exception as e:
        logger.error(" Fehler bei BERT-Pipeline-Analyse: %s", e)
        return []


def _regex_filter(doc: Doc) -> List[Dict[str, Any]]:
    """
    Regex-basierter Filter für römische Zahlen.

    Erkennt Muster wie:
    - I, II, III, IV, V, VI, VII, VIII, IX, X
    - XIV., XVI., XX., etc.
    - Komplexe Formen: MCMXCV (1995), MMXXV (2025)

    Returns:
        Liste von Dicts mit: {
            'word': str,        # Erkannte römische Zahl
            'start': int,       # Start-Position
            'end': int,         # End-Position
            'label': str,       # 'REGEX_ROMAN'
            'source': 'regex'   # Quelle: Regex-Filter
        }
    """
    if not config.ENABLE_REGEX_FILTER:
        return []

    text = doc.text
    errors = []

    for rule in config.REGEX_RULES:
        for match in rule["pattern"].finditer(text):
            errors.append(
                {
                    "word": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "label": rule["label"],
                    "source": "regex",
                }
            )

    return errors


def _merge_all_errors(
    rule_errors: List[Dict[str, Any]],
    bert_errors: List[Dict[str, Any]],
    regex_errors: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merged Fehler von 3 Quellen mit Deduplizierung.

    Quellen:
    1. Regelbasierter Filter (0-50, eindeutige Fälle)
    2. BERT-Pipeline (komplexe Zahlen, >50, Kontext)
    3. Regex-Filter (römische Zahlen)

    Deduplication-Strategie:
    - Bei 50%+ Überlappung der Positionen: Als Duplikat behandeln
    - Priorität: BERT > Regex > Regel (BERT hat höchste Accuracy)

    Args:
        rule_errors: Fehler vom regelbasierten Filter
        bert_errors: Fehler vom BERT-Modell
        regex_errors: Fehler vom Regex-Filter

    Returns:
        Merged und deduplizierte Fehlerliste
    """
    # Sammle alle Fehler mit Source-Tag
    all_errors = []
    all_errors.extend(rule_errors)
    all_errors.extend(bert_errors)
    all_errors.extend(regex_errors)

    if not all_errors:
        return []

    # Sortiere nach Position
    all_errors.sort(key=lambda x: x["start"])

    # Deduplizierung: Finde überlappende Fehler
    deduplicated = []
    used_indices = set()

    for i, error in enumerate(all_errors):
        if i in used_indices:
            continue

        # Finde alle überlappenden Fehler
        overlapping = [error]
        overlapping_indices = [i]

        for j in range(i + 1, len(all_errors)):
            if j in used_indices:
                continue

            other = all_errors[j]

            # Berechne Überlappung
            overlap_start = max(error["start"], other["start"])
            overlap_end = min(error["end"], other["end"])
            overlap_len = max(0, overlap_end - overlap_start)

            # Überlappungs-Ratio
            min_len = min(error["end"] - error["start"], other["end"] - other["start"])
            if min_len > 0 and overlap_len > 0:
                overlap_ratio = overlap_len / min_len

                if overlap_ratio >= 0.5:
                    # Überlappung gefunden
                    overlapping.append(other)
                    overlapping_indices.append(j)

        # Priorität: BERT > Regex > Regel
        priority_order = {"bert": 3, "regex": 2, "rule": 1}
        best_error = max(overlapping, key=lambda x: priority_order.get(x["source"], 0))

        deduplicated.append(best_error)
        used_indices.update(overlapping_indices)

    return deduplicated


def check_rule(doc: Doc) -> List[str]:
    """
    Main function: checks text for written-out numbers with 3-filter architecture.

    Called by the Leichte Sprache system; implements the
    `check_rule(doc: spacy.tokens.Doc) -> List[str]` interface.

    3-filter architecture:
    1. Rule-based filter: fast detection of unambiguous number words (0-50)
    2. BERT pipeline: context-aware detection (>50, years, context) + post-processing
    3. Regex filter: Roman numerals (I, II, III, IV, XIV, MCMXCV, etc.)
    4. Merge + deduplication: combines all 3 results (priority: BERT > Regex > Rule)

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of error messages (strings) in format:
        'Zahlwort "zwei" sollte als Ziffer geschrieben werden. Text-Position: 18-22'
    """
    # Empty text? No errors
    if not doc.text.strip():
        return []

    # Step 1: Rule-based filter (0-50, unambiguous cases)
    rule_errors = _rule_based_filter(doc)

    # Step 2: BERT pipeline analysis (complex cases, >50, context) + post-processing
    bert_errors = _bert_analysis(doc)

    # Step 3: Regex filter (Roman numerals)
    regex_errors = _regex_filter(doc)

    # Step 4: Merge + deduplication (3 sources)
    merged_errors = _merge_all_errors(rule_errors, bert_errors, regex_errors)

    # Step 5: Generate error messages
    error_messages = []

    for error in merged_errors:
        # Description: use extended label descriptions (incl. regex)
        if "label" in error:
            description = config.LABEL_DESCRIPTIONS_EXTENDED.get(
                error["label"], "Zahlwort sollte als Ziffer geschrieben werden"
            )
            label = error["label"]
        else:
            # Rule error: default description
            description = "Zahl als Wort geschrieben"
            label = "BAD_WORD_NUM"

        # Format error message with template
        message = config.ERROR_MESSAGE_TEMPLATE.format(
            word=error["word"],
            label=label,
            description=description,
            start=error["start"],
            end=error["end"],
        )

        error_messages.append(message)

    return error_messages


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    """
    Standalone-Test der Regel.

    Verwendung:
        python regeln/zahlwoerter/regel.py
    """
    import spacy

    logger.info("=" * 70)
    logger.info("BERT-basierte Zahlwörter-Regel - Standalone Test")
    logger.info("=" * 70)

    # spaCy-Modell laden
    try:
        nlp = spacy.load("de_core_news_lg")
    except OSError:
        logger.error(" spaCy-Modell 'de_core_news_lg' nicht gefunden!")
        logger.error(" Bitte installieren: python -m spacy download de_core_news_lg")
        exit(1)

    # Test-Texte
    test_texte = [
        "Das Kind ist acht Jahre alt.",
        "Der Film dauert zwei Stunden.",
        "Es gibt dreiundzwanzig Teilnehmer.",
        "Im Jahr neunzehnhundertfünfundachtzig begann alles.",
        "Der Anteil beträgt fünfzig Prozent.",
        "Bitte bringen Sie 2 Formulare mit.",  # Keine Fehler (Ziffer OK)
        "Die Veranstaltung findet am 15. März statt.",  # Keine Fehler
    ]

    logger.info("\nAnalysiere Test-Texte:\n")

    for i, text in enumerate(test_texte, 1):
        logger.info("Text %s: %s", i, text)
        doc = nlp(text)
        results = check_rule(doc)

        if results:
            for error in results:
                logger.error(" %s", error)
        else:
            logger.error(" Keine Fehler gefunden")

        logger.info("")

    logger.info("=" * 70)
