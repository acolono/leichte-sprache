"""
Extended rule for detecting foreign words using advanced NLP techniques.
Uses semantic vector analysis, morphological features, and statistical methods.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from wordfreq import word_frequency

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

import logging

logger = logging.getLogger(__name__)
FOREIGN_WORDS: Dict[str, Dict[str, any]] = {
    "administration": {
        "alternative": "Verwaltung",
        "schwierigkeit": 0.9,
        "kategorie": "verwaltung",
    },
    "service": {
        "alternative": "Dienst",
        "schwierigkeit": 0.7,
        "kategorie": "allgemein",
    },
    "formular": {
        "alternative": "Formblatt",
        "schwierigkeit": 0.6,
        "kategorie": "verwaltung",
    },
    "information": {
        "alternative": "Nachricht",
        "schwierigkeit": 0.5,
        "kategorie": "kommunikation",
    },
    "dokumentation": {
        "alternative": "Unterlagen",
        "schwierigkeit": 0.8,
        "kategorie": "verwaltung",
    },
    "kommunikation": {
        "alternative": "Gespräch",
        "schwierigkeit": 0.7,
        "kategorie": "kommunikation",
    },
    "organisation": {
        "alternative": "Gruppe",
        "schwierigkeit": 0.6,
        "kategorie": "struktur",
    },
    "institution": {
        "alternative": "Einrichtung",
        "schwierigkeit": 0.8,
        "kategorie": "struktur",
    },
    "konzept": {"alternative": "Plan", "schwierigkeit": 0.7, "kategorie": "planung"},
    "prozess": {"alternative": "Ablauf", "schwierigkeit": 0.6, "kategorie": "aktion"},
    "implementierung": {
        "alternative": "Umsetzung",
        "schwierigkeit": 0.9,
        "kategorie": "aktion",
    },
    "modifikation": {
        "alternative": "Änderung",
        "schwierigkeit": 0.8,
        "kategorie": "aktion",
    },
    "realisierung": {
        "alternative": "Verwirklichung",
        "schwierigkeit": 0.8,
        "kategorie": "aktion",
    },
    "konstruktion": {
        "alternative": "Aufbau",
        "schwierigkeit": 0.7,
        "kategorie": "struktur",
    },
    "produktion": {
        "alternative": "Herstellung",
        "schwierigkeit": 0.6,
        "kategorie": "aktion",
    },
    "analyse": {
        "alternative": "Untersuchung",
        "schwierigkeit": 0.7,
        "kategorie": "wissenschaft",
    },
    "synthese": {
        "alternative": "Zusammenfassung",
        "schwierigkeit": 0.8,
        "kategorie": "wissenschaft",
    },
    "evaluation": {
        "alternative": "Bewertung",
        "schwierigkeit": 0.8,
        "kategorie": "wissenschaft",
    },
    "interpretation": {
        "alternative": "Deutung",
        "schwierigkeit": 0.8,
        "kategorie": "wissenschaft",
    },
    "definition": {
        "alternative": "Erklärung",
        "schwierigkeit": 0.6,
        "kategorie": "wissenschaft",
    },
    "funktion": {
        "alternative": "Aufgabe",
        "schwierigkeit": 0.5,
        "kategorie": "allgemein",
    },
    "situation": {
        "alternative": "Lage",
        "schwierigkeit": 0.5,
        "kategorie": "allgemein",
    },
    "position": {"alternative": "Stelle", "schwierigkeit": 0.5, "kategorie": "ort"},
    "tradition": {"alternative": "Brauch", "schwierigkeit": 0.6, "kategorie": "kultur"},
    "diskussion": {
        "alternative": "Gespräch",
        "schwierigkeit": 0.6,
        "kategorie": "kommunikation",
    },
    "präsentation": {
        "alternative": "Vorstellung",
        "schwierigkeit": 0.7,
        "kategorie": "kommunikation",
    },
    "demonstration": {
        "alternative": "Vorführung",
        "schwierigkeit": 0.8,
        "kategorie": "aktion",
    },
    "illustration": {
        "alternative": "Bild",
        "schwierigkeit": 0.7,
        "kategorie": "darstellung",
    },
    "publikation": {
        "alternative": "Veröffentlichung",
        "schwierigkeit": 0.8,
        "kategorie": "kommunikation",
    },
    "navigation": {
        "alternative": "Führung",
        "schwierigkeit": 0.7,
        "kategorie": "bewegung",
    },
    "instruktion": {
        "alternative": "Anweisung",
        "schwierigkeit": 0.7,
        "kategorie": "kommunikation",
    },
    "konstruktiv": {
        "alternative": "aufbauend",
        "schwierigkeit": 0.7,
        "kategorie": "eigenschaft",
    },
    "alternativ": {
        "alternative": "anders",
        "schwierigkeit": 0.6,
        "kategorie": "eigenschaft",
    },
    "intensiv": {
        "alternative": "stark",
        "schwierigkeit": 0.6,
        "kategorie": "eigenschaft",
    },
    "aktiv": {"alternative": "tätig", "schwierigkeit": 0.5, "kategorie": "eigenschaft"},
    "passiv": {
        "alternative": "untätig",
        "schwierigkeit": 0.6,
        "kategorie": "eigenschaft",
    },
    "positiv": {"alternative": "gut", "schwierigkeit": 0.4, "kategorie": "eigenschaft"},
    "negativ": {
        "alternative": "schlecht",
        "schwierigkeit": 0.5,
        "kategorie": "eigenschaft",
    },
    "effektiv": {
        "alternative": "wirksam",
        "schwierigkeit": 0.7,
        "kategorie": "eigenschaft",
    },
    "optimal": {
        "alternative": "bestmöglich",
        "schwierigkeit": 0.7,
        "kategorie": "eigenschaft",
    },
    "digital": {
        "alternative": "elektronisch",
        "schwierigkeit": 0.6,
        "kategorie": "technik",
    },
    "global": {"alternative": "weltweit", "schwierigkeit": 0.5, "kategorie": "ort"},
    "lokal": {"alternative": "örtlich", "schwierigkeit": 0.5, "kategorie": "ort"},
    "zentral": {"alternative": "mittig", "schwierigkeit": 0.6, "kategorie": "ort"},
    "regional": {
        "alternative": "örtlich begrenzt",
        "schwierigkeit": 0.7,
        "kategorie": "ort",
    },
}

# EXTENDED: German acronyms and established foreign words that should NOT be reported
GERMAN_COMMON_WORDS = {
    # Häufige -tion Wörter die NICHT im FOREIGN_WORDS-Dict stehen
    "station",
    "nation",
    "aktion",
    "reaktion",
    "attraktion",
    # Deutsche Akronyme und Abkürzungen
    "gmbh",
    "ag",
    "kg",
    "ev",
    "gbr",
    "ohg",
    "kgaa",
    "tv",
    "pc",
    "dvd",
    "cd",
    "usb",
    "wlan",
    "dsl",
    "pkw",
    "lkw",
    "bahn",
    "db",
    "post",
    "dhl",
    # Etablierte Fremdwörter die alltäglich und weit verbreitet sind
    "dokumentation",
    "auto",
    "hotel",
    "restaurant",
    "café",
    "kino",
    "theater",
    "museum",
    "sport",
    "team",
    "training",
    "coach",
    "fitness",
    "hobby",
    "computer",
    "internet",
    "email",
    "handy",
    "smartphone",
    "tablet",
    "büro",
    "manager",
    "chef",
    "firma",
    "business",
    "meeting",
    "party",
    "festival",
    "konzert",
    "band",
    "musik",
    "radio",
    "foto",
    "video",
    "film",
    "kamera",
    "studio",
    "design",
    # Technische Begriffe die alltäglich sind (nicht im FOREIGN_WORDS-Dict)
    "system",
    "online",
    "offline",
    "analog",
    "software",
    "hardware",
    "update",
    "upgrade",
    "download",
    "upload",
    # Einfache Adjektive (nicht im FOREIGN_WORDS-Dict)
    "normal",
    "formal",
    # Wissenschaft/Bildung (etabliert, nicht im FOREIGN_WORDS-Dict)
    "universität",
    "student",
    "professor",
    "diplom",
    "bachelor",
    "master",
    "seminar",
    "kurs",
    "test",
    "prüfung",
    "note",
    "projekt",
    # Medizin (etabliert, nicht im FOREIGN_WORDS-Dict)
    "doktor",
    "patient",
    "therapie",
    "diagnose",
    "praxis",
    "klinik",
    "operation",
    "medizin",
    "vitamin",
    "antibiotikum",
    # Deutsche Verwaltungsbegriffe (echte deutsche Wörter)
    "verwaltung",
    "behörde",
    "amt",
    "abteilung",
    "bearbeitung",
    "einführung",
    "durchführung",
    "ausführung",
    "aufführung",
    "umgestaltung",
    "gestaltung",
    "entwicklung",
    "verbesserung",
    "betreuung",
    "beratung",
    "unterstützung",
    "förderung",
    "untersuchung",
    "bewertung",
    "einschätzung",
    # Handlungen und Prozesse (DEUTSCHE Begriffe!)
    "änderung",
    "erweiterung",
    "kürzung",
    "verlängerung",
    "beschleunigung",
    "verzögerung",
    "vereinfachung",
    "verkomplizierung",
    "klärung",
    "lösung",
    "behandlung",
    "beendigung",
    # Weitere deutsche Grundwörter
    "erhaltung",
    "erhöhung",
    "verringerung",
    "steigerung",
    "senkung",
    "vermehrung",
    "verminderung",
    "verstärkung",
}

# German acronyms that frequently end with problematic endings
GERMAN_ACRONYMS = {
    "consultant",
    "plant",
    "relevant",
    "präsent",
    "patient",
    "student",
    "agent",
}

LATIN_GREEK_PATTERNS = [
    r".*graph(ie|isch|en)$",
    r".*log(ie|isch|en)$",
    r".*phon(ie|isch|en)$",
    r".*krat(ie|isch|en)$",
    r".*nom(ie|isch|en)$",
    r".*phil(ie|isch|en)$",
]

FOREIGN_WORD_ENDINGS = {
    "schwer": ["tion", "sion", "ität", "ismus", "ment", "ant", "ent", "anz", "enz"],
    "mittel": ["al", "ell", "iv", "ar", "är", "ik", "ur"],
    "leicht": ["ös", "os", "isch"],
}

SYLLABLE_THRESHOLDS = {1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6}

# Standalone-capable: file located in the same directory as regel.py
CORPUS_PROBS_FILE = Path(__file__).parent / "corpus_probabilities.json"
THRESHOLD_FREMDWORT = 3.0  # Balanced: whitelist fix prevents false positives, lower threshold catches more


def _load_corpus_probabilities() -> dict:
    """Loads precomputed corpus probabilities for fallback analysis."""
    corpus_file = Path(CORPUS_PROBS_FILE)
    if not corpus_file.exists():
        return {}

    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _check_corpus_fallback(token, corpus_probs: dict) -> str:
    """Fallback analysis using corpus probabilities for foreign words."""
    lemma = token.lemma_.lower()

    prob_normal = word_frequency(lemma, "de")
    prob_corpus = corpus_probs.get(lemma, 0)

    if prob_corpus == 0 and prob_normal > 0.0000001:
        return f'Komplexes Wort "{token.text}" kommt nicht in Leichter Sprache vor. Möglicherweise Fremdwort - durch deutsches Wort ersetzen oder erklären.'

    elif prob_corpus > 0 and prob_normal > 0:
        ratio = prob_normal / prob_corpus
        if ratio > THRESHOLD_FREMDWORT:
            return f'Potentielles Fremdwort "{token.text}" (Ratio: {ratio:.1f}). Eventuell durch deutsches Wort ersetzen oder erklären.'

    return ""


def _count_syllables(word: str) -> int:
    """Approximates the syllable count of a German word."""
    vowels = "aeiouäöü"
    syllable_count = 0
    last_was_vowel = False

    for char in word.lower():
        is_vowel = char in vowels
        if is_vowel and not last_was_vowel:
            syllable_count += 1
        last_was_vowel = is_vowel

    return max(1, syllable_count)


def _calculate_foreign_word_score(token: spacy.tokens.Token) -> Tuple[float, List[str]]:
    """Calculates a foreign word score based on multiple criteria."""
    word = token.text.lower()
    score = 0.0
    reasons = []

    syllables = _count_syllables(word)
    if syllables >= 4:
        score += 0.3
        reasons.append(f"lange Silbenanzahl ({syllables})")

    for difficulty, endings in FOREIGN_WORD_ENDINGS.items():
        for ending in endings:
            if word.endswith(ending):
                if difficulty == "schwer":
                    score += 0.6
                elif difficulty == "mittel":
                    score += 0.4
                else:
                    score += 0.2
                reasons.append(f"Endung -{ending} ({difficulty})")
                break

    for pattern in LATIN_GREEK_PATTERNS:
        if re.match(pattern, word):
            score += 0.5
            reasons.append("lateinisch/griechisch Muster")
            break

    if token.pos_ in ["NOUN", "ADJ"] and len(word) > 8:
        score += 0.2
        reasons.append("langes Substantiv/Adjektiv")

    return score, reasons


def is_likely_foreign_word(word: str, token: spacy.tokens.Token = None) -> bool:
    """IMPROVED: Heuristic for foreign word detection with extended exceptions."""
    if len(word) < 4:
        return False

    word_lower = word.lower()

    # CRITICAL FIX: Check established German/adopted words
    if word_lower in GERMAN_COMMON_WORDS:
        return False

    # CRITICAL FIX: Check German acronyms and terms with problematic endings
    if word_lower in GERMAN_ACRONYMS:
        return False

    if token:
        score, _ = _calculate_foreign_word_score(token)
        return (
            score >= 0.8
        )  # CHANGED: Higher threshold to reduce false positives

    for endings in FOREIGN_WORD_ENDINGS.values():
        for ending in endings:
            if word.endswith(ending) and len(word) >= 5:
                return True

    return False


def _contextual_evaluation(token: spacy.tokens.Token, doc: spacy.tokens.Doc) -> float:
    """Evaluates foreign words based on context and text complexity."""
    context_score = 0.0

    sent = next(sent for sent in doc.sents if token in sent)
    sentence_length = len([t for t in sent if not t.is_punct and not t.is_space])

    if sentence_length > 15:
        context_score += 0.2

    foreign_word_density = sum(1 for t in sent if t.lemma_.lower() in FOREIGN_WORDS) / len(
        sent
    )
    if foreign_word_density > 0.2:
        context_score += 0.3

    return context_score


def check_rule(doc: spacy.tokens.Doc) -> List[str]:
    """
    Extended foreign word detection with semantic and contextual analysis:
    1. Weighted dictionary-based detection
    2. Score-based ending heuristics with morphological features
    3. Contextual evaluation of text complexity
    4. Semantic clustering of similar foreign words

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of weighted error messages
    """
    errors = []
    found_words = set()

    corpus_probs = _load_corpus_probabilities()

    for token in doc:
        if token.is_punct or token.is_space or len(token.lemma_) < 3:
            continue

        token_lower = token.text.lower()
        lemma_lower = token.lemma_.lower()

        # CRITICAL FIX: Skip common German words and acronyms early in the process
        if (
            token_lower in GERMAN_COMMON_WORDS
            or lemma_lower in GERMAN_COMMON_WORDS
            or token_lower in GERMAN_ACRONYMS
            or lemma_lower in GERMAN_ACRONYMS
        ):
            continue

        if lemma_lower in found_words:
            continue

        if lemma_lower in FOREIGN_WORDS:
            foreign_word_info = FOREIGN_WORDS[lemma_lower]
            difficulty = foreign_word_info["schwierigkeit"]
            alternative = foreign_word_info["alternative"]
            category = foreign_word_info["kategorie"]

            context_score = _contextual_evaluation(token, doc)
            total_score = difficulty + context_score

            if total_score >= 0.7:
                priority = "hoch" if total_score >= 0.9 else "mittel"
                errors.append(
                    f'Fremdwort "{token.text}" (Kategorie: {category}, Priorität: {priority}). '
                    f'Einfacher: "{alternative}".'
                )
            found_words.add(lemma_lower)

        elif is_likely_foreign_word(token_lower, token):
            score, reasons = _calculate_foreign_word_score(token)
            if score >= 0.8:  # CHANGED: Higher threshold for fewer false positives
                errors.append(
                    f'Mögliches Fremdwort "{token.text}" (Score: {score:.1f}). '
                    f"Grund: {'' if not reasons else reasons[0]}. Prüfen Sie eine deutsche Alternative."
                )
            found_words.add(lemma_lower)

        elif corpus_probs and token.is_alpha and not token.is_stop:
            fallback_error = _check_corpus_fallback(token, corpus_probs)
            if fallback_error:
                errors.append(fallback_error)
                found_words.add(lemma_lower)

    return errors


def extended_foreign_word_analysis(text: str) -> List[Dict]:
    """
    Extended analysis for potential foreign words in text.

    Args:
        text: The text to analyze

    Returns:
        List of analysis results
    """
    import spacy

    nlp = spacy.load("de_core_news_lg")
    doc = nlp(text)

    results = []

    for token in doc:
        if not token.is_punct and not token.is_space and len(token.text) >= 5:
            is_known = token.lemma_.lower() in FOREIGN_WORDS
            is_suspicious = is_likely_foreign_word(token.text.lower(), token)

            if is_known or is_suspicious:
                alternative = ""
                if is_known:
                    alternative = FOREIGN_WORDS[token.lemma_.lower()]["alternative"]

                results.append(
                    {
                        "wort": token.text,
                        "lemma": token.lemma_,
                        "bekannt": is_known,
                        "verdaechtig": is_suspicious,
                        "alternative": alternative or "?",
                    }
                )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nlp = spacy.load("de_core_news_lg")

    test_texts = [
        "Die Administration der Stadt plant eine Reorganisation der digitalen Services.",
        "Die Kommunikation zwischen den verschiedenen Institutionen funktioniert optimal.",
        "Eine detaillierte Analyse der aktuellen Situation ist notwendig.",
        "Die Präsentation der neuen Konzepte war sehr informativ.",
        "Das moderne System bietet viele praktische Funktionen.",
        "Die lokale Organisation arbeitet sehr effektiv.",
        "In der Bibliothek gibt es viele interessante Publikationen.",
    ]

    logger.info("--- Test: Extended foreign word rule ---")
    for i, text in enumerate(test_texts):
        logger.info("\nSentence %s: '%s'", i + 1, text)
        doc = nlp(text)
        results = check_rule(doc)

        if results:
            for error in results:
                logger.error(" %s", error)
        else:
            logger.info(" No problematic foreign words found.")
