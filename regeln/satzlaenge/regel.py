"""
Extended rule for checking sentence length using advanced NLP techniques.
Uses syntactic analysis, subclause detection, and contextual complexity evaluation.
"""

import re
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import spacy
from spacy.tokens import Doc, Span

from . import config

import logging

logger = logging.getLogger(__name__)
SENTENCE_LENGTH_CATEGORIES = {
    "kurz": {"max_woerter": 6, "priorität": "niedrig"},
    "mittel": {"max_woerter": 10, "priorität": "mittel"},
    "lang": {"max_woerter": 15, "priorität": "hoch"},
    "sehr_lang": {"max_woerter": 20, "priorität": "sehr_hoch"},
}

COMPLEXITY_FACTORS = {
    "nebensätze": 3.0,
    "einschübe": 2.5,
    "passiv": 2.0,
    "modalverben": 1.5,
    "nominalisierungen": 2.0,
    "fremdwörter": 1.5,
    "konjunktionen": 1.2,
}

CONJUNCTIONS = {
    "subordinierend": [
        "weil",
        "da",
        "obwohl",
        "während",
        "nachdem",
        "bevor",
        "falls",
        "wenn",
        "dass",
    ],
    "koordinierend": ["und", "oder", "aber", "sondern", "denn", "doch"],
    "modal": [
        "indem",
        "dadurch",
        "somit",
        "folglich",
        "dennoch",
        "jedoch",
        "allerdings",
    ],
}


def _count_real_words(sent: Span) -> int:
    """Counts only real words without punctuation."""
    return len(
        [
            token
            for token in sent
            if not token.is_punct and not token.is_space and token.text.strip()
        ]
    )


def _detect_subclauses(sent: Span) -> List[Tuple[str, str]]:
    """Detects subclauses and their type."""
    subclauses = []

    for token in sent:
        if (
            token.dep_ == "mark"
            and token.lemma_.lower() in CONJUNCTIONS["subordinierend"]
        ):
            subclause_start = token.i
            subclause_end = sent.end

            for other_token in sent[token.i :]:
                if other_token.dep_ in ["punct"] and other_token.text in [
                    ".",
                    "!",
                    "?",
                ]:
                    subclause_end = other_token.i
                    break
                elif other_token.dep_ == "mark" and other_token.i > token.i:
                    subclause_end = other_token.i
                    break

            subclause = sent.doc[subclause_start:subclause_end]
            subclauses.append((subclause.text.strip(), "subordiniert"))

    return subclauses


def _detect_parentheticals(sent: Span) -> List[str]:
    """Detects parenthetical insertions."""
    insertions = []
    text = sent.text

    parenthetical_patterns = [r"\([^)]+\)", r"\[[^\]]+\]", r"—[^—]+—", r",[^,]{3,},"]

    for pattern in parenthetical_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            insertions.append(match.group())

    return insertions


def _calculate_syntactic_complexity(sent: Span) -> Tuple[float, Dict[str, int]]:
    """Calculates the syntactic complexity of a sentence."""
    complexity_score = 0.0
    factors = defaultdict(int)

    subclauses = _detect_subclauses(sent)
    factors["nebensätze"] = len(subclauses)
    complexity_score += len(subclauses) * COMPLEXITY_FACTORS["nebensätze"]

    parentheticals = _detect_parentheticals(sent)
    factors["einschübe"] = len(parentheticals)
    complexity_score += len(parentheticals) * COMPLEXITY_FACTORS["einschübe"]

    for token in sent:
        # spaCy v3 passive detection: werden (AUX) + Partizip II child
        if token.lemma_ == "werden" and token.pos_ == "AUX":
            has_participle_child = any(
                child.tag_ in ["VVPP", "VAPP", "VMPP"] for child in token.children
            )
            if has_participle_child:
                factors["passiv"] += 1

        if token.tag_.startswith("VM"):
            factors["modalverben"] += 1

        if token.pos_ == "NOUN" and any(
            token.text.lower().endswith(ending)
            for ending in ["ung", "heit", "keit", "tion"]
        ):
            factors["nominalisierungen"] += 1

        if token.lemma_.lower() in CONJUNCTIONS["subordinierend"]:
            factors["konjunktionen"] += 1

    for factor, count in factors.items():
        if factor in COMPLEXITY_FACTORS:
            complexity_score += count * COMPLEXITY_FACTORS[factor]

    return complexity_score, dict(factors)


def _generate_simplification_suggestion(
    sent: Span, word_count: int, complexity_factors: Dict[str, int]
) -> str:
    """Generates specific suggestions for sentence simplification."""
    suggestions = []

    if complexity_factors.get("nebensätze", 0) > 0:
        suggestions.append("Teilen Sie Nebensätze in eigene Sätze auf")

    if complexity_factors.get("einschübe", 0) > 0:
        suggestions.append(
            "Entfernen Sie Einschübe oder machen Sie eigene Sätze daraus"
        )

    if complexity_factors.get("passiv", 0) > 0:
        suggestions.append("Verwenden Sie Aktivsätze statt Passivsätze")

    if complexity_factors.get("nominalisierungen", 0) > 1:
        suggestions.append("Verwenden Sie Verben statt Substantivierungen")

    if word_count > 15:
        suggestions.append("Teilen Sie den Satz in mehrere kurze Sätze auf")

    if not suggestions:
        suggestions.append("Verkürzen Sie den Satz oder teilen Sie ihn auf")

    return "; ".join(suggestions)


def _classify_sentence_type(sent: Span) -> Tuple[str, float]:
    """Classifies sentences by type and complexity."""
    word_count = _count_real_words(sent)
    complexity_score, factors = _calculate_syntactic_complexity(sent)

    if complexity_score > 8.0:
        return "hochkomplex", 1.0
    elif complexity_score > 5.0:
        return "komplex", 0.8
    elif word_count > 15:
        return "lang", 0.7
    elif word_count > 10:
        return "mittel", 0.5
    else:
        return "einfach", 0.2


def check_rule(doc: Doc) -> List[str]:
    """
    Extended sentence length analysis with syntactic complexity evaluation:
    1. Adaptive sentence length limits based on complexity
    2. Syntactic analysis (subclauses, parentheticals, passive)
    3. Contextual evaluation of sentence difficulty
    4. Specific simplification suggestions

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of context-specific improvement suggestions
    """
    errors = []
    sentence_stats = []

    for i, sent in enumerate(doc.sents, 1):
        word_count = _count_real_words(sent)
        complexity_score, factors = _calculate_syntactic_complexity(sent)
        sentence_type, difficulty = _classify_sentence_type(sent)

        sentence_stats.append(
            {
                "satz_nr": i,
                "woerter": word_count,
                "komplexität": complexity_score,
                "typ": sentence_type,
            }
        )

        adaptive_limit = config.MAX_WORDS_DEFAULT
        if complexity_score > 5.0:
            adaptive_limit = config.MAX_WORDS_HIGH
        elif complexity_score > 3.0:
            adaptive_limit = config.MAX_WORDS_MEDIUM

        if word_count > adaptive_limit or complexity_score > config.COMPLEXITY_TRIGGER:
            simplification = _generate_simplification_suggestion(
                sent, word_count, factors
            )

            if complexity_score > 8.0:
                priority = "sehr hoch"
            elif complexity_score > 5.0 or word_count > 15:
                priority = "hoch"
            elif word_count > 12:
                priority = "mittel"
            else:
                priority = "niedrig"

            errors.append(
                f"Satz {i} ist zu komplex ({word_count} Wörter, Komplexität: {complexity_score:.1f}, "
                f"Typ: {sentence_type}, Priorität: {priority}). {simplification}."
            )

    avg_length = (
        statistics.mean([s["woerter"] for s in sentence_stats])
        if sentence_stats
        else 0
    )
    if avg_length > config.MAX_MEAN_WORDS:
        errors.append(
            f"Durchschnittliche Satzlänge ist mit {avg_length:.1f} Wörtern zu hoch. "
            f"Ziel: unter {config.MAX_MEAN_WORDS} Wörter pro Satz."
        )

    return errors


def analyze_text_complexity(text: str) -> Dict[str, Any]:
    """Performs a comprehensive analysis of text complexity."""
    import spacy

    nlp = spacy.load("de_core_news_lg")
    doc = nlp(text)

    sentence_data = []
    total_words = 0

    for i, sent in enumerate(doc.sents, 1):
        word_count = _count_real_words(sent)
        complexity_score, factors = _calculate_syntactic_complexity(sent)
        sentence_type, difficulty = _classify_sentence_type(sent)

        total_words += word_count

        sentence_data.append(
            {
                "satz_nr": i,
                "text": sent.text.strip()[:50] + "..."
                if len(sent.text) > 50
                else sent.text.strip(),
                "woerter_anzahl": word_count,
                "komplexitaets_score": round(complexity_score, 1),
                "typ": sentence_type,
                "schwierigkeit": difficulty,
                "faktoren": factors,
            }
        )

    if sentence_data:
        avg_length = statistics.mean(
            [s["woerter_anzahl"] for s in sentence_data]
        )
        avg_complexity = statistics.mean(
            [s["komplexitaets_score"] for s in sentence_data]
        )

        problematic_sentences = [
            s
            for s in sentence_data
            if s["woerter_anzahl"] > 12 or s["komplexitaets_score"] > 5.0
        ]
    else:
        avg_length = 0
        avg_complexity = 0
        problematic_sentences = []

    return {
        "gesamt_saetze": len(sentence_data),
        "durchschnittliche_laenge": round(avg_length, 1),
        "durchschnittliche_komplexitaet": round(avg_complexity, 1),
        "problematische_saetze": len(problematic_sentences),
        "satz_details": sentence_data,
        "bewertung": "sehr schwer"
        if avg_length > 15
        else "schwer"
        if avg_length > 12
        else "mittel"
        if avg_length > 8
        else "einfach",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nlp = spacy.load("de_core_news_lg")

    test_texts = [
        "Das Auto ist rot.",
        "Der Mann, der gestern hier war, hat sein Auto vor dem Haus geparkt.",
        "Die komplexe Analyse, die von verschiedenen Experten durchgeführt wurde, zeigt deutlich, dass nachhaltige Entwicklungen in diesem Bereich unbedingt erforderlich sind.",
        "Nachdem die Verhandlungen beendet waren und alle Beteiligten sich geeinigt hatten, wurde der Vertrag unterzeichnet.",
        "Es regnet heute stark.",
        "Die Lösung des Problems erfordert eine sorgfältige Analyse aller verfügbaren Daten, wobei verschiedene Faktoren berücksichtigt werden müssen.",
        "Sie können gerne kommen.",
    ]

    logger.info("--- Test: Extended sentence length rule ---")
    for i, text in enumerate(test_texts):
        logger.info("\nSentence %s: '%s'", i + 1, text)
        doc = nlp(text)
        results = check_rule(doc)

        if results:
            for error in results:
                logger.error(" %s", error)
        else:
            logger.info(" Sentence length and complexity are appropriate.")
