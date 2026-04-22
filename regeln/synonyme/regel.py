"""
Advanced rule for synonym detection using NLP techniques.
Uses semantic vector analysis, word frequency evaluation, and contextual consistency checking.
"""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from spacy.tokens import Doc, Token
from wordfreq import word_frequency

from . import config  # noqa: F401


# OpenThesaurus-backed synset index. Loaded lazily on first call.
# Licensing: LGPL-2.1 or CC-BY-SA 4.0. File bundled under regeln/synonyme/data/.
_OT_INDEX: Optional[Dict[str, Set[int]]] = None


def _load_openthesaurus() -> Dict[str, Set[int]]:
    """Load OpenThesaurus synsets as lemma -> set-of-synset-ids.

    Returns empty dict if the data file is missing — rule degrades to the
    curated BEKANNTE_SYNONYME + vector-similarity path.
    """
    global _OT_INDEX
    if _OT_INDEX is not None:
        return _OT_INDEX
    index: Dict[str, Set[int]] = {}
    data_path = Path(__file__).parent / "data" / "openthesaurus.json"
    if data_path.is_file():
        try:
            synsets = json.loads(data_path.read_text(encoding="utf-8"))
            for sid, synset in enumerate(synsets):
                for word in synset:
                    lower = word.lower()
                    index.setdefault(lower, set()).add(sid)
        except (OSError, json.JSONDecodeError):
            index = {}
    _OT_INDEX = index
    return index


def _openthesaurus_synsets(lemma: str) -> Optional[Set[int]]:
    """Return the set of OpenThesaurus synset IDs a lemma belongs to."""
    idx = _load_openthesaurus()
    return idx.get(lemma)

# Known synonym pairs with recommendations and context information
import logging

logger = logging.getLogger(__name__)
BEKANNTE_SYNONYME: Dict[str, Dict[str, Any]] = {
    "automobil": {
        "synonyme": ["fahrzeug", "auto", "wagen", "kfz"],
        "empfehlung": "Auto",
        "schwierigkeit": 0.8,
        "kategorie": "verkehr",
    },
    "fahrzeug": {
        "synonyme": ["auto", "wagen", "kfz", "pkw"],
        "empfehlung": "Auto",
        "schwierigkeit": 0.6,
        "kategorie": "verkehr",
    },
    "office": {
        "synonyme": ["büro", "arbeitsplatz"],
        "empfehlung": "Büro",
        "schwierigkeit": 0.8,
        "kategorie": "arbeit",
    },
    "konzern": {
        "synonyme": ["unternehmen", "firma", "betrieb"],
        "empfehlung": "Firma",
        "schwierigkeit": 0.8,
        "kategorie": "wirtschaft",
    },
    "immobilie": {
        "synonyme": ["gebäude", "haus", "objekt"],
        "empfehlung": "Haus",
        "schwierigkeit": 0.9,
        "kategorie": "wohnen",
    },
    "individuum": {
        "synonyme": ["person", "mensch", "bürger"],
        "empfehlung": "Person",
        "schwierigkeit": 0.9,
        "kategorie": "mensch",
    },
    "kapital": {
        "synonyme": ["geld", "finanzen", "mittel"],
        "empfehlung": "Geld",
        "schwierigkeit": 0.8,
        "kategorie": "finanzen",
    },
    "tätigkeit": {
        "synonyme": ["arbeit", "job", "beruf"],
        "empfehlung": "Arbeit",
        "schwierigkeit": 0.7,
        "kategorie": "arbeit",
    },
    "residenz": {
        "synonyme": ["wohnung", "heim", "zuhause"],
        "empfehlung": "Wohnung",
        "schwierigkeit": 0.8,
        "kategorie": "wohnen",
    },
    "kommunikation": {
        "synonyme": ["gespräch", "unterhaltung", "diskussion"],
        "empfehlung": "Gespräch",
        "schwierigkeit": 0.8,
        "kategorie": "sprechen",
    },
    "information": {
        "synonyme": ["nachricht", "mitteilung", "auskunft"],
        "empfehlung": "Nachricht",
        "schwierigkeit": 0.7,
        "kategorie": "sprechen",
    },
    "analyse": {
        "synonyme": ["untersuchung", "prüfung", "bewertung"],
        "empfehlung": "Prüfung",
        "schwierigkeit": 0.8,
        "kategorie": "wissenschaft",
    },
    "implementierung": {
        "synonyme": ["umsetzung", "realisierung", "durchführung"],
        "empfehlung": "Umsetzung",
        "schwierigkeit": 0.9,
        "kategorie": "aktion",
    },
    "optimierung": {
        "synonyme": ["verbesserung", "perfektionierung"],
        "empfehlung": "Verbesserung",
        "schwierigkeit": 0.8,
        "kategorie": "aktion",
    },
}

ÄHNLICHKEITS_SCHWELLWERTE = {
    "hoch": 0.75,  # Sehr ähnliche Wörter
    "mittel": 0.65,  # Möglicherweise verwandte Wörter
    "niedrig": 0.55,  # Schwach verwandte Wörter
}

WORD_FREQ_CATEGORIES = {
    "sehr_häufig": 0.001,  # > 0.1%
    "häufig": 0.0001,  # 0.01% - 0.1%
    "mittel": 0.00001,  # 0.001% - 0.01%
    "selten": 0,  # < 0.001%
}

LÄNGEN_PRÄFERENZEN = {
    "sehr_kurz": {"max": 4, "bonus": 0.3},
    "kurz": {"max": 6, "bonus": 0.2},
    "mittel": {"max": 8, "bonus": 0.1},
    "lang": {"max": float("inf"), "bonus": 0},
}


def _detect_semantic_clusters(doc: Doc) -> Dict[str, List[Token]]:
    """Groups semantically similar words into clusters."""
    nomen = [
        token
        for token in doc
        if token.pos_ == "NOUN" and token.has_vector and len(token.text) > 3
    ]

    cluster = defaultdict(list)
    bereits_geclustert = set()

    for i, token1 in enumerate(nomen):
        if token1.lemma_ in bereits_geclustert:
            continue

        cluster_key = token1.lemma_
        cluster[cluster_key].append(token1)
        bereits_geclustert.add(token1.lemma_)

        for token2 in nomen[i + 1 :]:
            if (
                token2.lemma_ not in bereits_geclustert
                and token1.similarity(token2) >= ÄHNLICHKEITS_SCHWELLWERTE["mittel"]
            ):
                cluster[cluster_key].append(token2)
                bereits_geclustert.add(token2.lemma_)

    return {k: v for k, v in cluster.items() if len(v) > 1}


def _evaluate_word_quality(token: Token) -> float:
    """Evaluates word quality for Leichte Sprache."""
    score = 0.0
    lemma = token.lemma_.lower()

    # Frequency evaluation
    freq = word_frequency(lemma, "de")
    for category, threshold in WORD_FREQ_CATEGORIES.items():
        if freq > threshold:
            score += {"sehr_häufig": 0.4, "häufig": 0.3, "mittel": 0.2, "selten": 0.0}[
                category
            ]
            break

    # Length evaluation
    for _category, info in LÄNGEN_PRÄFERENZEN.items():
        if len(token.text) <= info["max"]:
            score += info["bonus"]
            break

    # German vs. foreign word evaluation
    for pattern in ["tion", "ment", "ung", "heit", "keit"]:
        if pattern in lemma:
            if pattern in ["ung", "heit", "keit"]:
                score += 0.2  # Prefer German suffixes
            else:
                score -= 0.1  # Penalize foreign word suffixes
            break

    # Complexity evaluation
    if re.match(r"^[a-zA-ZäöüÄÖÜß]+$", token.text):  # Letters only
        score += 0.1

    return score


def _find_best_alternative(token_gruppe: List[Token]) -> Tuple[str, float, str]:
    """Finds the best alternative from a group of similar tokens."""
    evaluations = []

    for token in token_gruppe:
        quality = _evaluate_word_quality(token)
        evaluations.append((token.text, quality, token.lemma_))

    evaluations.sort(key=lambda x: x[1], reverse=True)
    beste_option = evaluations[0]

    # Reason for selection
    if len(beste_option[0]) <= 6:
        grund = "kurzes Wort"
    elif word_frequency(beste_option[2].lower(), "de") > 0.0001:
        grund = "häufiges Wort"
    else:
        grund = "einfacheres Wort"

    return beste_option[0], beste_option[1], grund


def _evaluate_synonym_context(tokens: List[Token], doc: Doc) -> float:
    """Evaluates consistency issues based on context."""
    if len(tokens) < 2:
        return 0.0

    score = 0.0

    # Distance between synonyms
    positionen = [token.i for token in tokens]
    min_distanz = min(
        abs(positionen[i] - positionen[j])
        for i in range(len(positionen))
        for j in range(i + 1, len(positionen))
    )

    if min_distanz < 20:  # Close together
        score += 0.4
    elif min_distanz < 50:  # Moderately close
        score += 0.2

    # Sentence level: In same sentences?
    sentences = set()
    for token in tokens:
        for sent in doc.sents:
            if token in sent:
                sentences.add(sent.start)
                break

    if len(sentences) < len(tokens):  # Synonyms in same sentences
        score += 0.5

    # Domain context evaluation
    sent_texte = [token.sent.text.lower() for token in tokens]
    gesamter_kontext = " ".join(sent_texte)

    fach_indikatoren = [
        "wissenschaft",
        "forschung",
        "analyse",
        "verwaltung",
        "behörde",
        "unternehmen",
    ]
    if any(indikator in gesamter_kontext for indikator in fach_indikatoren):
        score += 0.3

    return score


def _is_genuine_synonym_pair(token1: Token, token2: Token) -> bool:
    """Checks if two tokens are genuine synonyms (not just similar words)."""
    lemma1 = token1.lemma_.lower()
    lemma2 = token2.lemma_.lower()

    # Check known synonyms
    if lemma1 in BEKANNTE_SYNONYME:
        return lemma2 in BEKANNTE_SYNONYME[lemma1]["synonyme"]
    if lemma2 in BEKANNTE_SYNONYME:
        return lemma1 in BEKANNTE_SYNONYME[lemma2]["synonyme"]

    # OpenThesaurus-backed synset check (DIN SPEC 33429: "one word per concept").
    # Opt-in via config.USE_OPENTHESAURUS because the raw thesaurus treats pairs
    # like "Haus/Familie" as synonyms (Habsburg-house reading) which produces
    # false positives on ordinary prose.
    if getattr(config, "USE_OPENTHESAURUS", False):
        ot_synsets1 = _openthesaurus_synsets(lemma1) or _openthesaurus_synsets(token1.text.lower())
        ot_synsets2 = _openthesaurus_synsets(lemma2) or _openthesaurus_synsets(token2.text.lower())
        if ot_synsets1 and ot_synsets2 and ot_synsets1 & ot_synsets2:
            if token1.pos_ == token2.pos_ and (token1.has_vector and token2.has_vector):
                # Require additional vector-similarity confirmation.
                if token1.similarity(token2) >= 0.6:
                    return True

    # Semantic similarity with stricter criteria
    if not (token1.has_vector and token2.has_vector):
        return False

    similarity = token1.similarity(token2)

    # ML-ENHANCED: Much stricter thresholds for scientific texts
    if len(token1.text) <= 5 and len(token2.text) <= 5:
        threshold = 0.85  # High for short words
    elif len(token1.text) <= 8 and len(token2.text) <= 8:
        threshold = 0.80  # Moderate for medium words
    else:
        threshold = 0.75  # More moderate for long words

    # ML-ENHANCED: Additional semantic checks
    if similarity >= threshold:
        # Check if words have different POS tags (then probably not synonyms)
        if token1.pos_ != token2.pos_:
            return False
        # Check if they have different grammatical properties
        if token1.tag_ != token2.tag_ and abs(len(token1.text) - len(token2.text)) > 3:
            return False
        return True

    return False


def check_rule(doc: Doc) -> List[str]:
    """
    Advanced synonym detection with semantic cluster analysis:
    1. Semantic clustering of similar terms
    2. Frequency- and length-based quality evaluation
    3. Contextual consistency evaluation
    4. German vs. foreign word preferences

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of context-specific improvement suggestions
    """
    errors = []

    # Collect all relevant nouns
    nomen = [
        token
        for token in doc
        if token.pos_ == "NOUN"
        and not token.is_punct
        and not token.is_space
        and token.has_vector
        and len(token.text) > 3
    ]

    bereits_verarbeitet = set()

    # Find semantic clusters
    cluster = _detect_semantic_clusters(doc)

    for _cluster_key, tokens in cluster.items():
        if len(tokens) < 2:
            continue

        # Check for genuine synonyms
        echte_synonyme = []
        for i, token1 in enumerate(tokens):
            for token2 in tokens[i + 1 :]:
                if _is_genuine_synonym_pair(token1, token2):
                    if token1 not in echte_synonyme:
                        echte_synonyme.append(token1)
                    if token2 not in echte_synonyme:
                        echte_synonyme.append(token2)

        # Synonym cluster: at least 2 genuine synonyms
        if len(echte_synonyme) >= 2:
            kontext_score = _evaluate_synonym_context(echte_synonyme, doc)

            # Context threshold for relevance
            if kontext_score >= 0.5:
                beste_alternative, quality, grund = _find_best_alternative(
                    echte_synonyme
                )

                other_words = [
                    t.text for t in echte_synonyme if t.text != beste_alternative
                ]

                priority = "hoch" if kontext_score >= 0.8 else "mittel"

                errors.append(
                    f"Synonyme gefunden: {', '.join(other_words)} (Priorität: {priority}). "
                    f'Einheitlich "{beste_alternative}" verwenden ({grund}).'
                )

                # Mark as processed
                for token in echte_synonyme:
                    bereits_verarbeitet.add(token.lemma_)

    # Direct pair analysis for non-clustered synonyms
    for i, token1 in enumerate(nomen):
        if token1.lemma_ in bereits_verarbeitet:
            continue

        for token2 in nomen[i + 1 :]:
            if (
                token2.lemma_ not in bereits_verarbeitet
                and token1.lemma_ != token2.lemma_
                and _is_genuine_synonym_pair(token1, token2)
            ):
                kontext_score = _evaluate_synonym_context([token1, token2], doc)

                if kontext_score >= 0.4:
                    beste_alternative, _, grund = _find_best_alternative(
                        [token1, token2]
                    )

                    similarity = token1.similarity(token2)
                    priority = "hoch" if similarity >= 0.8 else "mittel"

                    errors.append(
                        f'Synonym-Paar "{token1.text}" und "{token2.text}" (Ähnlichkeit: {similarity:.2f}, '
                        f'Priorität: {priority}). Einheitlich "{beste_alternative}" verwenden ({grund}).'
                    )

                    bereits_verarbeitet.add(token1.lemma_)
                    bereits_verarbeitet.add(token2.lemma_)
                    break

    return errors


def extended_synonym_analysis(text: str) -> Dict[str, Any]:
    """Performs a comprehensive synonym analysis."""
    import spacy

    nlp = spacy.load("de_core_news_lg")
    doc = nlp(text)

    nomen = [
        token
        for token in doc
        if token.pos_ == "NOUN" and token.has_vector and len(token.text) > 3
    ]

    synonym_cluster = []
    semantic_similarities = []
    konsistenz_probleme = 0

    # Cluster analysis
    cluster = _detect_semantic_clusters(doc)

    for _cluster_key, tokens in cluster.items():
        if len(tokens) >= 2:
            # Compute internal similarities
            similarities = []
            for i, token1 in enumerate(tokens):
                for token2 in tokens[i + 1 :]:
                    if token1.has_vector and token2.has_vector:
                        sim = token1.similarity(token2)
                        similarities.append(sim)
                        semantic_similarities.append(sim)

            if similarities:
                durchschnitt_sim = np.mean(similarities)
                beste_alternative, quality, grund = _find_best_alternative(tokens)

                if durchschnitt_sim >= ÄHNLICHKEITS_SCHWELLWERTE["mittel"]:
                    konsistenz_probleme += 1

                    synonym_cluster.append(
                        {
                            "wörter": [t.text for t in tokens],
                            "empfehlung": beste_alternative,
                            "ähnlichkeit": round(durchschnitt_sim, 3),
                            "qualitaet_grund": grund,
                            "cluster_größe": len(tokens),
                        }
                    )

    # Overall statistics
    all_similarities = semantic_similarities
    average_similarity = (
        np.mean(all_similarities) if all_similarities else 0
    )

    konsistenz_score = 1.0 - (konsistenz_probleme / len(nomen)) if nomen else 1.0

    return {
        "gesamt_nomen": len(nomen),
        "synonym_cluster": synonym_cluster,
        "konsistenz_probleme": konsistenz_probleme,
        "konsistenz_score": round(konsistenz_score, 3),
        "durchschnittliche_ähnlichkeit": round(average_similarity, 3),
        "bewertung": "problematisch"
        if konsistenz_score < 0.7
        else "mittel"
        if konsistenz_score < 0.9
        else "gut",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nlp = spacy.load("de_core_news_lg")

    test_texte = [
        "Das Automobil und das Fahrzeug stehen im Parkhaus.",
        "Die Firma expandiert, während das Unternehmen wächst.",
        "Seine Residenz ist ein schönes Haus mit großer Wohnung.",
        "Die Kommunikation und das Gespräch verliefen erfolgreich.",
        "Das Auto ist rot und das Fahrzeug schnell.",
        "Die Analyse zeigt, dass die Untersuchung wichtig ist.",
        "Der Mann arbeitet, während die Person ruht.",
        "Die Implementierung der Umsetzung dauert lange.",
        "Das Haus steht neben der Straße.",
        "Die Information und Nachricht erreichten alle Bürger.",
        "Die Optimierung der Verbesserung ist notwendig.",
    ]

    logger.info("--- Test: Erweiterte Regel Synonyme ---")
    for i, text in enumerate(test_texte):
        logger.info("\nSatz %s: '%s'", i + 1, text)
        doc = nlp(text)
        results = check_rule(doc)

        if results:
            for error in results:
                logger.error(" %s", error)
        else:
            logger.info(" Keine problematischen Synonyme gefunden.")
