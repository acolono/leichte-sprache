"""
Rule for detecting complex sentence structures.

Combines syntactic analysis (spaCy) with BERT-based complexity evaluation
to identify sentences that are too complex for Leichte Sprache.

Focus: sentence structure complexity (not word complexity -- that is handled by the komplexitaet rule)
"""

from pathlib import Path
from typing import Any, Dict, List

import spacy
from spacy.tokens import Doc, Span

from . import config

# Model path for the perplexity n-gram model (per CONTEXT.md normalization decision).
# The rule currently uses purely syntactic analysis via spaCy. When perplexity-based
# scoring is integrated, load the pickle from this path.
MODEL_PATH = Path(__file__).parent / "model" / "perplexity_model.pkl"

# Subordinating conjunctions (introduce subordinate clauses)
import logging

logger = logging.getLogger(__name__)
SUBORDINATE_CONJUNCTIONS = {
    "dass",
    "weil",
    "da",
    "obwohl",
    "obgleich",
    "wenn",
    "falls",
    "sofern",
    "damit",
    "bevor",
    "nachdem",
    "während",
    "sobald",
    "solange",
    "bis",
    "ob",
    "als",
    "wie",
    "indem",
    "ohne",
    "anstatt",
    "wodurch",
    "weshalb",
    "weswegen",
    "worauf",
    "womit",
    "woran",
    "worin",
    "worüber",
}

# Relative pronouns (introduce relative clauses)
RELATIVE_PRONOUNS = {"der", "die", "das", "welcher", "welche", "welches", "wer", "was"}


def _count_subordinate_clauses(sent: Span) -> int:
    """
    Counts the number of subordinate clauses in a sentence.

    Detects:
    - Conjunction-introduced subordinate clauses (weil, dass, wenn, ...)
    - Relative clauses (der/die/das + verb at end)
    """
    count = 0

    for token in sent:
        # Subordinierende Konjunktionen
        if token.lemma_.lower() in SUBORDINATE_CONJUNCTIONS:
            count += 1
        # Relativpronomen als Satzeinleitung (dep_ == 'sb' oder 'oa' nach Komma)
        elif (
            token.lemma_.lower() in RELATIVE_PRONOUNS
            and token.dep_ in ("sb", "oa", "da", "og")
            and token.i > 0
            and sent.doc[token.i - 1].text == ","
        ):
            count += 1

    return count


def _get_dependency_depth(sent: Span) -> int:
    """
    Computes the maximum depth of the dependency tree.

    Deeper trees = more complex sentence structure.
    """

    def get_depth(token, current_depth=0):
        children_depths = [
            get_depth(child, current_depth + 1) for child in token.children
        ]
        return max(children_depths) if children_depths else current_depth

    root = None
    for token in sent:
        if token.dep_ == "ROOT":
            root = token
            break

    if root is None:
        return 0

    return get_depth(root)


def _get_subject_verb_distance(sent: Span) -> int:
    """
    Computes the distance between subject and main verb.

    Large distance = harder to understand (reader must hold more in memory).
    """
    subject_pos = None
    verb_pos = None

    for token in sent:
        if token.dep_ in ("sb", "nsubj") and subject_pos is None:
            subject_pos = token.i
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            verb_pos = token.i

    if subject_pos is not None and verb_pos is not None:
        return abs(verb_pos - subject_pos)

    return 0


def _has_passive_construction(sent: Span) -> bool:
    """Checks if the sentence contains a passive construction."""
    for token in sent:
        # Passiv-Hilfsverb "werden" + Partizip II
        if token.lemma_ == "werden" and token.pos_ == "AUX":
            for child in token.head.children:
                if child.tag_ in ("VVPP", "VAPP"):  # Partizip Perfekt
                    return True
            # Oder token selbst ist ROOT und hat Partizip als Kind
            for child in token.children:
                if child.tag_ in ("VVPP", "VAPP"):
                    return True
    return False


def _count_verbs(sent: Span) -> int:
    """Counts the number of verbs in the sentence."""
    return len([t for t in sent if t.pos_ in ("VERB", "AUX")])


def _count_words(sent: Span) -> int:
    """Counts the number of words (excluding punctuation)."""
    return len([t for t in sent if not t.is_punct and not t.is_space])


def _analyze_sentence_structure(sent: Span) -> Dict[str, Any]:
    """
    Analyzes the structure of a sentence.

    Returns:
        Dictionary with structure metrics
    """
    return {
        "word_count": _count_words(sent),
        "subordinate_clauses": _count_subordinate_clauses(sent),
        "dependency_depth": _get_dependency_depth(sent),
        "subject_verb_distance": _get_subject_verb_distance(sent),
        "has_passive": _has_passive_construction(sent),
        "verb_count": _count_verbs(sent),
    }


def _is_sentence_complex(analysis: Dict[str, Any]) -> bool:
    """
    Determines if a sentence is complex based on structure analysis.

    A sentence is considered complex if at least one applies:
    - More than 1 subordinate clause
    - Dependency depth > 5
    - Subject-verb distance > 6
    - More than 15 words AND at least one additional factor
    """
    # Schwellenwerte aus config
    issues_count = 0

    if analysis["subordinate_clauses"] >= config.MAX_SUBORDINATE_CLAUSES:
        issues_count += 1

    if analysis["dependency_depth"] > config.MAX_DEPENDENCY_DEPTH:
        issues_count += 1

    if analysis["subject_verb_distance"] > config.MAX_SUBJECT_VERB_DISTANCE:
        issues_count += 1

    if analysis["has_passive"]:
        issues_count += 1

    # Lange Sätze sind problematisch, wenn kombiniert mit anderen Faktoren
    if analysis["word_count"] > config.MAX_WORDS_PER_SENTENCE:
        if issues_count > 0 or analysis["word_count"] > config.MAX_WORDS_HARD_LIMIT:
            issues_count += 1

    return issues_count >= config.MIN_ISSUES_FOR_FLAG


def _format_feedback(sent: Span, analysis: Dict[str, Any]) -> str:
    """
    Creates a user-friendly error message with concrete improvement suggestions.
    """
    sentence_text = sent.text.strip()
    # Kürze sehr lange Sätze in der Anzeige
    display_text = (
        sentence_text[:80] + "..." if len(sentence_text) > 80 else sentence_text
    )

    issues = []
    suggestions = []

    # Nebensätze
    if analysis["subordinate_clauses"] >= config.MAX_SUBORDINATE_CLAUSES:
        if analysis["subordinate_clauses"] == 1:
            issues.append("1 Nebensatz")
        else:
            issues.append(f"{analysis['subordinate_clauses']} Nebensätze")
        suggestions.append(
            "Teilen Sie den Satz auf. Jeder Gedanke sollte ein eigener Satz sein"
        )

    # Satzlänge
    if analysis["word_count"] > config.MAX_WORDS_PER_SENTENCE:
        issues.append(
            f"{analysis['word_count']} Wörter (Ziel: max. {config.MAX_WORDS_PER_SENTENCE})"
        )
        suggestions.append("Kürzen Sie den Satz")

    # Passiv
    if analysis["has_passive"]:
        issues.append("Passiv-Konstruktion")
        suggestions.append("Schreiben Sie im Aktiv: Wer tut was?")

    # Subjekt-Verb-Distanz
    if analysis["subject_verb_distance"] > config.MAX_SUBJECT_VERB_DISTANCE:
        issues.append("Subjekt und Verb sind weit voneinander entfernt")
        suggestions.append("Stellen Sie das Verb näher zum Subjekt")

    # Dependency-Tiefe (nur bei sehr komplexen Sätzen erwähnen)
    if analysis["dependency_depth"] > config.MAX_DEPENDENCY_DEPTH:
        issues.append("verschachtelte Satzstruktur")
        suggestions.append("Vereinfachen Sie die Satzstruktur")

    # Formatiere die Nachricht
    msg = f'Satz ist schwer zu verstehen: "{display_text}"'

    if issues:
        msg += f" (Probleme: {', '.join(issues)})"

    if suggestions:
        # Nimm nur die wichtigsten 2 Vorschläge
        top_suggestions = suggestions[:2]
        msg += f". Besser: {'; '.join(top_suggestions)}."

    return msg


def check_rule(doc: Doc) -> List[str]:
    """
    Checks text for complex sentence structures.

    This rule focuses on STRUCTURAL complexity:
    - Subordinate clauses and nesting
    - Sentence length
    - Passive constructions
    - Subject-verb distance

    Word complexity is handled by the 'komplexitaet' rule.

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of error messages with concrete improvement suggestions
    """
    errors = []

    for sent in doc.sents:
        # Überspringe sehr kurze Sätze
        if len(sent.text.strip()) < 10:
            continue

        # Analysiere Satzstruktur
        analysis = _analyze_sentence_structure(sent)

        # Prüfe ob komplex
        if _is_sentence_complex(analysis):
            feedback = _format_feedback(sent, analysis)
            errors.append(feedback)

    return errors


# Für detaillierte Analyse (z.B. API-Endpunkt)
def analyze_sentence_structure(text: str) -> Dict[str, Any]:
    """
    Detaillierte Satzstruktur-Analyse für einen Text.

    Args:
        text: Zu analysierender Text

    Returns:
        Dictionary mit detaillierten Analyse-Ergebnissen
    """
    import spacy

    nlp = spacy.load("de_core_news_lg")
    doc = nlp(text)

    satz_analysen = []

    for i, sent in enumerate(doc.sents, 1):
        if len(sent.text.strip()) < 10:
            continue

        analysis = _analyze_sentence_structure(sent)
        is_complex = _is_sentence_complex(analysis)

        satz_analysen.append(
            {
                "nummer": i,
                "text": sent.text.strip(),
                "wort_anzahl": analysis["word_count"],
                "nebensaetze": analysis["subordinate_clauses"],
                "dependency_tiefe": analysis["dependency_depth"],
                "subjekt_verb_distanz": analysis["subject_verb_distance"],
                "hat_passiv": analysis["has_passive"],
                "verb_anzahl": analysis["verb_count"],
                "ist_komplex": is_complex,
            }
        )

    komplexe_saetze = [s for s in satz_analysen if s["ist_komplex"]]

    return {
        "satz_analysen": satz_analysen,
        "statistiken": {
            "anzahl_saetze": len(satz_analysen),
            "komplexe_saetze": len(komplexe_saetze),
            "durchschnitt_woerter": (
                sum(s["wort_anzahl"] for s in satz_analysen) / len(satz_analysen)
                if satz_analysen
                else 0
            ),
            "durchschnitt_nebensaetze": (
                sum(s["nebensaetze"] for s in satz_analysen) / len(satz_analysen)
                if satz_analysen
                else 0
            ),
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test der Regel
    import spacy

    logger.info("--- Test: Satzstruktur-Komplexität ---")

    test_texts = [
        "Der Mann geht in das Haus.",
        "Die 4m breite Feuerwehrzufahrt sowie der Reversierplatz für das Feuerwehrfahrzeug sind in jedem Fall freizuhalten.",
        "Es darf auch kurzfristig keine Ware in diesem Bereich gelagert werden.",
        "Wenn Sie das Formular ausfüllen, das wir Ihnen geschickt haben, nachdem Sie den Antrag gestellt haben, können wir Ihren Fall bearbeiten.",
        "Die Katze schläft.",
    ]

    nlp = spacy.load("de_core_news_lg")

    for i, text in enumerate(test_texts, 1):
        logger.info("\n%s. Text: '%s'", i, text)
        doc = nlp(text)
        results = check_rule(doc)

        if results:
            logger.info(" Probleme gefunden:")
            for problem in results:
                logger.info(" %s", problem)
        else:
            logger.info(" Satzstruktur ist einfach genug.")
