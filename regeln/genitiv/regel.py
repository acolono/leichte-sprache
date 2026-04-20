"""
Robust detection of genitive constructions with spaCy (v2.0 - structure-based).

CORE PHILOSOPHY:
- NO WORD LISTS: All detections based on morphological tags and dependency structures
- COMPLETE COVERAGE: Detects all genitive types (morphological, Saxon, prepositional, verb/adjective-governed)
- DEPENDENCY-BASED: Uses spaCy's dependency parsing for precise structural analysis

DETECTED GENITIVE TYPES:
1. Morphological genitive (Case=Gen): "des Nachbarn", "der Mutter"
2. Saxon genitive (possessive-s): "Peters Auto", "Annas Buch", "Deutschlands Wirtschaft"
3. Prepositional genitive: "wegen des Regens", "innerhalb der Frist"
4. Verb-governed genitive: "gedenken der Opfer", "bedarf des Geldes"
5. Adjective-governed genitive: "sich seiner Schuld bewusst", "des Lobes würdig"
"""

from typing import List, Set, Tuple

import spacy
from spacy.tokens import Doc, Span, Token

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

# --- HELPER FUNCTIONS --------------------------------------------------------


import logging

logger = logging.getLogger(__name__)
def _is_possessive_s_form(token: Token) -> bool:
    """
    Checks if a token is a Saxon genitive form (possessive-s).

    Patterns:
    - Peters, Annas, Deutschlands (ends with 's' without apostrophe)
    - Peter's, Anna's (with apostrophe, less common in German)
    - des Vaters, des Landes (morphological genitive, not possessive-s)
    """
    text = token.text

    # Pattern 1: Mit Apostroph (Peter's, Anna's)
    if "'" in text and text.endswith("'s"):
        return True

    # Pattern 2: Ohne Apostroph, endet auf 's' (Peters, Annas)
    # ABER: Muss PROPN (Eigenname) oder NOUN sein
    if text.endswith("s") and token.pos_ in ["PROPN", "NOUN"]:
        # Prüfe: Ist es ein normales Nomen im Genitiv? (hat dann Case=Gen)
        # Sächsischer Genitiv hat oft KEIN Case=Gen Tag
        if token.morph.get("Case") == ["Gen"]:
            return False  # Das ist morphologischer Genitiv, nicht Possessiv-s

        # Prüfe ob ein Nomen folgt (Possessiv-Konstruktion)
        # Dependency: ag (genitive attribute) oder nk (noun kernel)
        for child in token.children:
            if child.pos_ in ["NOUN", "PROPN"]:
                return True

        # Prüfe ob Token selbst Kind eines Nomens ist
        if token.head.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["ag", "nk", "sb"]:
            return True

    # Pattern 3: Endet auf 'es' oder 'ens' (seltener)
    if token.pos_ == "PROPN" and (text.endswith("es") or text.endswith("ens")):
        if token.head.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["ag", "nk"]:
            return True

    return False


def _collect_genitive_phrase(token: Token, doc: Doc) -> Tuple[Span, str]:
    """
    Collects a complete genitive phrase starting from a genitive token.

    Returns:
        (span, context_info)
        context_info: type of the governing element (preposition, verb, adjective, noun)
    """
    phrase_tokens = [token]

    # Sammle Determiner und Adjektive (als Kinder des Genitiv-Nomens)
    for child in token.children:
        if child.dep_ in ["det", "nk"] or (
            child.pos_ == "ADJ" and child.morph.get("Case") == ["Gen"]
        ):
            phrase_tokens.append(child)

    # Suche auch nach Determinern, die als Geschwister vor dem Nomen stehen
    # Beispiel: "des Nachbarn" - "des" könnte separates Token sein
    for t in doc:
        if t.morph.get("Case") == ["Gen"] and t.pos_ == "DET":
            # Prüfe ob der Determiner direkt vor unserem Token steht
            if t.i == token.i - 1 and t.head == token:
                phrase_tokens.append(t)

    # Sortiere nach Position
    phrase_tokens = sorted(set(phrase_tokens), key=lambda t: t.i)

    start_idx = min(t.i for t in phrase_tokens)
    end_idx = max(t.i for t in phrase_tokens) + 1
    span = doc[start_idx:end_idx]

    # Bestimme Kontext (was regiert den Genitiv?)
    head = token.head
    if head.pos_ == "ADP":
        kontext = f"Präposition '{head.text}'"
    elif head.pos_ == "VERB":
        kontext = f"Verb '{head.lemma_}'"
    elif head.pos_ in ["ADJ", "ADV"]:  # ADV für Adjektive wie "bewusst", "würdig"
        kontext = f"Adjektiv '{head.text}'"
    elif head.pos_ in ["NOUN", "PROPN"]:
        kontext = f"Nomen '{head.text}'"
    else:
        kontext = "unbekannter Kontext"

    return span, kontext


def _find_reference_noun_for_possessive_s(token: Token) -> Token:
    """
    Finds the reference noun for a possessive-s token.
    Example: "Peters Auto" -> Token=Peters, Reference=Auto
    """
    # Suche in Kindern
    for child in token.children:
        if child.pos_ in ["NOUN", "PROPN"]:
            return child

    # Suche im Head
    if token.head.pos_ in ["NOUN", "PROPN"] and token.head != token:
        return token.head

    return None


# --- MAIN DETECTION FUNCTION -------------------------------------------------


def check_rule(doc: Doc) -> List[str]:
    """
    Main function for detecting all genitive constructions.

    Detects:
    1. Morphological genitives (Case=Gen)
    2. Saxon genitives (possessive-s)
    3. Prepositional genitives (governed by ADP)
    4. Verb-governed genitives (oa/og dependency)
    5. Adjective-governed genitives

    Args:
        doc: spaCy Doc object

    Returns:
        List of error messages
    """
    errors = []
    processed_indices: Set[int] = set()

    for token in doc:
        if token.i in processed_indices:
            continue

        # --- KATEGORIE 1: MORPHOLOGISCHER GENITIV (Case=Gen) ---
        # Handle multi-value morphology: morph.get("Case") may return ["Gen"] or ["Gen", "Dat"]
        case_values = token.morph.get("Case")
        is_genitive = "Gen" in case_values if case_values else False

        if is_genitive and token.pos_ in ["NOUN", "PROPN", "PRON"]:
            span, kontext = _collect_genitive_phrase(token, doc)

            # Bestimme Typ basierend auf Kontext
            head = token.head
            if head.pos_ == "ADP":
                typ = "Präpositionaler Genitiv"
            elif head.pos_ == "VERB":
                typ = "Verbal-regierter Genitiv"
            elif head.pos_ in ["ADJ", "ADV"]:  # ADV für prädikative Adjektive
                typ = "Adjektiv-regierter Genitiv"
            else:
                typ = "Genitiv-Konstruktion"

            error_msg = (
                f'{typ} gefunden: "{span.text}" (regiert von {kontext}). '
                f'Besser: Verwenden Sie "von" oder "gehört zu".'
            )
            errors.append(error_msg)

            # Markiere all token indices as processed
            for t in span:
                processed_indices.add(t.i)

        # --- KATEGORIE 2: SÄCHSISCHER GENITIV (Possessiv-s) ---
        elif _is_possessive_s_form(token):
            reference_noun = _find_reference_noun_for_possessive_s(token)

            if reference_noun:
                pass
            else:
                pass

            error_msg = (
                f'Sächsischer Genitiv gefunden: "{token.text}". '
                f'Besser: "{token.text[:-1]} hat..." oder "Das ... gehört {token.text[:-1]}".'
            )
            errors.append(error_msg)
            processed_indices.add(token.i)

    return errors


# --- TESTING -----------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import spacy

    logger.info("=" * 80)
    logger.info("GENITIV-ERKENNUNG v2.0 - STRUKTURBASIERT (KEINE WORTLISTEN)")
    logger.info("=" * 80)

    try:
        nlp = spacy.load("de_core_news_lg")

        # Testfälle für alle Genitiv-Typen
        test_texte = [
            # Morphologische Genitive
            ("Das Auto des Nachbarn steht vor der Tür.", "Morphologisch: des Nachbarn"),
            ("Die Qualität der Arbeit ist wichtig.", "Morphologisch: der Arbeit"),
            (
                "Der Geschmack des Weines war ausgezeichnet.",
                "Morphologisch: des Weines",
            ),
            # Sächsische Genitive
            ("Peters Fahrrad ist rot.", "Sächsisch: Peters"),
            ("Annas Buch liegt auf dem Tisch.", "Sächsisch: Annas"),
            ("Deutschlands Wirtschaft wächst.", "Sächsisch: Deutschlands"),
            ("Goethes Werke sind weltbekannt.", "Sächsisch: Goethes"),
            # Präpositionale Genitive
            (
                "Wegen des Regens blieben wir zu Hause.",
                "Präpositional: wegen des Regens",
            ),
            (
                "Innerhalb der Frist müssen Sie antworten.",
                "Präpositional: innerhalb der Frist",
            ),
            (
                "Anstelle des Chefs kam der Stellvertreter.",
                "Präpositional: anstelle des Chefs",
            ),
            (
                "Trotz des Protests ging die Veranstaltung weiter.",
                "Präpositional: trotz des Protests",
            ),
            # Verbal-regierte Genitive
            ("Wir gedenken der Opfer.", "Verbal: gedenken der Opfer"),
            ("Er bedarf des Geldes.", "Verbal: bedarf des Geldes"),
            ("Sie erinnert sich des Vorfalls.", "Verbal: erinnert sich des Vorfalls"),
            # Adjektiv-regierte Genitive
            ("Er ist sich seiner Schuld bewusst.", "Adjektiv: seiner Schuld bewusst"),
            ("Sie war des Lobes würdig.", "Adjektiv: des Lobes würdig"),
        ]

        logger.info("\n--- GENITIV-ERKENNUNGS-TESTS ---\n")

        gefunden = 0
        gesamt = len(test_texte)

        for i, (text, erwartung) in enumerate(test_texte, 1):
            logger.info("[%s/%s] %s", i, gesamt, text)
            logger.info(" Erwartet: %s", erwartung)

            doc = nlp(text)
            results = check_rule(doc)

            if results:
                gefunden += 1
                for error in results:
                    logger.error(" %s", error)
            else:
                logger.info(" ✗ Kein Genitiv erkannt!")
            logger.info("")

        logger.info("=" * 80)
        logger.info("ERGEBNIS: %s/%s Genitive erkannt (%.1f%)", gefunden, gesamt, gefunden / gesamt * 100)
        logger.info("=" * 80)

    except OSError:
        logger.error(" spaCy-Modell 'de_core_news_lg' nicht gefunden!")
        logger.error(" Bitte ausführen: python -m spacy download de_core_news_lg")
