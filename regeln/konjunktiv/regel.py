"""
Morphology-based subjunctive (Konjunktiv) detection for Leichte Sprache.

CORE APPROACH:
- Morphological analysis via spaCy (token.morph.get("Mood") == "Sub")
- NO pure dictionary or regex -- robust context detection
- Resolves ambiguity K2 vs. Praeteritum ("sagte" vs. "verdiente" in subjunctive)
- Detects ALL irregular forms (spraeche, faende, stuerbe, etc.)
- Detects passive/modal structures ("solle erstellt werden")
"""

from typing import Dict, List

from spacy.tokens import Doc, Token

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

# =========================================================================
# RULE FUNCTION FOR LEICHTE SPRACHE
# =========================================================================


import logging

logger = logging.getLogger(__name__)
def check_rule(doc: Doc) -> List[str]:
    """
    Main function for Leichte Sprache: detects subjunctive forms and provides improvement suggestions.

    Uses morphological analysis (token.morph) for robust detection.

    Args:
        doc: spaCy Doc object

    Returns:
        List of error messages with improvement suggestions
    """
    errors = []
    processed_tokens = set()

    for token in doc:
        if token.i in processed_tokens:
            continue

        # Check only verbs and auxiliaries
        if token.pos_ not in ["VERB", "AUX"]:
            continue

        # Morphological analysis
        morph_tags = token.morph.to_dict()

        if morph_tags.get("Mood") == "Sub":
            # Skip idiomatic exceptions
            if is_idiomatic_exception(token):
                processed_tokens.add(token.i)
                continue

            # Analyze subjunctive type and function
            subjunctive_info = classify_subjunctive_syntactically(token)

            errors.append(
                f'{subjunctive_info["typ"]} "{token.text}" (Lemma: {token.lemma_}) - '
                f"{subjunctive_info['funktion']}. "
                f"Besser: {subjunctive_info['vorschlag']}"
            )

            processed_tokens.add(token.i)

    return errors


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================


def is_idiomatic_exception(token: Token) -> bool:
    """
    Detects idiomatic exceptions that may be acceptable in Leichte Sprache.
    """
    # "es sei denn" - feste Wendung
    if token.lower_ == "sei" and token.i > 0 and token.i < len(token.doc) - 1:
        if (
            token.doc[token.i - 1].lower_ == "es"
            and token.doc[token.i + 1].lower_ == "denn"
        ):
            return True

    # Additional fixed expressions can be added here
    return False


def classify_subjunctive_syntactically(token: Token) -> Dict[str, str]:
    """
    Classifies subjunctives PURELY SYNTACTICALLY without word lists.

    PRINCIPLE: Uses ONLY abstract grammatical features:
    - token.morph (morphology)
    - token.dep_ (dependency)
    - token.head (syntactic head)
    - token.pos_ (part-of-speech)

    NO semantic word lists!

    Returns:
        Dictionary with {"typ": ..., "funktion": ..., "vorschlag": ...}
    """
    morph_tags = token.morph.to_dict()
    tense = morph_tags.get("Tense")
    dep = token.dep_
    head = token.head

    # =========================================================================
    # REGEL 1: INDIREKTE REDE (Konjunktiv I)
    # =========================================================================
    # KRITERIUM: Tense=Pres UND dep_ in [ccomp, oc] (clausal complement / object complement)
    # BEDEUTUNG: Das Verb ist Teil eines Nebensatzes, der als Objekt dient
    # BEISPIEL: "Er sagte, [dass er krank sei (ccomp/oc)]"
    if tense == "Pres" and dep in ["ccomp", "oc"]:
        return {
            "typ": "Konjunktiv I",
            "funktion": "Indirekte Rede",
            "vorschlag": "Verwenden Sie direkte Rede: \"Er sagt: 'Ich...'\"",
        }

    # =========================================================================
    # REGEL 2: IRREALIS - KONDITIONALER NEBENSATZ (Konjunktiv II)
    # =========================================================================
    # KRITERIUM: Tense=Past UND dep_ in [advcl, mo] (adverbial clause / modifier)
    # BEDEUTUNG: Bedingungssatz (wenn-Satz) oder Modifier in Konditionalsatz
    # BEISPIEL: "Wenn ich reich wäre [advcl/mo], ..."
    # HINWEIS: 'mo' wird verwendet wenn das Verb in einem wenn-Satz als Modifier fungiert
    if tense == "Past" and dep in ["advcl", "mo"]:
        # Zusätzliche Prüfung: Ist das HEAD-Verb auch K2? → dann ist dies ein Bedingungssatz
        head_morph = head.morph.to_dict()
        if (
            dep == "mo"
            and head_morph.get("Mood") == "Sub"
            and head_morph.get("Tense") == "Past"
        ):
            return {
                "typ": "Konjunktiv II",
                "funktion": "Irrealis (Bedingung)",
                "vorschlag": 'Beschreiben Sie die reale Situation oder verwenden Sie "wenn...dann"',
            }
        elif dep == "advcl":
            return {
                "typ": "Konjunktiv II",
                "funktion": "Irrealis (Bedingung)",
                "vorschlag": 'Beschreiben Sie die reale Situation oder verwenden Sie "wenn...dann"',
            }

    # =========================================================================
    # REGEL 3: IRREALIS - KONDITIONALER HAUPTSATZ (Konjunktiv II)
    # =========================================================================
    # KRITERIUM: Tense=Past UND dep_=ROOT UND hat Child mit dep_=advcl
    # BEDEUTUNG: Hauptsatz mit abhängigem Konditionalsatz
    # BEISPIEL: "...hätte [ROOT] ich ein Haus" (mit "wenn..."-Satz als advcl)
    if tense == "Past" and dep == "ROOT":
        # Prüfe ob ein Kind advcl ist
        if any(child.dep_ == "advcl" for child in token.children):
            return {
                "typ": "Konjunktiv II",
                "funktion": "Irrealis (Hypothese)",
                "vorschlag": 'Verwenden Sie "vielleicht" oder "möglicherweise" mit Präsens',
            }

    # =========================================================================
    # REGEL 4: WÜRDE-KONSTRUKTION (Konjunktiv II)
    # =========================================================================
    # KRITERIUM: Lemma=werden UND Tense=Past
    # BEDEUTUNG: Ersatzform für Konjunktiv II
    # BEISPIEL: "würde gehen", "würde sagen"
    if token.lemma_ == "werden" and tense == "Past":
        return {
            "typ": "Würde-Konstruktion (Konjunktiv II)",
            "funktion": "Hypothese/Konditionalis",
            "vorschlag": 'Verwenden Sie Präsens: "Wenn...dann" oder "vielleicht"',
        }

    # =========================================================================
    # REGEL 5: MODALVERB-KONJUNKTIV (Höflichkeit)
    # =========================================================================
    # KRITERIUM: pos_=AUX UND Tense=Past (Modalverben sind AUX in spaCy)
    # BEDEUTUNG: Höfliche oder vorsichtige Formulierung
    # BEISPIEL: "könnte", "sollte", "müsste", "dürfte"
    if token.pos_ == "AUX" and tense == "Past" and token.lemma_ != "werden":
        return {
            "typ": "Konjunktiv II (Modal)",
            "funktion": "Höflichkeit/Möglichkeit",
            "vorschlag": 'Formulieren Sie direkt: "Bitte" oder "Es ist möglich"',
        }

    # =========================================================================
    # REGEL 6: OPTATIV/WUNSCH (Konjunktiv I)
    # =========================================================================
    # KRITERIUM: Tense=Pres UND dep_=ROOT
    # BEDEUTUNG: Wunschsatz oder Ausruf
    # BEISPIEL: "Es lebe [ROOT] der König!", "Möge [ROOT] er gesund werden"
    if tense == "Pres" and dep == "ROOT":
        return {
            "typ": "Konjunktiv I",
            "funktion": "Wunsch/Optativ",
            "vorschlag": 'Formulieren Sie direkt: "Ich wünsche mir" oder "Ich hoffe"',
        }

    # =========================================================================
    # REGEL 7: HABEN/SEIN PERFEKT-KONSTRUKTION (Konjunktiv II)
    # =========================================================================
    # KRITERIUM: Lemma=haben/sein UND Tense=Past UND hat Partizip-Kind
    # BEDEUTUNG: Vergangenheitshypothese
    # BEISPIEL: "hätte gesagt", "wäre gekommen"
    if token.lemma_ in ["haben", "sein"] and tense == "Past":
        # Prüfe ob ein Kind ein Partizip ist
        if any(child.tag_ == "VVPP" for child in token.children):
            return {
                "typ": "Konjunktiv II (Perfekt)",
                "funktion": "Vergangenheitshypothese",
                "vorschlag": "Beschreiben Sie was wirklich passiert ist",
            }
        # Einfaches haben/sein im K2
        else:
            return {
                "typ": "Konjunktiv II",
                "funktion": "Irrealis/Hypothese",
                "vorschlag": 'Verwenden Sie "vielleicht" oder "möglicherweise" mit Präsens',
            }

    # =========================================================================
    # REGEL 8: KONJUNKTIV I IN NEBENSÄTZEN (Allgemein)
    # =========================================================================
    # KRITERIUM: Tense=Pres UND dep_ ist Nebensatz-Marker
    # BEDEUTUNG: Allgemeiner K1 in untergeordnetem Satz
    # BEISPIEL: Relativsätze, Finalsätze, etc.
    if tense == "Pres" and dep in ["acl", "relcl", "advcl", "csubj"]:
        return {
            "typ": "Konjunktiv I",
            "funktion": "Nebensatz",
            "vorschlag": "Verwenden Sie die einfache Wirklichkeitsform (Indikativ)",
        }

    # =========================================================================
    # FALLBACK: UNKLARER KONTEXT
    # =========================================================================
    # Wenn keine der obigen Regeln greift
    if tense == "Pres":
        return {
            "typ": "Konjunktiv I",
            "funktion": "Unbestimmt",
            "vorschlag": "Verwenden Sie die einfache Wirklichkeitsform (Indikativ)",
        }
    elif tense == "Past":
        return {
            "typ": "Konjunktiv II",
            "funktion": "Unbestimmt",
            "vorschlag": "Verwenden Sie die einfache Wirklichkeitsform (Indikativ)",
        }
    else:
        return {
            "typ": "Konjunktiv",
            "funktion": "Unbekannt",
            "vorschlag": "Verwenden Sie die einfache Wirklichkeitsform (Indikativ)",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import spacy

    testfaelle = [
        "Er sagte, er sei krank und habe Fieber.",
        "Wenn ich reich wäre, hätte ich ein Haus.",
        "Es lebe der König!",
    ]

    nlp = spacy.load("de_core_news_lg")

    for test_text in testfaelle:
        logger.info("Text: %s", test_text)
        doc = nlp(test_text)
        results = check_rule(doc)
        for error in results:
            logger.error(" -> %s", error)
        logger.info("")
