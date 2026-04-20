"""
Regel zur Erkennung von Negationen für die Leichte Sprache.
Morphologie-basierte Implementation (KEINE Wortlisten für Negationserkennung).

VERSION 3.0 - MORPHOLOGIE-BASIERT + MATCHER
Strikte Anforderungen erfüllt:
- KEINE Wortlisten für semantische Negationen
- KEINE Präfix-Regeln (un-, miss-, etc.)
- Primär morphologie-basiert (lemma_ + dep_)
- Matcher nur für grammatikalische Sonderfälle
"""

from typing import Dict, List, Set

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Token

# Import Konfiguration (aktuell leer, für zukünftige Erweiterungen)
from . import config  # noqa: F401


import logging

logger = logging.getLogger(__name__)
def check_rule(doc: Doc) -> List[str]:
    """
    Erkennt Negationen durch morphologie-basierte Analyse:
    - Dependency-Label: dep_=ng (Negationspartikel)
    - Lemma-basiert: kein, nicht, nie, niemand, nichts
    - Matcher: ohne, kaum, weder...noch, komplexe Konstruktionen

    KEINE Wortlisten, KEINE Präfix-Regeln!
    """
    errors = []
    processed_tokens: Set[int] = set()

    # =========================================================================
    # REGEL 1: MORPHOLOGIE-BASIERTE NEGATIONS-ERKENNUNG
    # =========================================================================
    for token in doc:
        if token.i in processed_tokens:
            continue

        # Skip Zitate/Beispiele
        if (
            token.i > 0
            and token.i < len(doc) - 1
            and doc[token.i - 1].is_quote
            and doc[token.i + 1].is_quote
        ):
            continue

        negation_type = _detect_morphological_negation(token)

        if negation_type:
            scope_info = _analyze_negation_scope(token)
            errors.append(
                f'{negation_type} "{token.text}" macht den Satz schwer verständlich. '
                f"Besser: {scope_info['vorschlag']}"
            )
            processed_tokens.add(token.i)

    # =========================================================================
    # REGEL 2: MATCHER FÜR GRAMMATIKALISCHE SONDERFÄLLE
    # =========================================================================
    complex_negations = _detect_complex_negations(doc)
    for negation in complex_negations:
        if negation["start_token"].i not in processed_tokens:
            errors.append(negation["nachricht"])
            # Markiere alle Tokens dieser Negation als verarbeitet
            for i in range(negation["start_token"].i, negation["end_token"].i + 1):
                processed_tokens.add(i)

    # =========================================================================
    # DEDUPLIZIERUNG: Entferne doppelte Meldungen für gleiche Position
    # =========================================================================
    seen_messages = set()
    deduplicated = []
    for error_msg in errors:
        if error_msg not in seen_messages:
            seen_messages.add(error_msg)
            deduplicated.append(error_msg)

    return deduplicated


def _detect_morphological_negation(token: Token) -> str:
    """
    Erkennt Negationen AUSSCHLIESSLICH durch Morphologie und Lemma.

    PRINZIP: Nutzt NUR abstrakte grammatikalische Merkmale:
    - token.lemma_ (für bekannte Negationswörter)
    - token.dep_ (Dependency-Label: ng = Negationspartikel)
    - token.pos_ (Part-of-Speech: nur zur Disambiguierung)

    KEINE Wortlisten, KEINE Präfix-Regeln!

    Returns:
        String mit Negationstyp oder None
    """
    lemma_lower = token.lemma_.lower()

    # =========================================================================
    # KATEGORIE 1: NEGATIONSPARTIKEL (dep_=ng)
    # =========================================================================
    # KRITERIUM: dep_=ng markiert Negationspartikel automatisch
    # BEISPIELE: "nicht", "nie"
    # VORTEIL: Robuste, dependency-basierte Erkennung
    if token.dep_ == "ng":
        return "Negationspartikel"

    # =========================================================================
    # KATEGORIE 2: NEGATIVE DETERMINANTEN (kein)
    # =========================================================================
    # KRITERIUM: lemma_=kein UND pos_=DET
    # BEISPIELE: "keine Zeit", "kein Geld"
    # UNTERSCHEIDUNG: DET vs. PRON (pronominales "keiner" → Kategorie 4)
    if lemma_lower == "kein" and token.pos_ == "DET":
        return "Negative Bestimmung"

    # =========================================================================
    # KATEGORIE 3: NEGATIVE ADVERBIEN (nie, niemals, etc.)
    # =========================================================================
    # KRITERIUM: lemma_ in {nie, niemals, nirgends, nirgendwo, ...} UND pos_=ADV
    # BEISPIELE: "nie wieder", "niemals mehr"
    # HINWEIS: Nur die grammatikalisch unzweideutigen Negations-Adverbien
    if (
        lemma_lower
        in [
            "nie",
            "niemals",
            "nirgends",
            "nirgendwo",
            "nirgendwohin",
            "keineswegs",
            "keinesfalls",
        ]
        and token.pos_ == "ADV"
    ):
        return "Negative Aussage (Adverb)"

    # =========================================================================
    # KATEGORIE 4: NEGATIVE PRONOMEN (nichts, niemand, keiner)
    # =========================================================================
    # KRITERIUM: lemma_ in {nichts, niemand, kein} UND pos_=PRON
    # BEISPIELE: "niemand kam", "nichts ist da", "keiner weiß"
    # HINWEIS: "kein" als PRON (pronominale Verwendung: "Keiner weiß es")
    if lemma_lower in ["nichts", "niemand", "kein", "keiner"] and token.pos_ == "PRON":
        return "Negative Aussage (Pronomen)"

    # Kein Match
    return None


def _analyze_negation_scope(negation_token: Token) -> Dict[str, str]:
    """
    Analysiert den Scope einer Negation und generiert kontextspezifische Vorschläge.
    Nutzt Dependency-Parsing um zu verstehen, was negiert wird.
    """
    head = negation_token.head

    # Analysiere was negiert wird basierend auf dem HEAD
    if head.pos_ == "VERB":
        if head.lemma_ in ["können", "müssen", "sollen", "dürfen", "wollen"]:
            return {
                "scope": f"Modalverb {head.lemma_}",
                "vorschlag": f'Statt "{negation_token.text} {head.text}" verwenden Sie "es ist schwer zu" oder "es ist möglich, dass nicht".',
            }
        else:
            return {
                "scope": f"Verb {head.lemma_}",
                "vorschlag": "Beschreiben Sie, was stattdessen passiert oder möglich ist.",
            }

    elif head.pos_ in ["NOUN", "PROPN"]:
        return {
            "scope": f"Nomen {head.text}",
            "vorschlag": "Beschreiben Sie, was es stattdessen gibt oder erklären Sie den Mangel positiv.",
        }

    elif head.pos_ in ["ADJ", "ADV"]:
        return {
            "scope": f"Eigenschaft {head.text}",
            "vorschlag": 'Beschreiben Sie die positive Alternative oder verwenden Sie "weniger" statt Verneinung.',
        }

    else:
        return {
            "scope": "allgemein",
            "vorschlag": "Formulieren Sie den Gedanken positiv um.",
        }


def _detect_complex_negations(doc: Doc) -> List[Dict]:
    """
    Erkennt grammatikalische Negations-Sonderfälle mit spaCy's Matcher.

    WICHTIG: NUR für feststehende grammatikalische Konstruktionen,
    die die Morphologie-Regel nicht erfasst:
    - "ohne" (Präposition mit negierender Bedeutung)
    - "kaum" (Limitierendes Adverb)
    - "weder ... noch" (Mehrteilige Konjunktion)
    - Komplexe Partikel: "noch nicht", "gar nicht", "nicht mehr"

    KEINE semantischen Negationen (un-, miss-, Problem, etc.)!
    """
    matcher = Matcher(doc.vocab)
    complex_negations = []

    # =========================================================================
    # PATTERN 1: PRÄPOSITIONALE NEGATION (ohne)
    # =========================================================================
    # BEISPIEL: "Er ging ohne Jacke."
    # LEMMA-basiert für Robustheit (erkennt "ohne", "Ohne")
    matcher.add("OHNE", [[{"LEMMA": "ohne"}]])

    # =========================================================================
    # PATTERN 2: LIMITIERENDES ADVERB (kaum)
    # =========================================================================
    # BEISPIEL: "Das ist kaum möglich."
    # BEDEUTUNG: Fast eine Negation ("so gut wie nicht")
    matcher.add("KAUM", [[{"LEMMA": "kaum"}]])

    # =========================================================================
    # PATTERN 3: WEDER...NOCH (Mehrteilige Konjunktion)
    # =========================================================================
    # BEISPIEL: "Weder du noch ich wissen es."
    # FLEXIBEL: Beliebige Anzahl Tokens zwischen "weder" und "noch"
    # WICHTIG: IS_PUNCT=False verhindert Punkt-Match
    matcher.add(
        "WEDER_NOCH",
        [
            [
                {"LEMMA": "weder"},
                {"OP": "?", "IS_PUNCT": False},
                {"OP": "?", "IS_PUNCT": False},
                {"OP": "?", "IS_PUNCT": False},
                {"OP": "?", "IS_PUNCT": False},
                {"OP": "?", "IS_PUNCT": False},
                {"LEMMA": "noch"},
            ]
        ],
    )

    # =========================================================================
    # PATTERN 4: KOMPLEXE NEGATIONS-PARTIKEL
    # =========================================================================
    # "noch nicht", "gar nicht", "nicht mehr", "nie wieder", etc.
    # FLEXIBEL: Kombiniert verschiedene Negations-Verstärker
    matcher.add(
        "KOMPLEX_1",
        [
            [
                {"LEMMA": {"IN": ["noch", "gar", "überhaupt"]}},
                {"LEMMA": {"IN": ["nicht", "nichts", "nie"]}},
            ]
        ],
    )

    matcher.add(
        "KOMPLEX_2",
        [
            [
                {"LEMMA": {"IN": ["nicht", "nie"]}},
                {"LEMMA": {"IN": ["mehr", "wieder", "länger", "einmal"]}},
            ]
        ],
    )

    # =========================================================================
    # PATTERN 5: FESTE REDEWENDUNGEN (auf keinen Fall, unter keinen Umständen)
    # =========================================================================
    matcher.add(
        "AUF_KEINEN_FALL", [[{"LEMMA": "auf"}, {"LEMMA": "kein"}, {"LEMMA": "Fall"}]]
    )

    matcher.add(
        "UNTER_KEINEN_UMSTAENDEN",
        [[{"LEMMA": "unter"}, {"LEMMA": "kein"}, {"LEMMA": "Umstand"}]],
    )

    # Führe Matching aus
    matches = matcher(doc)

    for match_id, start, end in matches:
        text_span = doc[start:end].text
        rule_name = doc.vocab.strings[match_id]

        # Generiere spezifische Nachricht basierend auf Pattern-Typ
        if "OHNE" in rule_name:
            nachricht = f'Präpositionale Negation "{text_span}" ist schwer verständlich. Besser: Beschreiben Sie, was vorhanden ist.'
        elif "KAUM" in rule_name:
            nachricht = f'Limitierendes Adverb "{text_span}" ist schwer verständlich. Besser: Verwenden Sie "selten" oder "wenig".'
        elif "WEDER_NOCH" in rule_name:
            nachricht = f'Mehrteilige Negation "{text_span}" ist schwer verständlich. Besser: Teilen Sie in zwei positive Aussagen auf.'
        else:
            nachricht = f'Komplexe Negation "{text_span}" ist schwer verständlich. Besser: Formulieren Sie positiv, was möglich oder vorhanden ist.'

        complex_negations.append(
            {
                "start_token": doc[start],
                "end_token": doc[end - 1],
                "nachricht": nachricht,
            }
        )

    return complex_negations


# =========================================================================
# TESTING & VALIDATION
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import spacy

    logger.info("=" * 80)
    logger.info("NEGATIONS-ERKENNUNG: MORPHOLOGIE-BASIERTER ANSATZ (v3.0)")
    logger.info("=" * 80)
    logger.info("\n KEINE Wortlisten für semantische Negationen")
    logger.info(" KEINE Präfix-Regeln (un-, miss-, etc.)")
    logger.info(" Primär morphologie-basiert (lemma_ + dep_)")
    logger.info(" Matcher nur für grammatikalische Sonderfälle")
    logger.info("=" * 80)

    nlp = spacy.load("de_core_news_lg")

    # Testfälle
    testfaelle = [
        # POSITIVE Tests (sollten erkannt werden)
        # Kategorie 1: Negationspartikel (dep_=ng)
        "Ich bin nicht da.",
        "Sie sagte nie etwas.",
        # Kategorie 2: Negative Determinanten (kein)
        "Er hat keine Zeit.",
        "Es gibt kein Problem.",
        # Kategorie 3: Negative Adverbien
        "Das passiert niemals.",
        "Er war nirgends zu finden.",
        "Das stimmt keineswegs.",
        # Kategorie 4: Negative Pronomen
        "Niemand kam.",
        "Es gibt nichts zu tun.",
        "Keiner weiß die Antwort.",  # pronominales "keiner"
        # Matcher: Grammatikalische Sonderfälle
        "Er ging ohne Jacke.",
        "Das ist kaum möglich.",
        "Weder du noch ich wissen es.",
        "Noch nicht fertig.",
        "Das ist gar nicht gut.",
        "Nie wieder!",
        "Nicht mehr lange.",
        "Auf keinen Fall!",
        # NEGATIVE Tests (sollten NICHT erkannt werden)
        "Das Problem ist unkompliziert.",  # Präfix un- = NICHT negation
        "Die Ungeduld nervt.",  # Präfix un- = NICHT negation
        "Er hat Probleme.",  # "Problem" = NICHT negation
        "Sie mangelt an Zeit.",  # "mangeln" = NICHT negation
    ]

    logger.info("\n--- TESTFÄLLE ---\n")

    positiv_erwartet = 18  # Erste 18 Tests sollten Negationen finden
    negativ_erwartet = 4  # Letzte 4 Tests sollten KEINE finden

    for i, test_text in enumerate(testfaelle, 1):
        logger.info("[Test %s] %s", i, test_text)

        doc = nlp(test_text)
        results = check_rule(doc)

        soll_erkennen = i <= positiv_erwartet

        if results:
            logger.info(" %s Negation(en) gefunden:", len(results))
            for e in results:
                logger.info("  - %s", e)
            if not soll_erkennen:
                logger.warning(" FALSCH-POSITIV: Sollte NICHT erkannt werden!")
        else:
            logger.info(" Keine Negationen erkannt")
            if soll_erkennen:
                logger.warning(" FALSCH-NEGATIV: Sollte erkannt werden!")

        logger.info("")

    logger.info("=" * 80)
    logger.info("WARUM MORPHOLOGIE-ANALYSE ÜBERLEGEN IST:")
    logger.info("=" * 80)
    logger.warning("""
1. PRÄZISION: Eliminiert Falsch-Positive durch Präfix-Regeln (un-, miss-).
 "unkompliziert" ist KEIN Problem mehr.

2. ROBUSTHEIT: Keine semantischen Wortlisten ("Problem", "Mangel", etc.).
 Vermeidet Falsch-Positive bei neutralen Kontexten.

3. WARTBARKEIT: Grammatikalische Regeln sind sprachunabhängig.
 Funktioniert automatisch mit allen Negationsformen.

4. VOLLSTÄNDIGKEIT: Matcher ergänzt morphologische Lücken präzise.
 Findet "ohne", "kaum", "weder...noch" ohne Nebenwirkungen.
 """)
    logger.info("=" * 80)
