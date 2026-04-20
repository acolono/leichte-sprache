"""
Advanced rule for idiom detection using NLP techniques.
Uses semantic analysis, syntactic patterns, and contextual evaluation.
"""

import re
from typing import Any, Dict, List, Set, Tuple

import spacy
from spacy.tokens import Doc, Span, Token

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

import logging

logger = logging.getLogger(__name__)
REDEWENDUNGEN: Dict[str, Dict[str, Any]] = {
    "die flinte ins korn werfen": {
        "alternative": "aufgeben",
        "schwierigkeit": 0.9,
        "kategorie": "aufgeben",
    },
    "ins gras beißen": {
        "alternative": "sterben",
        "schwierigkeit": 0.8,
        "kategorie": "tod",
    },
    "den löffel abgeben": {
        "alternative": "sterben",
        "schwierigkeit": 0.8,
        "kategorie": "tod",
    },
    "über den berg sein": {
        "alternative": "das Schlimmste überstanden haben",
        "schwierigkeit": 0.7,
        "kategorie": "erfolg",
    },
    "auf wolke sieben schweben": {
        "alternative": "sehr glücklich sein",
        "schwierigkeit": 0.8,
        "kategorie": "gefühl",
    },
    "eine sache auf eis legen": {
        "alternative": "etwas verschieben",
        "schwierigkeit": 0.7,
        "kategorie": "verzögern",
    },
    "das handtuch werfen": {
        "alternative": "aufgeben",
        "schwierigkeit": 0.8,
        "kategorie": "aufgeben",
    },
    "die katze aus dem sack lassen": {
        "alternative": "ein Geheimnis verraten",
        "schwierigkeit": 0.9,
        "kategorie": "enthüllen",
    },
    "den nagel auf den kopf treffen": {
        "alternative": "genau richtig liegen",
        "schwierigkeit": 0.8,
        "kategorie": "korrektheit",
    },
    "jemandem einen bären aufbinden": {
        "alternative": "jemanden anlügen",
        "schwierigkeit": 0.9,
        "kategorie": "täuschung",
    },
    "die ohren steif halten": {
        "alternative": "nicht aufgeben",
        "schwierigkeit": 0.8,
        "kategorie": "durchhalten",
    },
    "auf dem holzweg sein": {
        "alternative": "falsch liegen",
        "schwierigkeit": 0.7,
        "kategorie": "irrtum",
    },
    "den mund zu voll nehmen": {
        "alternative": "übertreiben",
        "schwierigkeit": 0.7,
        "kategorie": "übertreibung",
    },
    "butter bei die fische geben": {
        "alternative": "zur Sache kommen",
        "schwierigkeit": 0.9,
        "kategorie": "direktheit",
    },
    "zwei fliegen mit einer klappe schlagen": {
        "alternative": "zwei Probleme auf einmal lösen",
        "schwierigkeit": 0.8,
        "kategorie": "effizienz",
    },
    "aus einer mücke einen elefanten machen": {
        "alternative": "etwas übertreiben",
        "schwierigkeit": 0.8,
        "kategorie": "übertreibung",
    },
    "ins fettnäpfchen treten": {
        "alternative": "sich blamieren",
        "schwierigkeit": 0.8,
        "kategorie": "fehler",
    },
    "auf dem schlauch stehen": {
        "alternative": "etwas nicht verstehen",
        "schwierigkeit": 0.7,
        "kategorie": "verständnis",
    },
    "die nase voll haben": {
        "alternative": "genug haben",
        "schwierigkeit": 0.6,
        "kategorie": "überdruss",
    },
    "den kopf in den sand stecken": {
        "alternative": "Probleme ignorieren",
        "schwierigkeit": 0.8,
        "kategorie": "vermeidung",
    },
    "mit dem kopf durch die wand wollen": {
        "alternative": "zu stur sein",
        "schwierigkeit": 0.8,
        "kategorie": "sturheit",
    },
    # ML-ENHANCED: Neue Redewendungen aus text_beispiel.txt
    "die spreu vom weizen trennen": {
        "alternative": "das Wichtige vom Unwichtigen unterscheiden",
        "schwierigkeit": 0.9,
        "kategorie": "unterscheidung",
    },
    "das kind mit dem bade ausschütten": {
        "alternative": "etwas Gutes zusammen mit etwas Schlechtem wegwerfen",
        "schwierigkeit": 0.9,
        "kategorie": "übertreibung",
    },
    "das fähnlein in den wind hängen": {
        "alternative": "seine Meinung schnell ändern",
        "schwierigkeit": 0.9,
        "kategorie": "opportunismus",
    },
    "über den tellerrand blicken": {
        "alternative": "über das Gewohnte hinaus denken",
        "schwierigkeit": 0.8,
        "kategorie": "perspektive",
    },
    "auf herz und nieren prüfen": {
        "alternative": "sehr gründlich untersuchen",
        "schwierigkeit": 0.8,
        "kategorie": "prüfung",
    },
    "im sande verlaufen": {
        "alternative": "erfolglos beenden",
        "schwierigkeit": 0.7,
        "kategorie": "scheitern",
    },
}

FIGURATIVE_EXPRESSIONS = {
    "unter vier augen": {"alternative": "allein sprechen", "schwierigkeit": 0.6},
    "mit offenen karten spielen": {"alternative": "ehrlich sein", "schwierigkeit": 0.7},
    "aus dem stegreif": {"alternative": "spontan", "schwierigkeit": 0.8},
    "ins schwarze treffen": {"alternative": "genau richtig sein", "schwierigkeit": 0.7},
    "auf nummer sicher gehen": {"alternative": "vorsichtig sein", "schwierigkeit": 0.6},
    "grünes licht geben": {"alternative": "erlauben", "schwierigkeit": 0.5},
    "hinter verschlossenen türen": {"alternative": "geheim", "schwierigkeit": 0.7},
    "über nacht": {"alternative": "plötzlich", "schwierigkeit": 0.4},
    # ML-ENHANCED: Bildhafte Ausdrücke aus text_beispiel.txt
    "inflationär beschwören": {
        "alternative": "immer wieder sagen",
        "schwierigkeit": 0.8,
    },
    "als postfaktisch charakterisiert": {
        "alternative": "als Zeit nach der Wahrheit beschrieben",
        "schwierigkeit": 0.9,
    },
    "paralysieren lassen": {"alternative": "lähmen lassen", "schwierigkeit": 0.7},
    "fähnlein in den wind hängen": {
        "alternative": "schnell die Meinung ändern",
        "schwierigkeit": 0.9,
    },
}

METAPHORICAL_PATTERNS = [
    r"\b(herz|kopf|hand|auge|ohr)\s+(ist|war|wird)\s+\w+",
    r"\b(stein|berg|wolke|himmel|erde)\s+(fällt|steigt|sinkt)",
    r"\b(feuer|wasser|luft|wind)\s+(brennt|fließt|weht)",
    r"\b(türe?n?|fenster|brücke)\s+(öffnen|schließen|bauen)",
    # ML-ENHANCED: Erweiterte Patterns für text_beispiel.txt
    r"\b(spreu|weizen)\s+(vom|trennen)",  # "Spreu vom Weizen trennen"
    r"\b(kind)\s+(mit\s+dem\s+bade?)\s+(ausschütten)",  # "Kind mit dem Bade ausschütten"
    r"\b(tellerrand)\s+(blicken|schauen)",  # "über den Tellerrand blicken"
    r"\b(fähnlein)\s+(in\s+den\s+wind)\s+(hängen)",  # "Fähnlein in den Wind hängen"
    r"\b(herz\s+und\s+nieren)\s+(prüfen)",  # "auf Herz und Nieren prüfen"
    r"\b(sande?)\s+(verlaufen)",  # "im Sande verlaufen"
    r"\b(lippenbekenntnisse|lippenbekenntnis)",  # Metaphorisches Wort
    r"\b(mammutaufgabe)",  # Tier-Metapher
    r"\b(mosaiksteine?)",  # Bild-Metapher
]

BODY_PART_VERBS = {
    "kopf": ["zerbrechen", "rauchen", "platzen", "schwirren"],
    "herz": ["bluten", "brennen", "schlagen", "zerreißen"],
    "hand": ["brennen", "jucken", "zittern", "binden"],
    "auge": ["tränen", "brennen", "stechen", "zudrücken"],
    "nase": ["bluten", "jucken", "rümpfen", "voll haben"],
}


def _detect_metaphorical_language(sent: Span) -> List[Tuple[str, str, float]]:
    """Detects metaphorical language patterns in a sentence."""
    metaphern = []
    sent_text = sent.text.lower()

    for pattern in METAPHORICAL_PATTERNS:
        matches = re.finditer(pattern, sent_text)
        for match in matches:
            phrase = match.group()
            schwierigkeit = 0.7
            metaphern.append((phrase, "metaphorische Sprache", schwierigkeit))

    for koerperteil, verben in BODY_PART_VERBS.items():
        for verb in verben:
            if koerperteil in sent_text and verb in sent_text:
                if not any(
                    conj in sent_text for conj in ["krank", "schmerz", "verletzt"]
                ):
                    phrase = f"{koerperteil} {verb}"
                    metaphern.append((phrase, "körperteil-metapher", 0.8))

    return metaphern


def _evaluate_idiom_context(token: Token, doc: Doc, phrase: str) -> float:
    """Evaluates the context of an idiom."""
    sent = next(sent for sent in doc.sents if token in sent)
    score = 0.0

    satzlaenge = len([t for t in sent if not t.is_punct and not t.is_space])
    if satzlaenge > 15:
        score += 0.2

    if any(
        wort in sent.text.lower()
        for wort in ["jedoch", "dennoch", "allerdings", "freilich"]
    ):
        score += 0.3

    bildhafte_woerter = sum(
        1
        for t in sent
        if any(
            tier in t.text.lower()
            for tier in ["hund", "katze", "maus", "schwein", "vogel"]
        )
    )
    if bildhafte_woerter > 1:
        score += 0.4

    return score


def _find_contiguous_phrase(
    token: Token, doc: Doc, phrase_tokens: List[str]
) -> Tuple[Span, float]:
    """Finds contiguous phrases in the text."""
    sent = next(sent for sent in doc.sents if token in sent)
    sent_tokens = [t.text.lower() for t in sent if not t.is_punct]

    for i in range(len(sent_tokens) - len(phrase_tokens) + 1):
        window = sent_tokens[i : i + len(phrase_tokens)]

        matches = sum(
            1
            for j, phrase_token in enumerate(phrase_tokens)
            if j < len(window) and phrase_token == window[j]
        )
        match_ratio = matches / len(phrase_tokens)

        if match_ratio >= 0.8:
            start_idx = i
            end_idx = i + len(phrase_tokens)
            span_tokens = [t for t in sent if not t.is_punct][start_idx:end_idx]
            if span_tokens:
                span = doc[span_tokens[0].i : span_tokens[-1].i + 1]
                return span, match_ratio

    return None, 0.0


def _is_literal_usage(phrase: str, kontext: str) -> bool:
    """Checks if a phrase is used literally or metaphorically."""
    literale_indikatoren = [
        "wirklich",
        "tatsächlich",
        "buchstäblich",
        "echt",
        "konkret",
    ]

    return any(indikator in kontext.lower() for indikator in literale_indikatoren)


def check_rule(doc: Doc) -> List[str]:
    """
    Advanced idiom detection with semantic and syntactic analysis:
    1. Exact phrase matching with context evaluation
    2. Flexible pattern matching for variants
    3. Metaphorical language detection
    4. Literal vs. figurative distinction

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of context-specific improvement suggestions
    """
    errors = []
    gefundene_phrasen: Set[str] = set()
    text_lower = doc.text.lower()

    for phrase, info in REDEWENDUNGEN.items():
        # Use word-boundary-aware matching instead of substring
        phrase_pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(phrase_pattern, text_lower) and phrase not in gefundene_phrasen:
            phrase_tokens = phrase.split()

            phrase_span, match_ratio = _find_contiguous_phrase(
                doc[0], doc, phrase_tokens
            )

            if phrase_span and not _is_literal_usage(
                phrase, phrase_span.sent.text
            ):
                alternative = info["alternative"]
                schwierigkeit = info["schwierigkeit"]
                kategorie = info["kategorie"]

                kontext_score = _evaluate_idiom_context(
                    phrase_span[0], doc, phrase
                )
                gesamt_score = schwierigkeit + kontext_score

                if gesamt_score >= 0.8:
                    priorität = "hoch" if gesamt_score >= 1.0 else "mittel"
                    errors.append(
                        f'Redewendung "{phrase}" (Kategorie: {kategorie}, Priorität: {priorität}). '
                        f'Wörtlich: "{alternative}".'
                    )
                gefundene_phrasen.add(phrase)

    for phrase, info in FIGURATIVE_EXPRESSIONS.items():
        phrase_pattern = r"\b" + re.escape(phrase) + r"\b"
        if re.search(phrase_pattern, text_lower) and phrase not in gefundene_phrasen:
            alternative = info["alternative"]
            schwierigkeit = info["schwierigkeit"]

            if schwierigkeit >= 0.6:
                errors.append(
                    f'Bildhafter Ausdruck "{phrase}". Wörtlich: "{alternative}".'
                )
            gefundene_phrasen.add(phrase)

    for sent in doc.sents:
        metaphern = _detect_metaphorical_language(sent)
        for phrase, typ, schwierigkeit in metaphern:
            if phrase not in gefundene_phrasen and schwierigkeit >= 0.7:
                errors.append(
                    f'Metaphorische Sprache "{phrase}" ({typ}). '
                    f"Verwenden Sie wörtliche Beschreibungen."
                )
                gefundene_phrasen.add(phrase)

    return errors


def extended_idiom_analysis(text: str) -> Dict[str, Any]:
    """Performs a detailed analysis of all idioms."""
    import spacy

    nlp = spacy.load("de_core_news_lg")
    doc = nlp(text)

    gefundene_redewendungen = []
    alle_woerter = len([t for t in doc if not t.is_punct and not t.is_space])

    text_lower = text.lower()

    for phrase, info in REDEWENDUNGEN.items():
        if phrase in text_lower:
            gefundene_redewendungen.append(
                {
                    "phrase": phrase,
                    "alternative": info["alternative"],
                    "schwierigkeit": info["schwierigkeit"],
                    "kategorie": info["kategorie"],
                    "typ": "redewendung",
                }
            )

    for phrase, info in FIGURATIVE_EXPRESSIONS.items():
        if phrase in text_lower:
            gefundene_redewendungen.append(
                {
                    "phrase": phrase,
                    "alternative": info["alternative"],
                    "schwierigkeit": info["schwierigkeit"],
                    "kategorie": "bildlich",
                    "typ": "bildhafter_ausdruck",
                }
            )

    for sent in doc.sents:
        metaphern = _detect_metaphorical_language(sent)
        for phrase, typ, schwierigkeit in metaphern:
            gefundene_redewendungen.append(
                {
                    "phrase": phrase,
                    "alternative": "wörtlich beschreiben",
                    "schwierigkeit": schwierigkeit,
                    "kategorie": "metapher",
                    "typ": typ,
                }
            )

    bildhafte_dichte = (
        (len(gefundene_redewendungen) / alle_woerter * 100) if alle_woerter else 0
    )

    return {
        "gesamt_woerter": alle_woerter,
        "redewendungen_anzahl": len(gefundene_redewendungen),
        "bildhafte_dichte": round(bildhafte_dichte, 1),
        "gefundene_redewendungen": gefundene_redewendungen,
        "bewertung": "sehr hoch"
        if bildhafte_dichte > 8
        else "hoch"
        if bildhafte_dichte > 4
        else "mittel"
        if bildhafte_dichte > 2
        else "niedrig",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nlp = spacy.load("de_core_news_lg")

    test_texte = [
        "Er hat die Flinte ins Korn geworfen und das Handtuch geschmissen.",
        "Sie sollten nicht den Mund zu voll nehmen bei dieser Aufgabe.",
        "Unter vier Augen können wir offen reden.",
        "Das Auto steht vor dem Haus.",
        "Er traf den Nagel auf den Kopf mit seiner Aussage.",
        "Aus einer Mücke wurde hier ein Elefant gemacht.",
        "Die Kinder spielen im Garten und haben Spaß.",
    ]

    logger.info("--- Test: Erweiterte Regel Redewendungen ---")
    for i, text in enumerate(test_texte):
        logger.info("\nSatz %s: '%s'", i + 1, text)
        doc = nlp(text)
        results = check_rule(doc)

        if results:
            for error in results:
                logger.error(" %s", error)
        else:
            logger.info(" Keine Redewendungen oder bildhafte Sprache gefunden.")
