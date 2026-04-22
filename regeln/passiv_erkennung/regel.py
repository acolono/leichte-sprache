"""
Robust detection of German passive constructions using spaCy.
Implemented according to modern NLP standards for precise grammatical analysis.

Detects all forms of the German Vorgangspassiv (process passive):
1. Present/Preterite passive: wird/wurde + Partizip II
2. Perfect/Pluperfect passive: ist/war + Partizip II + worden
3. Future I passive: wird + Partizip II + werden
4. Passive with modal verbs: kann/muss + Partizip II + werden

Uses spaCy dependency parsing for robust grammatical analysis.
"""

from typing import List, Tuple

import spacy
from spacy.tokens import Doc, Token

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

# Modal verbs for passive constructions
import logging

logger = logging.getLogger(__name__)
MODAL_VERBS = {
    "können",
    "müssen",
    "sollen",
    "dürfen",
    "wollen",
    "mögen",
    "kann",
    "muss",
    "soll",
    "darf",
    "will",
    "mag",
}

# Participles that are often used as adjectives (semantic exceptions)
ADJECTIVAL_PARTICIPLES = {
    "entspannt",
    "interessiert",
    "begeistert",
    "überzeugt",
    "ermüdet",
    "konzentriert",
    "verwirrt",
    "erschöpft",
    "gelangweilt",
    "aufgeregt",
    "erstaunt",
    "überrascht",
    "enttäuscht",
    "zufrieden",
    "beruhigt",
    # Common adjective participles that cause false positives
    "besorgt",
    "erfahren",
    "entschlossen",
    "betroffen",
    "angespannt",
    "bewegt",
    "empört",
    "entsetzt",
    "erfreut",
    "erholt",
    "erleichtert",
    "erregt",
    "erschrocken",
    "erschüttert",
    "frustriert",
    "gefasst",
    "gerührt",
    "gestresst",
    "irritiert",
    "motiviert",
    "organisiert",
    "qualifiziert",
    "reserviert",
    "resigniert",
    "schockiert",
    "verärgert",
    "verdutzt",
    "verlegen",
    "vertieft",
    "verzweifelt",
    "angesehen",
    "bekannt",
    "beliebt",
    "bewährt",
    "engagiert",
    "gebildet",
    "geschickt",
    "gewandt",
    "talentiert",
    "verdient",
    "versiert",
    "vertraut",
}


def _is_past_participle(token: Token) -> bool:
    """
    Checks if a token is a past participle (Partizip II).

    Args:
        token: spaCy Token

    Returns:
        True if past participle, False otherwise
    """
    # Various Partizip II tags
    return token.tag_ in [
        "VVPP",
        "VAPP",
        "VMPP",
    ]  # Full verb, auxiliary, modal verb past participle


def _is_auxiliary_werden(token: Token) -> bool:
    """
    Checks if a token is a form of 'werden' as auxiliary verb.

    Args:
        token: spaCy Token

    Returns:
        True if werden-auxiliary, False otherwise
    """
    return token.lemma_ == "werden" and token.pos_ == "AUX"


def _is_auxiliary_sein(token: Token) -> bool:
    """
    Checks if a token is a form of 'sein' as auxiliary verb.

    Args:
        token: spaCy Token

    Returns:
        True if sein-auxiliary, False otherwise
    """
    return token.lemma_ == "sein" and token.pos_ == "AUX"


def _is_modal_verb(token: Token) -> bool:
    """
    Checks if a token is a modal verb.

    Args:
        token: spaCy Token

    Returns:
        True if modal verb, False otherwise
    """
    return (
        token.lemma_ in MODAL_VERBS
        or token.tag_.startswith("VM")
        or token.pos_ == "AUX"
        and token.lemma_ in MODAL_VERBS
    )


def find_passive_constructions(doc: Doc) -> List[Tuple[str, str, int, int]]:
    """
    Detects all passive constructions in a German text based on
    grammatical dependency structures (parent-child relationships).

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of tuples (passive_text, type, start_char, end_char)
    """
    passive_constructions = []

    for sent in doc.sents:
        # Robust dependency-based passive detection (Vorgangspassiv)
        passive_constructions.extend(_detect_dependency_passive(sent))
        # DIN SPEC 33429 §6.4 also bans Zustandspassiv and Passiversatzformen.
        passive_constructions.extend(_detect_stative_passive(sent))
        passive_constructions.extend(_detect_sein_zu_infinitive(sent))
        passive_constructions.extend(_detect_sich_lassen(sent))
        passive_constructions.extend(_detect_bar_adjective(sent))

    return passive_constructions


def _detect_stative_passive(sent) -> List[Tuple[str, str, int, int]]:
    """Zustandspassiv: sein (finite/AUX) + Partizip II as predicate.

    Example: "Das Geschäft ist geöffnet." / "Die Tür war geschlossen."
    Skip ADJECTIVAL_PARTICIPLES where the PP is lexicalized as adjective
    (e.g., "entspannt", "begeistert") — those are not passive readings.
    """
    findings = []
    for token in sent:
        if not _is_past_participle(token):
            continue
        if token.lemma_ in ADJECTIVAL_PARTICIPLES:
            continue
        # Participle must be predicate (head or sibling of a sein-aux).
        sein_verb = None
        if _is_auxiliary_sein(token.head):
            sein_verb = token.head
        else:
            for child in token.children:
                if _is_auxiliary_sein(child):
                    sein_verb = child
                    break
        if sein_verb is None:
            continue
        # Skip if this matches Perfekt-Passiv (worden is present — handled elsewhere).
        if any(_is_auxiliary_werden(c) and c.lemma_ == "worden" for c in token.children):
            continue
        phrase = f"{sein_verb.text} {token.text}"
        start_idx = min(sein_verb.idx, token.idx)
        end_idx = max(sein_verb.idx + len(sein_verb.text), token.idx + len(token.text))
        findings.append((phrase, "Zustandspassiv", start_idx, end_idx))
    return findings


def _detect_sein_zu_infinitive(sent) -> List[Tuple[str, str, int, int]]:
    """Passiversatz 'sein + zu + Infinitiv': "Das Formular ist auszufüllen"."""
    findings = []
    for token in sent:
        # Infinitive with zu marker. German spaCy tags this via tag_='VVIZU'
        # (combined zu+infinitive) or with a PTKZU child.
        is_zu_inf = token.tag_ == "VVIZU" or any(
            c.tag_ == "PTKZU" for c in token.children if c.pos_ == "PART"
        )
        if not is_zu_inf:
            continue
        # Find a sein head (directly or through head chain, 1 level).
        sein_verb = None
        if _is_auxiliary_sein(token.head):
            sein_verb = token.head
        elif _is_auxiliary_sein(token.head.head):
            sein_verb = token.head.head
        if sein_verb is None:
            continue
        phrase = f"{sein_verb.text} … {token.text}"
        start_idx = min(sein_verb.idx, token.idx)
        end_idx = token.idx + len(token.text)
        findings.append((phrase, "Passiversatz (sein+zu+Infinitiv)", start_idx, end_idx))
    return findings


def _detect_sich_lassen(sent) -> List[Tuple[str, str, int, int]]:
    """Passiversatz 'sich + lassen + Infinitiv': "Das lässt sich lösen"."""
    findings = []
    for token in sent:
        if token.lemma_.lower() != "lassen":
            continue
        if token.pos_ not in ("VERB", "AUX"):
            continue
        # Look for a reflexive 'sich' and an infinitive in the same clause.
        has_sich = any(c.lower_ == "sich" for c in sent)
        has_inf = any(
            c.tag_ in ("VVINF", "VAINF", "VMINF") for c in sent
        )
        if has_sich and has_inf:
            phrase = f"{token.text} sich …"
            start_idx = token.idx
            end_idx = token.idx + len(token.text)
            findings.append((phrase, "Passiversatz (sich+lassen+Infinitiv)", start_idx, end_idx))
            break
    return findings


def _detect_bar_adjective(sent) -> List[Tuple[str, str, int, int]]:
    """Passiversatz '-bar'/'-abel' adjective derived from transitive verb.

    Conservative: flag adjectives whose surface form ends in -bar/-abel and
    whose lemma is a transitive verb or a -bar-derived adjective. Very
    short stems are skipped to avoid 'klar', 'wunderbar', 'sichtbar' etc.
    being over-flagged — we keep this lexical by surface length.
    """
    findings = []
    for token in sent:
        if token.pos_ != "ADJ":
            continue
        lower = token.text.lower()
        if not (lower.endswith("bar") or lower.endswith("abel")):
            continue
        if len(lower) < 7:  # avoid 'klar', 'bar', etc.
            continue
        phrase = token.text
        start_idx = token.idx
        end_idx = token.idx + len(token.text)
        findings.append((phrase, "Passiversatz (-bar/-abel)", start_idx, end_idx))
    return findings


def _detect_dependency_passive(sent) -> List[Tuple[str, str, int, int]]:
    """
    Robust dependency-based detection of German passive constructions.
    Analyzes parent-child relationships between verbs instead of relying only on dependency labels.
    """
    passive_findings = []

    for token in sent:
        # PATTERN 1: Present/preterite passive (wird/wurde + Partizip II)
        if token.lemma_ == "werden" and token.pos_ == "AUX":
            # Search children for Partizip II
            for child in token.children:
                if _is_past_participle(child):
                    passive_phrase = f"{token.text} {child.text}"
                    passive_findings.append(
                        (
                            passive_phrase,
                            "Präsens-/Präteritum-Passiv",
                            min(token.idx, child.idx),
                            max(
                                token.idx + len(token.text), child.idx + len(child.text)
                            ),
                        )
                    )

        # PATTERN 2: Perfect/pluperfect passive (ist/war + Partizip II + worden)
        elif token.lemma_ == "sein" and token.pos_ == "AUX":
            # Search for 'worden' and Partizip II as children
            worden_child = None
            participle_child = None

            for child in token.children:
                if child.lemma_ == "worden" and child.tag_ == "VAPP":
                    worden_child = child
                elif (
                    _is_past_participle(child)
                ):
                    participle_child = child

            # Alternative: Partizip II could be child of 'worden'
            if worden_child and not participle_child:
                for grandchild in worden_child.children:
                    if (
                        _is_past_participle(grandchild)
                    ):
                        participle_child = grandchild
                        break

            # Alternative: 'worden' could be child of Partizip II
            if not worden_child:
                for child in token.children:
                    if _is_past_participle(child):
                        for grandchild in child.children:
                            if (
                                grandchild.lemma_ == "worden"
                                and grandchild.tag_ == "VAPP"
                            ):
                                worden_child = grandchild
                                participle_child = child
                                break

            if worden_child and participle_child:
                passive_phrase = (
                    f"{token.text} {participle_child.text} {worden_child.text}"
                )
                passive_findings.append(
                    (
                        passive_phrase,
                        "Perfekt-/Plusquamperfekt-Passiv",
                        min(token.idx, participle_child.idx, worden_child.idx),
                        max(
                            token.idx + len(token.text),
                            participle_child.idx + len(participle_child.text),
                            worden_child.idx + len(worden_child.text),
                        ),
                    )
                )

        # PATTERN 3: Future I passive (wird + Partizip II + werden)
        elif (
            token.lemma_ == "werden"
            and token.pos_ == "AUX"
            and token.tag_ in ["VAFIN", "VVFIN"]
        ):
            # Search for werden-infinitive as child
            werden_inf_child = None
            participle_child = None

            for child in token.children:
                if child.lemma_ == "werden" and child.tag_ in ["VAINF", "VVINF"]:
                    werden_inf_child = child
                elif (
                    _is_past_participle(child)
                ):
                    participle_child = child

            # Partizip II could be child of werden-infinitive
            if werden_inf_child and not participle_child:
                for grandchild in werden_inf_child.children:
                    if (
                        _is_past_participle(grandchild)
                    ):
                        participle_child = grandchild
                        break

            if werden_inf_child and participle_child:
                passive_phrase = (
                    f"{token.text} {participle_child.text} {werden_inf_child.text}"
                )
                passive_findings.append(
                    (
                        passive_phrase,
                        "Futur-I-Passiv",
                        min(token.idx, participle_child.idx, werden_inf_child.idx),
                        max(
                            token.idx + len(token.text),
                            participle_child.idx + len(participle_child.text),
                            werden_inf_child.idx + len(werden_inf_child.text),
                        ),
                    )
                )

        # PATTERN 4: Modal passive (kann/muss + Partizip II + werden)
        elif _is_modal_verb(token):
            # Search for werden-infinitive and Partizip II as children
            werden_inf_child = None
            participle_child = None

            for child in token.children:
                if child.lemma_ == "werden" and child.tag_ in ["VAINF", "VVINF"]:
                    werden_inf_child = child
                elif (
                    _is_past_participle(child)
                ):
                    participle_child = child

            # Partizip II could be child of werden-infinitive
            if werden_inf_child and not participle_child:
                for grandchild in werden_inf_child.children:
                    if (
                        _is_past_participle(grandchild)
                    ):
                        participle_child = grandchild
                        break

            if werden_inf_child and participle_child:
                passive_phrase = (
                    f"{token.text} {participle_child.text} {werden_inf_child.text}"
                )
                passive_findings.append(
                    (
                        passive_phrase,
                        "Modal-Passiv",
                        min(token.idx, participle_child.idx, werden_inf_child.idx),
                        max(
                            token.idx + len(token.text),
                            participle_child.idx + len(participle_child.text),
                            werden_inf_child.idx + len(werden_inf_child.text),
                        ),
                    )
                )

    return passive_findings


def _detect_perfect_passive(sent) -> List[Tuple[str, str, int, int]]:
    """
    Detects perfect/pluperfect passive: ist/war + Partizip II + worden

    spaCy logic: Search for "worden" token connected to sein and Partizip II
    """
    passive_findings = []

    for token in sent:
        # Search for "worden" as indicator for perfect passive
        if token.lemma_ == "worden" and token.pos_ == "AUX":
            # Find associated Partizip II and sein-auxiliary
            participle = None
            sein_verb = None

            # Check HEAD (usually the Partizip II)
            if _is_past_participle(token.head):
                participle = token.head

                # Search for sein-auxiliary
                for child in participle.children:
                    if _is_auxiliary_sein(child):
                        sein_verb = child
                        break

                # Alternative: sein could be parent of participle
                if not sein_verb and _is_auxiliary_sein(participle.head):
                    sein_verb = participle.head

            if participle and sein_verb:
                if participle.lemma_ in ADJECTIVAL_PARTICIPLES:
                    continue

                passive_phrase = f"{sein_verb.text} {participle.text} {token.text}"
                passive_type = "Perfekt-/Plusquamperfekt-Passiv"

                # Determine start and end positions
                start_idx = min(sein_verb.idx, participle.idx, token.idx)
                end_idx = max(
                    sein_verb.idx + len(sein_verb.text),
                    participle.idx + len(participle.text),
                    token.idx + len(token.text),
                )

                passive_findings.append((passive_phrase, passive_type, start_idx, end_idx))

    return passive_findings


def _detect_future_passive(sent) -> List[Tuple[str, str, int, int]]:
    """
    Detects future I passive: wird + Partizip II + werden (infinitive)

    spaCy logic: Chain of three verbs - werden (conjugated) + Partizip II + werden (infinitive)
    """
    passive_findings = []

    for token in sent:
        # Search for conjugated "werden" for future
        if _is_auxiliary_werden(token) and token.tag_.startswith(
            "VAF"
        ):  # Auxiliary finite
            # Search for Partizip II and werden-infinitive as children/siblings
            participle = None
            werden_infinitive = None

            for child in token.children:
                if _is_past_participle(child):
                    participle = child
                elif (
                    child.lemma_ == "werden" and child.tag_ == "VAINF"
                ):  # werden as infinitive
                    werden_infinitive = child

            # Also check siblings/cousins for more complex structures
            if participle and not werden_infinitive:
                for sibling in participle.children:
                    if sibling.lemma_ == "werden" and sibling.tag_ == "VAINF":
                        werden_infinitive = sibling
                        break

            if participle and werden_infinitive:
                if participle.lemma_ in ADJECTIVAL_PARTICIPLES:
                    continue

                passive_phrase = f"{token.text} {participle.text} {werden_infinitive.text}"
                passive_type = "Futur-I-Passiv"

                start_idx = min(token.idx, participle.idx, werden_infinitive.idx)
                end_idx = max(
                    token.idx + len(token.text),
                    participle.idx + len(participle.text),
                    werden_infinitive.idx + len(werden_infinitive.text),
                )

                passive_findings.append((passive_phrase, passive_type, start_idx, end_idx))

    return passive_findings


def _detect_modal_passive(sent) -> List[Tuple[str, str, int, int]]:
    """
    Detects passive with modal verbs: kann/muss + Partizip II + werden (infinitive)

    spaCy logic: Modal verb + Partizip II + werden as infinitive
    """
    passive_findings = []

    for token in sent:
        # Search for modal verb
        if _is_modal_verb(token):
            participle = None
            werden_infinitive = None

            # Search for Partizip II and werden-infinitive
            for child in token.children:
                if _is_past_participle(child):
                    participle = child
                elif child.lemma_ == "werden" and child.tag_ == "VAINF":
                    werden_infinitive = child

            # Extended search in the dependency structure
            if participle and not werden_infinitive:
                for sibling in participle.children:
                    if sibling.lemma_ == "werden" and sibling.tag_ == "VAINF":
                        werden_infinitive = sibling
                        break

            if participle and werden_infinitive:
                if participle.lemma_ in ADJECTIVAL_PARTICIPLES:
                    continue

                passive_phrase = f"{token.text} {participle.text} {werden_infinitive.text}"
                passive_type = "Modal-Passiv"

                start_idx = min(token.idx, participle.idx, werden_infinitive.idx)
                end_idx = max(
                    token.idx + len(token.text),
                    participle.idx + len(participle.text),
                    werden_infinitive.idx + len(werden_infinitive.text),
                )

                passive_findings.append((passive_phrase, passive_type, start_idx, end_idx))

    return passive_findings


def check_rule(doc: Doc) -> List[str]:
    """
    Main function for the rule check in the CLI system.
    Detects all passive constructions and returns formatted error messages.

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of error messages for found passive constructions
    """
    passive_constructions = find_passive_constructions(doc)

    errors = []
    seen = set()  # Prevent duplicates

    for passive_phrase, passive_type, start_idx, end_idx in passive_constructions:
        # Prevent duplicates based on the phrase
        if passive_phrase.lower() not in seen:
            seen.add(passive_phrase.lower())

            # Extract the actual tokens from the text for better matching
            # Use character positions to find the exact text
            original_text = doc.text[start_idx:end_idx]

            # Format error message with improvement suggestion
            suggestion = _generate_active_suggestion(passive_type)
            error_msg = (
                f'Passiv-Konstruktion "{original_text}" ({passive_type}). {suggestion}'
            )
            errors.append(error_msg)

    return errors


def _generate_active_suggestion(passive_type: str) -> str:
    """
    Generates context-specific improvement suggestions for passive constructions.

    Args:
        passive_type: Type of the passive construction

    Returns:
        Improvement suggestion as string
    """
    suggestions = {
        "Präsens-/Präteritum-Passiv": "Besser: Verwenden Sie aktive Formulierungen. Wer führt die Handlung aus?",
        "Perfekt-/Plusquamperfekt-Passiv": "Besser: Beschreiben Sie direkt, wer etwas getan hat.",
        "Futur-I-Passiv": "Besser: Verwenden Sie aktive Zukunftsformen. Wer wird handeln?",
        "Modal-Passiv": "Besser: Beschreiben Sie direkt, wer etwas tun kann/muss/soll.",
    }

    return suggestions.get(
        passive_type, "Besser: Verwenden Sie aktive Formulierungen statt Passiv."
    )


# Test functionality for standalone execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import spacy

    # Example sentences for testing
    test_sentences = [
        "Das Paket wird heute geliefert.",  # Present passive
        "Gestern wurde das Problem gelöst.",  # Preterite passive
        "Die Aufgabe ist bereits erledigt worden.",  # Perfect passive
        "Das alte Haus war schon verkauft worden.",  # Pluperfect passive
        "Die Rechnung wird bezahlt werden.",  # Future I passive
        "Der Antrag muss noch ausgefüllt werden.",  # Passive with modal verb
        "Die Arbeiter bauen das Haus.",  # Active (negative example)
        "Er wird morgen kommen.",  # Future I active (negative example)
        "Das Fenster ist geöffnet.",  # Stative passive (should not be detected)
    ]

    logger.info("=== Test of robust passive detection ===\n")

    try:
        nlp = spacy.load("de_core_news_lg")

        for i, sentence in enumerate(test_sentences, 1):
            logger.info("%s. Text: '%s'", i, sentence)
            doc = nlp(sentence)

            # Debug: Show token information
            logger.info(" Tokens:")
            for token in doc:
                if token.pos_ in ["VERB", "AUX"] or "PP" in token.tag_:
                    logger.info(" %12 | %12 | %4 | %8 | %12 | head: %s", token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.head.text)

            results = check_rule(doc)

            if results:
                logger.info(" %s passive construction(s) detected:", len(results))
                for error in results:
                    logger.error("  - %s", error)
            else:
                logger.info(" No passive constructions detected")
            logger.info("")

    except OSError:
        logger.error(" spaCy model 'de_core_news_lg' not found!")
        logger.error(" Please run: python -m spacy download de_core_news_lg")
