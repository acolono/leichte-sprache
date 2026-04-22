"""Personal pronoun rule for Leichte Sprache.
Uses inlined PyTorch pronoun classifier with BaseMLRule for thread-safe loading.

This rule detects specific pronoun problems for Leichte Sprache:
1. Pronoun accumulation (>3 pronouns per sentence)
2. Unclear pronoun references ("es", "das" without clear reference)
3. Reflexive pronouns (impede comprehension)
4. Inconsistent address (Du/Sie mixing)
5. Ambiguous gender assignment
6. Impersonal pronoun "man"
"""

from pathlib import Path
from typing import List, Optional

from spacy.tokens import Doc, Token

import logging

from regeln.base_ml_rule import BaseMLRule
from .analyzer import PronounAnalyzer

logger = logging.getLogger(__name__)


class PronounModel(BaseMLRule):
    """Thread-safe lazy loader for the pronoun classifier."""

    @classmethod
    def _load_model(cls):
        model_dir = Path(__file__).parent / "model"
        model_path = model_dir / "best_model.pt"
        label_encoders_path = model_dir / "label_encoders.json"
        return PronounAnalyzer(
            model_path=model_path,
            label_encoders_path=label_encoders_path,
        )


def get_german_article(gender: str, case: str = "Nom", number: str = "Sing") -> str:
    """
    Generate grammatically correct German article based on morphological features.

    Args:
        gender: Gender of the noun (Masc, Fem, Neut)
        case: Grammatical case (Nom, Gen, Dat, Acc)
        number: Number (Sing, Plur)

    Returns:
        Correct German article
    """
    article_table = {
        # Masculine singular
        ("Masc", "Nom", "Sing"): "der",
        ("Masc", "Gen", "Sing"): "des",
        ("Masc", "Dat", "Sing"): "dem",
        ("Masc", "Acc", "Sing"): "den",
        # Feminine singular
        ("Fem", "Nom", "Sing"): "die",
        ("Fem", "Gen", "Sing"): "der",
        ("Fem", "Dat", "Sing"): "der",
        ("Fem", "Acc", "Sing"): "die",
        # Neuter singular
        ("Neut", "Nom", "Sing"): "das",
        ("Neut", "Gen", "Sing"): "des",
        ("Neut", "Dat", "Sing"): "dem",
        ("Neut", "Acc", "Sing"): "das",
        # Plural (same for all genders)
        (gender, "Nom", "Plur"): "die",
        (gender, "Gen", "Plur"): "der",
        (gender, "Dat", "Plur"): "den",
        (gender, "Acc", "Plur"): "die",
    }

    # Try to get specific article, fall back to nominative if not found
    article = article_table.get((gender, case, number))
    if not article:
        # Fallback to nominative
        article = article_table.get((gender, "Nom", number))
    if not article:
        # Ultimate fallback based on gender
        fallback = {"Masc": "der", "Fem": "die", "Neut": "das"}
        article = fallback.get(gender, "das")

    return article


def find_spacy_token_for_word(
    doc: Doc, word: str, search_before_position: Optional[int] = None
) -> Optional[Token]:
    """
    Find the spaCy token corresponding to a word.

    Args:
        doc: spaCy Doc object
        word: Word to find
        search_before_position: Optional character position to search before

    Returns:
        spaCy Token if found, None otherwise
    """
    word_lower = word.lower()

    # If we have a position, prioritize tokens before it
    if search_before_position is not None:
        # Search backwards from the position
        for token in reversed(doc):
            if token.idx < search_before_position and token.text.lower() == word_lower:
                # Prefer nouns and proper nouns
                if token.pos_ in ["NOUN", "PROPN"]:
                    return token
        # If no noun found, try any matching token
        for token in reversed(doc):
            if token.idx < search_before_position and token.text.lower() == word_lower:
                return token

    # Fallback: search entire document
    for token in doc:
        if token.text.lower() == word_lower:
            if token.pos_ in ["NOUN", "PROPN"]:
                return token

    # Final fallback: any matching token
    for token in doc:
        if token.text.lower() == word_lower:
            return token

    return None


def get_article_for_referent(
    doc: Doc, referent_word: str, pronoun_token: Optional[Token] = None
) -> str:
    """
    Get the appropriate article for a referent noun using spaCy morphology.

    Args:
        doc: spaCy Doc object
        referent_word: The referent noun
        pronoun_token: Optional pronoun token to get case information from

    Returns:
        Grammatically correct article with the noun
    """
    # Find the spaCy token for the referent
    referent_token = find_spacy_token_for_word(doc, referent_word)

    if not referent_token:
        # Fallback if we can't find the token
        return f"das {referent_word.capitalize()}"

    # Get morphological features
    gender_list = referent_token.morph.get("Gender")
    number_list = referent_token.morph.get("Number")

    gender = gender_list[0] if gender_list else "Neut"
    number = number_list[0] if number_list else "Sing"

    # Get case from pronoun context if available
    case = "Nom"  # Default to nominative
    if pronoun_token:
        case_list = pronoun_token.morph.get("Case")
        if case_list:
            case = case_list[0]

    # Generate the article
    article = get_german_article(gender, case, number)

    # Return article with properly capitalized noun
    noun = referent_word.capitalize() if referent_word[0].islower() else referent_word
    return f"{article} {noun}"


def check_rule(doc: Doc) -> List[str]:
    """
    Main function for the personal pronoun rule in the CLI system.
    Detects various pronoun problems for Leichte Sprache.

    Args:
        doc: spaCy Doc object with processed text

    Returns:
        List of error messages for found pronoun problems
    """
    errors = []
    seen = set()

    # Always runs (no model needed)
    for msg in _detect_man_pronoun(doc):
        if msg not in seen:
            errors.append(msg)
            seen.add(msg)

    # DIN SPEC 33429 §5.4.9 / Hildesheim Regelbuch: avoid 3rd-person
    # personal pronouns (er/sie/es/ihn/ihm/ihr/ihnen/…). Runs without
    # the ML model so the rule degrades gracefully.
    for msg in _detect_third_person_pronoun_use(doc):
        if msg not in seen:
            errors.append(msg)
            seen.add(msg)

    # ML-dependent checks via BaseMLRule singleton
    analyzer = PronounModel.get_model()
    if analyzer is None:
        return errors  # Graceful degradation: man-only + DIN SPEC

    try:
        results = analyzer.analyze(doc.text)

        # Problem 1: Pronoun accumulation
        for msg in _detect_pronoun_accumulation(results, doc):
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)

        # Problem 2: Unclear pronoun references
        for msg in _detect_unclear_references_enhanced(results, doc):
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)

        # Problem 3: Reflexive pronouns (extracted from results, no separate call)
        for msg in _detect_reflexive_pronouns(results, doc):
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)

        # Problem 4: Address inconsistency (extracted from results, no separate call)
        for msg in _detect_address_inconsistency(results, doc):
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)

        # Problem 5: Ambiguous pronouns
        ambiguous_errors = _detect_ambiguous_pronouns(results, doc)
        for msg in ambiguous_errors[:3]:
            if msg not in seen:
                errors.append(msg)
                seen.add(msg)

    except Exception as e:
        errors.append(f"Fehler bei Personalpronomen-Analyse: {str(e)}")

    return errors


def _detect_pronoun_accumulation(results: dict, doc: Doc) -> List[str]:
    """Detects sentences with too many pronouns (>3 per sentence)."""
    errors = []

    for sent_data in results["sentences"]:
        pronoun_count = len(sent_data["pronouns"])
        if pronoun_count > 3:
            sentence = sent_data["sentence"]
            pronoun_list = [p["token"] for p in sent_data["pronouns"]]

            # Remove duplicates but keep order
            seen = set()
            unique_pronouns = []
            for p in pronoun_list:
                p_clean = p.strip(".,!?;:")
                if p_clean not in seen:
                    seen.add(p_clean)
                    unique_pronouns.append(p_clean)

            errors.append(
                f'Pronomen-Häufung: "{sentence[:60]}..." '
                f"({pronoun_count} Pronomen: {', '.join(unique_pronouns)}). "
                f"Besser: Teilen Sie den Satz auf oder ersetzen Sie Pronomen durch Hauptwörter."
            )

    return errors


def _detect_unclear_references_enhanced(results: dict, doc: Doc) -> List[str]:
    """
    Detects unclear pronoun references with specific suggestions.
    For Leichte Sprache, problematic pronouns are always reported,
    especially when they appear at the beginning of a sentence.
    """
    errors = []
    unclear_pronouns = ["es", "das", "dies", "diese", "jene", "solche"]

    for sent_data in results["sentences"]:
        sentence = sent_data["sentence"]
        sent_start_char = sent_data.get("start_char", 0)

        for pronoun in sent_data["pronouns"]:
            token = pronoun["token"].lower().strip(".,!?;:")

            if token in unclear_pronouns:
                # For Leichte Sprache, always flag these pronouns, especially at sentence start
                is_sentence_start = pronoun["position"] == 0

                # Get confidence for reference checking
                gender_feature = pronoun["features"].get("gender", {})
                gender_conf = gender_feature.get("confidence", 1.0)

                # For Leichte Sprache: Always flag "es" at sentence start or with any uncertainty
                # Other unclear pronouns: flag if confidence is not very high
                should_flag = False
                if token == "es":
                    # "Es" is especially problematic in Leichte Sprache
                    should_flag = is_sentence_start or gender_conf < 0.95
                else:
                    # Other unclear pronouns: flag if at sentence start or lower confidence
                    should_flag = is_sentence_start or gender_conf < 0.85

                if should_flag:
                    # Try to find what the pronoun refers to from candidates
                    candidates = pronoun.get("candidates", [])

                    if candidates and len(candidates) > 0:
                        # For sentence-initial pronouns, prefer candidates from previous sentence
                        referent = None
                        if is_sentence_start and len(candidates) > 1:
                            # For sentence-initial pronouns, the second candidate is often
                            # from the previous sentence and more likely the correct referent
                            # Filter out determiners and pronouns
                            noun_candidates = []
                            for cand in candidates:
                                cand_word = cand.get("word", "").lower()
                                if cand_word not in [
                                    "das",
                                    "die",
                                    "der",
                                    "es",
                                    "sie",
                                    "er",
                                    "den",
                                    "dem",
                                    "des",
                                ]:
                                    noun_candidates.append(cand)

                            # Prefer the second noun candidate (often from previous sentence)
                            # for sentence-initial pronouns
                            if len(noun_candidates) > 1:
                                referent = noun_candidates[
                                    1
                                ]  # Second noun is often from previous sentence
                            elif noun_candidates:
                                referent = noun_candidates[0]

                        # Fallback to first candidate if no better option found
                        if not referent:
                            referent = candidates[0]

                        referent_word = referent.get("word", "")

                        if referent_word:
                            # Find the pronoun token in spaCy doc for case information
                            pronoun_char_pos = sent_start_char + pronoun["position"]
                            pronoun_spacy_token = None
                            for spacy_token in doc:
                                if spacy_token.idx == pronoun_char_pos:
                                    pronoun_spacy_token = spacy_token
                                    break

                            # Get grammatically correct article and noun
                            suggestion = get_article_for_referent(
                                doc, referent_word, pronoun_spacy_token
                            )

                            # Create specific error message
                            if is_sentence_start:
                                errors.append(
                                    f'Satz beginnt mit unklarem Pronomen "{pronoun["token"]}". '
                                    f'"{pronoun["token"]}" bezieht sich auf "{referent_word}". '
                                    f'Besser: Verwenden Sie "{suggestion}" statt "{pronoun["token"]}".'
                                )
                            else:
                                errors.append(
                                    f'Unklarer Pronomen-Bezug: "{pronoun["token"]}" in "{sentence[:50]}...". '
                                    f'"{pronoun["token"]}" bezieht sich auf "{referent_word}". '
                                    f'Besser: Ersetzen Sie "{pronoun["token"]}" durch "{suggestion}".'
                                )
                        else:
                            # Fallback to generic message if no referent found
                            errors.append(
                                f'Unklarer Pronomen-Bezug: "{pronoun["token"]}" in "{sentence[:50]}...". '
                                f'Besser: Ersetzen Sie "{pronoun["token"]}" durch das passende Hauptwort.'
                            )
                    else:
                        # No candidates available - use generic message
                        if is_sentence_start:
                            errors.append(
                                f'Satz beginnt mit unklarem Pronomen "{pronoun["token"]}". '
                                f'Besser: Verwenden Sie ein konkretes Hauptwort statt "{pronoun["token"]}".'
                            )
                        else:
                            errors.append(
                                f'Unklarer Pronomen-Bezug: "{pronoun["token"]}" in "{sentence[:50]}...". '
                                f'Besser: Ersetzen Sie "{pronoun["token"]}" durch das passende Hauptwort.'
                            )

    return errors


def _detect_reflexive_pronouns(results: dict, doc: Doc) -> List[str]:
    """Detects problematic reflexive pronouns from analysis results."""
    errors = []
    for sent_data in results["sentences"]:
        for pronoun in sent_data["pronouns"]:
            reflex_feature = pronoun["features"].get("reflex", {})
            # Predictor stores str(pred_label), so value is "True" or "False"
            if str(reflex_feature.get("value", "")).lower() == "true":
                confidence = reflex_feature.get("confidence", 0.0)
                if confidence > 0.8:
                    sentence = sent_data["sentence"]
                    errors.append(
                        f'Reflexivpronomen: "{pronoun["token"]}" in "{sentence[:50]}...". '
                        f"Besser: Verwenden Sie eine einfachere Formulierung ohne Reflexivpronomen."
                    )
    return errors


def _detect_address_inconsistency(results: dict, doc: Doc) -> List[str]:
    """Detects mixing of Du/Sie address forms."""
    errors = []
    # Extract polite forms from results
    polite_pronouns = []
    for sent_data in results["sentences"]:
        for pronoun in sent_data["pronouns"]:
            if pronoun["features"].get("polite", {}).get("value") == "Form":
                polite_pronouns.append({
                    "token": pronoun["token"],
                    "confidence": pronoun["features"]["polite"]["confidence"],
                    "sentence": sent_data["sentence"],
                })

    # Search for informal pronouns (2nd person)
    informal_pronouns = []
    informal_tokens = ["du", "dich", "dir", "dein", "deine"]
    for sent_data in results["sentences"]:
        for pronoun in sent_data["pronouns"]:
            token = pronoun["token"].lower().strip(".,!?;:")
            person_feature = pronoun["features"].get("person", {}).get("value")
            if person_feature == "2" and token in informal_tokens:
                informal_pronouns.append(pronoun)

    # If both polite and informal forms found
    if polite_pronouns and informal_pronouns:
        polite_tokens_list = list(
            set([p["token"].strip(".,!?;:") for p in polite_pronouns])
        )
        informal_tokens_found = list(
            set([p["token"].strip(".,!?;:") for p in informal_pronouns])
        )
        errors.append(
            f"Gemischte Anredeformen: Höflich ({', '.join(polite_tokens_list[:3])}) und "
            f"Informell ({', '.join(informal_tokens_found[:3])}). "
            f'Besser: Verwenden Sie einheitlich entweder "Du" oder "Sie".'
        )
    return errors


def _detect_ambiguous_pronouns(results: dict, doc: Doc) -> List[str]:
    """Detects pronouns with low confidence for gender/number."""
    errors = []

    for sent_data in results["sentences"]:
        for pronoun in sent_data["pronouns"]:
            token = pronoun["token"]
            features = pronoun["features"]

            # Check confidence for critical features
            gender_conf = features.get("gender", {}).get("confidence", 1.0)
            number_conf = features.get("number", {}).get("confidence", 1.0)

            # Low confidence = ambiguity
            if gender_conf < 0.6 or number_conf < 0.6:
                sentence = sent_data["sentence"]
                conf_info = f"Geschlecht: {gender_conf:.2f}, Anzahl: {number_conf:.2f}"

                errors.append(
                    f'Mehrdeutiges Pronomen: "{token}" in "{sentence[:50]}...". '
                    f"Die grammatischen Eigenschaften sind unklar ({conf_info}). "
                    f"Besser: Verwenden Sie das entsprechende Hauptwort statt des Pronomens."
                )

    return errors


def _detect_man_pronoun(doc: Doc) -> List[str]:
    """Detects the impersonal pronoun 'man' (should be avoided in Leichte Sprache)."""
    errors = []

    for token in doc:
        if token.text.lower() == "man" and token.pos_ == "PRON":
            sent = token.sent
            sent_text = sent.text[:60] + "..." if len(sent.text) > 60 else sent.text

            errors.append(
                f'Unpersönliches Pronomen "man" in: "{sent_text}". '
                f'"man" ist unklar und abstrakt. '
                f'Besser: "die Menschen" oder "die Leute".'
            )

    return errors


_THIRD_PERSON_FORMS = {
    "er", "ihn", "ihm", "seiner",
    "sie", "ihr", "ihnen", "ihrer",
    "es",
}
_POLITE_CAPITALIZED = {
    "Sie", "Ihnen", "Ihr", "Ihre", "Ihrer", "Ihrem", "Ihren", "Ihres",
}
_EXPLETIVE_VERBS = {
    "regnen", "schneien", "hageln", "donnern", "blitzen",
    "dämmern", "tagen", "nachten", "frieren",
    "geben",  # "es gibt"
}


def _is_polite_form(token: Token) -> bool:
    """Return True if the pronoun is formal 'Sie/Ihnen/Ihr…' (Höflichkeitsform).

    In German the polite 2nd-person address is morphologically identical to
    the 3rd-person-plural pronoun. We disambiguate via two rules:
      1. Non-sentence-start capitalized 'Sie/Ihnen/Ihr…' is always polite.
      2. Sentence-start capitalized 'Sie' + plural-number verb is polite;
         'Sie' + singular-number verb is referential 'sie' (she).
    """
    surface = token.text
    if surface not in _POLITE_CAPITALIZED:
        return False

    # Any "Ihnen/Ihr/Ihre/Ihrer/…" (capital I) mid-text is unambiguously polite.
    if surface != "Sie":
        return True

    # For "Sie" — verb agreement disambiguates.
    verb = token.head
    if verb is not None and verb.pos_ in ("VERB", "AUX"):
        verb_number = verb.morph.get("Number")
        if verb_number == ["Sing"]:
            # 3rd-sing verb -> referential "Sie" (she), e.g. "Sie geht"
            return False
    # Default: capitalized "Sie" is polite.
    return True


def _is_expletive_es(token: Token) -> bool:
    """Return True if 'es' is the expletive subject of a weather/existential verb."""
    if token.text.lower() != "es":
        return False
    if token.dep_ == "expl":
        return True
    verb = token.head
    if verb is not None and verb.pos_ in ("VERB", "AUX"):
        if verb.lemma_.lower() in _EXPLETIVE_VERBS:
            return True
    return False


def _detect_third_person_pronoun_use(doc: Doc) -> List[str]:
    """Flag every 3rd-person personal pronoun per DIN SPEC 33429 §5.4.9.

    Carve-outs (Hildesheim Regelbuch, Bredel & Maaß, pp. 143 ff.):
      - polite formal address 'Sie/Ihnen/Ihr…' (Höflichkeitsform)
      - expletive 'es' without antecedent ('Es regnet', 'Es gibt')
      - impersonal 'man' (handled by _detect_man_pronoun)
    """
    errors = []
    reported_positions = set()

    for token in doc:
        if token.pos_ != "PRON":
            continue
        lower = token.text.lower()
        if lower not in _THIRD_PERSON_FORMS:
            continue
        # spaCy Person feature — skip 1st/2nd person pronouns that happen to
        # share a surface form (e.g. "ihr" as 2nd-plur-informal).
        person = token.morph.get("Person")
        if person and "3" not in person:
            continue
        if _is_polite_form(token):
            continue
        if lower == "es" and _is_expletive_es(token):
            continue
        if token.i in reported_positions:
            continue
        reported_positions.add(token.i)

        sent = token.sent
        sent_text = sent.text[:60] + "..." if len(sent.text) > 60 else sent.text
        errors.append(
            f'Personalpronomen "{token.text}" in: "{sent_text}". '
            "Leichte Sprache: Personalpronomen der 3. Person sollten vermieden werden. "
            "Wiederholen Sie stattdessen das Haupt-Wort."
        )

    return errors
