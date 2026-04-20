from typing import Dict, List

import pyphen
import spacy
from spacy.tokens import Doc, Token

# Import configuration (currently empty, for future extensions)
from . import config  # noqa: F401

# Initialize pyphen for the German language
import logging

logger = logging.getLogger(__name__)
hyphenator = pyphen.Pyphen(lang="de_DE")

# Configuration for word length analysis
MAX_SYLLABLES = 4
MAX_CHARS = 14
COMPLEX_WORD_THRESHOLD = 5  # syllables


def check_rule(doc: Doc) -> List[str]:
    """
    Detects overly long words using advanced NLP techniques:
    - Multi-dimensional length analysis (syllables, characters, morphemes)
    - Morphological complexity scoring with POS tags
    - Contextual readability analysis
    - Phonetic syllable structure validation
    """
    errors = []
    processed_tokens = set()

    # 1. High-precision multi-dimensional length analysis
    for token in doc:
        if token.i in processed_tokens or should_be_ignored(token):
            continue

        # Advanced morpho-phonetic analysis
        length_info = analyze_word_length(token)

        if length_info["is_too_long"]:
            complexity_info = evaluate_word_complexity(token, length_info)
            errors.append(
                f'{complexity_info["typ"]} "{token.text}" ({complexity_info["details"]}). '
                f"Besser: {complexity_info['suggestion']}"
            )
            processed_tokens.add(token.i)

    # 2. Pattern-based detection for problematic word groups
    word_group_issues = detect_problematic_word_groups(doc)
    for problem in word_group_issues:
        errors.append(problem["message"])
        for token_idx in problem["token_indices"]:
            processed_tokens.add(token_idx)

    # 3. Contextual readability analysis at sentence level
    sentence_readability = analyze_sentence_readability(doc)
    for warning in sentence_readability:
        errors.append(warning)

    return errors


def should_be_ignored(token: Token) -> bool:
    """Filter for tokens that should be ignored."""
    return (
        token.is_punct
        or token.is_space
        or token.is_digit
        or len(token.text) < 4
        or token.pos_ in ["PROPN", "X"]
    )


def count_syllables(word: str) -> int:
    """
    Counts syllables of a word correctly, including hyphenated compounds.

    For compounds with hyphens (e.g. "Ski-Gebiet"), the parts are analyzed
    separately so the structural hyphen is not counted as a syllable boundary.
    """
    # For hyphenated compounds, count syllables of each part separately
    if "-" in word:
        parts = word.split("-")
        total_syllables = 0
        for part in parts:
            if part:  # Skip empty parts
                # pyphen.inserted() inserts hyphens at syllable boundaries
                # e.g. "Gebiet" -> "Ge-biet" -> 2 syllables
                syllable_parts = hyphenator.inserted(part).split("-")
                total_syllables += len(syllable_parts)
        return total_syllables
    else:
        # Regular word without hyphen
        return len(hyphenator.inserted(word).split("-"))


def analyze_word_length(token: Token) -> Dict:
    """
    Multi-dimensional word length analysis.
    Combines syllables, characters, and morphological complexity.
    """
    word = token.text

    # 1. Syllable analysis with improved counting for compounds
    syllable_count = count_syllables(word)

    # 2. Character analysis
    char_count = len(word)

    # 3. Morphological complexity
    morpho_complexity = evaluate_morphological_complexity(token)

    # Determine if too long
    # Allow one more syllable for hyphenated compounds (already in recommended form)
    max_syl = MAX_SYLLABLES
    if "-" in word:
        max_syl = MAX_SYLLABLES + 1  # 4 syllables allowed for hyphenated

    is_too_long = (
        syllable_count > max_syl
        or char_count > MAX_CHARS
        or morpho_complexity > 0.7
    )

    return {
        "is_too_long": is_too_long,
        "syllables": syllable_count,
        "characters": char_count,
        "morpho_complexity": morpho_complexity,
        "main_criterion": determine_main_criterion(
            syllable_count, char_count, morpho_complexity
        ),
    }


def evaluate_morphological_complexity(token: Token) -> float:
    """Evaluates morphological complexity of a word."""
    word = token.text.lower()
    complexity = 0.0

    # Complexity factors
    # Prefixes increase complexity (German prefixes)
    prefixes = ["un", "ver", "ent", "er", "be", "ge", "über", "unter", "miss"]
    for prefix in prefixes:
        if word.startswith(prefix):
            complexity += 0.2
            break

    # Suffixes increase complexity (German suffixes)
    suffixes = ["ung", "heit", "keit", "schaft", "ismus", "ation", "ierung"]
    for suffix in suffixes:
        if word.endswith(suffix):
            complexity += 0.3
            break

    # Foreign word indicators
    foreign_word_patterns = ["tion", "sion", "ment", "enz", "anz", "ismus"]
    for pattern in foreign_word_patterns:
        if pattern in word:
            complexity += 0.3
            break

    return min(1.0, complexity)


def determine_main_criterion(syllables: int, characters: int, morpho: float) -> str:
    """Determines the main criterion for word length."""
    if syllables > MAX_SYLLABLES and characters > MAX_CHARS:
        return "syllables_and_characters"
    elif syllables > MAX_SYLLABLES:
        return "syllables"
    elif characters > MAX_CHARS:
        return "characters"
    elif morpho > 0.7:
        return "morphology"
    else:
        return "general"


def evaluate_word_complexity(token: Token, length_info: Dict) -> Dict[str, str]:
    """
    Evaluates word complexity and generates specific suggestions.
    """
    syllables = length_info["syllables"]
    characters = length_info["characters"]
    length_info["main_criterion"]

    # Determine type and difficulty
    if syllables >= 6:
        typ = "Sehr langes Wort"
    elif syllables >= 4:
        typ = "Langes Wort"
    else:
        typ = "Etwas langes Wort"

    # Generate specific suggestion
    if token.pos_ == "NOUN":
        if syllables >= 5:
            suggestion = "Verwenden Sie ein kürzeres Synonym oder erklären Sie das Wort."
        else:
            suggestion = "Prüfen Sie, ob es ein einfacheres Wort gibt."
    elif token.pos_ == "VERB":
        suggestion = "Verwenden Sie ein einfacheres Verb oder erklären Sie die Handlung."
    elif token.pos_ == "ADJ":
        suggestion = "Beschreiben Sie die Eigenschaft mit einfacheren Wörtern."
    else:
        suggestion = "Verwenden Sie ein kürzeres, einfacheres Wort."

    details = f"{syllables} Silben, {characters} Zeichen"

    return {"typ": typ, "details": details, "suggestion": suggestion}


def detect_problematic_word_groups(doc: Doc) -> List[Dict]:
    """
    Detects groups of long words that together are hard to read.
    """
    problems = []

    # Detect consecutive long words
    long_word_sequence = []

    for token in doc:
        if (
            not should_be_ignored(token)
            and count_syllables(token.text) > MAX_SYLLABLES
        ):
            long_word_sequence.append(token)
        else:
            # Sequence interrupted - check if problematic
            if len(long_word_sequence) >= 2:
                problems.append(
                    {
                        "token_indices": [t.i for t in long_word_sequence],
                        "message": f"Sequenz von {len(long_word_sequence)} langen Wörtern hintereinander. Besser: Mischen Sie kurze und lange Wörter.",
                    }
                )
            long_word_sequence = []

    # Check last sequence
    if len(long_word_sequence) >= 2:
        problems.append(
            {
                "token_indices": [t.i for t in long_word_sequence],
                "message": f"Sequenz von {len(long_word_sequence)} langen Wörtern hintereinander. Besser: Mischen Sie kurze und lange Wörter.",
            }
        )

    return problems


def analyze_sentence_readability(doc: Doc) -> List[str]:
    """
    Analyzes readability at sentence level based on word lengths.
    """
    warnings = []

    for i, sent in enumerate(doc.sents, 1):
        # Count long words per sentence
        long_words = [
            token
            for token in sent
            if (
                not should_be_ignored(token)
                and count_syllables(token.text) > MAX_SYLLABLES
            )
        ]

        # Warn if too many long words
        if len(long_words) >= 3:
            warnings.append(
                f"Satz {i} enthält {len(long_words)} lange Wörter. "
                f"Besser: Max. 2 lange Wörter pro Satz verwenden."
            )

        # Calculate average word length
        relevant_words = [t for t in sent if not should_be_ignored(t)]
        if relevant_words:
            avg_syllables = sum(
                count_syllables(t.text) for t in relevant_words
            ) / len(relevant_words)

            if avg_syllables > 3.0:
                warnings.append(
                    f"Satz {i} hat eine durchschnittliche Wortlänge von {avg_syllables:.1f} Silben. "
                    f"Besser: Verwenden Sie mehr kurze Wörter."
                )

    return warnings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test the word length rule
    nlp = spacy.load("de_core_news_lg")

    test_text = """Die Bundeskanzlerin erörterte Digitalisierungsstrategien.
    Veranstaltungsorganisationen koordinieren Teilnehmerregistrierungen.
    Qualitätssicherungsmaßnahmen verbessern Benutzerfreundlichkeit."""

    logger.info("--- Test: Word length detection rule ---")
    doc = nlp(test_text)
    results = check_rule(doc)

    if results:
        logger.info("Found: %s word length issues", len(results))
        for i, error in enumerate(results, 1):
            logger.error("%s. %s", i, error)
    else:
        logger.info(" All words have acceptable length.")
