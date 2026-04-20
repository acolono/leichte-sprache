"""
Modernized rule for detecting compound words (Komposita) in Leichte Sprache.
Uses german_compound_splitter for fast and precise compound decomposition.
"""

import logging
import os
from typing import Dict, List, Optional

import spacy
from spacy.tokens import Doc, Token
from wordfreq import word_frequency

logger = logging.getLogger(__name__)

# Import configuration from separate file
from .config import (  # noqa: E402
    KNOWN_COMPOUNDS,
    OWN_DICTIONARY,
    LINKING_SOUNDS,
    MIN_COMPOUND_LENGTH,
    MIN_PART_COUNT,
    MIN_WORD_LENGTH,
    MODE,
    MODE_ALL_COMPOUNDS,
    MODE_COMBINED,
    MODE_BY_LENGTH,
    MODE_BY_PARTS,
    LANGUAGE,
    WORD_FREQ_THRESHOLD,
)


class CompoundConfig:
    """
    Configuration class for compound word detection.
    Allows flexible adjustment of detection criteria.
    """

    # Detection modes (constants)
    ALL_COMPOUNDS = MODE_ALL_COMPOUNDS
    BY_LENGTH = MODE_BY_LENGTH
    BY_PARTS = MODE_BY_PARTS
    COMBINED = MODE_COMBINED

    # Main configuration
    MODE = MODE
    MIN_WORD_LENGTH = MIN_WORD_LENGTH
    MIN_PART_COUNT = MIN_PART_COUNT
    MIN_COMPOUND_LENGTH = MIN_COMPOUND_LENGTH

    @classmethod
    def config_strict(cls):
        """Strict configuration: All compounds are detected."""
        cls.MODE = cls.ALL_COMPOUNDS
        cls.MIN_PART_COUNT = 2
        cls.MIN_WORD_LENGTH = 8
        logger.info(" STRICT compound detection activated: All compounds (2+ parts)")

    @classmethod
    def config_moderate(cls):
        """Moderate configuration: Only complex compounds (default)."""
        cls.MODE = cls.BY_PARTS
        cls.MIN_PART_COUNT = 3
        cls.MIN_WORD_LENGTH = 12
        logger.info(" MODERATE compound detection activated: Only 3+ parts")

    @classmethod
    def config_length(cls):
        """Length-based configuration: Only very long words."""
        cls.MODE = cls.BY_LENGTH
        cls.MIN_PART_COUNT = 3
        cls.MIN_WORD_LENGTH = 15
        logger.info(" LENGTH-BASED compound detection activated: Only 15+ characters")

    @classmethod
    def config_liberal(cls):
        """Liberal configuration: Only very complex compounds."""
        cls.MODE = cls.BY_PARTS
        cls.MIN_PART_COUNT = 4
        cls.MIN_WORD_LENGTH = 20
        logger.info(" LIBERAL compound detection activated: Only 4+ parts")

# German Compound Splitter import for professional compound decomposition
try:
    from german_compound_splitter.comp_split import dissect, read_dictionary_from_file

    COMPOUND_SPLITTER_AVAILABLE = True
except ImportError:
    logger.info(
        "german_compound_splitter nicht verfügbar. Fallback auf einfache Implementierung."
    )
    COMPOUND_SPLITTER_AVAILABLE = False

# GermaLemma import for word validation
try:
    from germalemma import GermaLemma

    GERMALEMMA_AVAILABLE = True
    _germalemma_instance = None
except ImportError:
    logger.info("GermaLemma nicht verfügbar. Fallback auf einfache Wort-Validierung.")
    GERMALEMMA_AVAILABLE = False
    _germalemma_instance = None


def _get_germalemma():
    """Returns a GermaLemma instance (singleton)."""
    global _germalemma_instance
    if _germalemma_instance is None and GERMALEMMA_AVAILABLE:
        _germalemma_instance = GermaLemma()
    return _germalemma_instance


# Globally cached Aho-Corasick data structure
_compound_splitter_ahocs = None
_compound_splitter_loaded = False

# Performance cache for already decomposed words
_decomposition_cache = {}
_compound_cache = {}

# --- COMPOUND-SPLITTER-BASED CORE LOGIC ---


def _get_compound_splitter():
    """
    Loads the German Compound Splitter once and caches the Aho-Corasick data structure.
    """
    global _compound_splitter_ahocs, _compound_splitter_loaded

    if _compound_splitter_loaded:
        return _compound_splitter_ahocs

    if not COMPOUND_SPLITTER_AVAILABLE:
        _compound_splitter_loaded = True
        return None

    try:
        # Standalone-capable: Dictionary in the same directory as regel.py
        test_dict_path = os.path.join(os.path.dirname(__file__), "test_german_dict.txt")
        if os.path.exists(test_dict_path):
            _compound_splitter_ahocs = read_dictionary_from_file(test_dict_path)
            logger.info("German Compound Splitter mit Test-Dictionary geladen")
        else:
            logger.warning(
                "Kein Dictionary für german_compound_splitter gefunden. Fallback aktiv."
            )
            _compound_splitter_ahocs = None
    except Exception as e:
        logger.warning("Fehler beim Laden des Compound Splitter: %s", e)
        _compound_splitter_ahocs = None

    _compound_splitter_loaded = True
    return _compound_splitter_ahocs


def is_compound_by_config(word: str) -> bool:
    """
    Configurable compound detection based on CompoundConfig.

    Args:
        word: The German word to check

    Returns:
        bool: True if the word is considered problematic per configuration
    """
    if not word or len(word) < CompoundConfig.MIN_COMPOUND_LENGTH:
        return False

    # Performance cache: check if already computed
    cache_key = f"{word.lower()}_{CompoundConfig.MODE}_{CompoundConfig.MIN_WORD_LENGTH}_{CompoundConfig.MIN_PART_COUNT}"
    if cache_key in _compound_cache:
        return _compound_cache[cache_key]

    try:
        parts = _split_with_compound_splitter(word)

        if CompoundConfig.MODE == CompoundConfig.ALL_COMPOUNDS:
            # All compounds with 2+ parts
            result = len(parts) >= 2

        elif CompoundConfig.MODE == CompoundConfig.BY_LENGTH:
            # Compounds above certain word length (regardless of parts)
            result = len(word) >= CompoundConfig.MIN_WORD_LENGTH

        elif CompoundConfig.MODE == CompoundConfig.BY_PARTS:
            # Compounds with certain number of parts
            result = len(parts) >= CompoundConfig.MIN_PART_COUNT

        elif CompoundConfig.MODE == CompoundConfig.COMBINED:
            # Combined mode: length OR part count
            length_met = len(word) >= CompoundConfig.MIN_WORD_LENGTH
            parts_met = len(parts) >= CompoundConfig.MIN_PART_COUNT
            result = length_met or parts_met

        else:
            # Fallback: like before (3+ parts)
            result = len(parts) >= 3

        # Cache the result
        _compound_cache[cache_key] = result
        return result

    except Exception:
        # On errors: conservatively return False and cache
        _compound_cache[cache_key] = False
        return False


def is_complex_compound(word: str) -> bool:
    """
    Legacy function: Checks if a word is a complex compound (3+ components).
    Uses the new configurable function with default settings.

    Args:
        word: The German word to check

    Returns:
        bool: True if the word consists of 3 or more components
    """
    return is_compound_by_config(word)


def _reconstruct_with_linking_sounds(original_word: str, parts: List[str]) -> List[str]:
    """
    Reconstructs compound parts with the original linking sounds.
    Appends linking sounds to the respective left word.

    Args:
        original_word: The original compound word
        parts: The decomposed parts without linking sounds

    Returns:
        List[str]: Parts with reconstructed linking sounds
    """
    if not parts or len(parts) <= 1:
        return parts

    # Convert everything to lowercase for comparison
    original_lower = original_word.lower()
    parts_lower = [part.lower() for part in parts]

    reconstructed_parts = []
    position = 0

    for i, part in enumerate(parts_lower):
        # Find the part in the original from the current position
        part_start = original_lower.find(part, position)

        if part_start == -1:
            # Fallback: part not found, use original
            reconstructed_parts.append(parts[i])
            position += len(part)
            continue

        # Calculate the end of this part
        part_end = part_start + len(part)

        # Check if there is a next part
        if i < len(parts_lower) - 1:
            next_part = parts_lower[i + 1]
            next_part_start = original_lower.find(next_part, part_end)

            if next_part_start > part_end:
                # There is a gap = linking element
                linking_element = original_lower[part_end:next_part_start]

                # Check if it is a known linking element
                if linking_element in ["s", "es", "n", "en", "er", "ns", "ens"]:
                    # Append linking element to current part
                    part_with_link = parts[i] + linking_element
                    reconstructed_parts.append(part_with_link)
                else:
                    # Unknown linking element, use original
                    reconstructed_parts.append(parts[i])
            else:
                # No linking element
                reconstructed_parts.append(parts[i])
        else:
            # Last part, no linking element needed
            reconstructed_parts.append(parts[i])

        position = part_end

    return reconstructed_parts


def _quality_filter_decomposition(word: str, parts: List[str]) -> bool:
    """
    Checks if a compound decomposition is qualitatively acceptable.
    Filters out obviously broken decompositions.

    Args:
        word: The original word
        parts: The proposed decomposition

    Returns:
        bool: True if the decomposition is acceptable
    """
    if not parts or len(parts) <= 1:
        return False

    # Filter 1: No very short or fragmented parts
    for part in parts:
        if len(part) < 3:
            return False
        # Check for suspicious fragmentation (only consonants or only vowels)
        if len(part) <= 4 and (
            part.isalpha()
            and (
                all(c in "bcdfghjklmnpqrstvwxyz" for c in part.lower())
                or all(c in "aeiou" for c in part.lower())
            )
        ):
            return False

    # Filter 2: Reconstructed word should resemble the original
    reconstructed = "".join(parts).lower()
    original_lower = word.lower()

    # Allow linking sound loss, but basic structure must match
    if len(reconstructed) < len(original_lower) * 0.6:
        return False

    # Filter 3: No nonsense decompositions like "ner venzu sam"
    # Check if at least one part resembles the original
    has_meaningful_part = False
    for part in parts:
        if len(part) >= 4 and part.lower() in original_lower:
            has_meaningful_part = True
            break

    if not has_meaningful_part:
        return False

    # Filter 4: Avoid typical error patterns
    # Note: "teil" removed — it's a valid compound component (Stadtteil, Vorteil, etc.)
    problematic_patterns = ["ung", "sam", "ven", "ner"]
    short_problematic_parts = [
        p for p in parts if len(p) <= 4 and p.lower() in problematic_patterns
    ]
    if len(short_problematic_parts) > 1:
        return False

    return True


def _split_with_compound_splitter(word: str) -> List[str]:
    """
    Splits a German word using the german_compound_splitter.
    Checks known compounds first for better quality.
    Implements quality filters against broken decompositions.

    Args:
        word: The word to decompose

    Returns:
        List[str]: List of components
    """
    # Performance cache: check if already decomposed
    word_lower = word.lower()
    if word_lower in _decomposition_cache:
        return _decomposition_cache[word_lower]

    # Check known compounds first for better quality
    if word_lower in KNOWN_COMPOUNDS:
        result = KNOWN_COMPOUNDS[word_lower]
        _decomposition_cache[word_lower] = result
        return result

    ahocs = _get_compound_splitter()

    if not ahocs or not COMPOUND_SPLITTER_AVAILABLE:
        # Fallback: simple length-based heuristic
        result = _simple_fallback_split(word)
        _decomposition_cache[word_lower] = result
        return result

    try:
        # Use german_compound_splitter for precise decomposition
        components = dissect(word, ahocs, only_nouns=True, make_singular=False)

        # Quality filter: check if decomposition is acceptable
        if not _quality_filter_decomposition(word, components):
            # Try alternative settings
            try:
                alt_components = dissect(
                    word, ahocs, only_nouns=False, make_singular=False
                )
                if _quality_filter_decomposition(word, alt_components):
                    components = alt_components
                else:
                    # Both decompositions are bad - use fallback
                    result = _simple_fallback_split(word)
                    _decomposition_cache[word_lower] = result
                    return result
            except Exception:
                # On errors: use fallback
                result = _simple_fallback_split(word)
                _decomposition_cache[word_lower] = result
                return result

        # If the first decomposition is incomplete, try more aggressive settings
        if len(components) <= 2 and len(word) > 15:
            try:
                # Try without only_nouns restriction for better decomposition
                alt_components = dissect(
                    word, ahocs, only_nouns=False, make_singular=False
                )
                if len(alt_components) > len(
                    components
                ) and _quality_filter_decomposition(word, alt_components):
                    components = alt_components
            except Exception:
                pass  # Fallback to original decomposition

        # Filter empty or very short parts
        filtered_parts = [part for part in components if len(part) >= 3]

        result = filtered_parts if filtered_parts else [word]

        # Cache the result
        _decomposition_cache[word_lower] = result
        return result

    except Exception as e:
        logger.warning("Fehler bei Compound-Splitting von '%s': %s", word, e)
        result = [word]
        _decomposition_cache[word_lower] = result
        return result


def _simple_fallback_split(word: str) -> List[str]:
    """
    Intelligent fallback when compound_splitter is unavailable or fails.
    Uses known German compound patterns.
    """
    word_lower = word.lower()

    # 1. Check for known endings that signal compound boundaries
    known_endings = [
        ("verwaltung", 10),
        ("management", 10),
        ("entwicklung", 11),
        ("sicherheit", 10),
        ("dienstleistung", 14),
        ("organisation", 12),
        ("kampagne", 8),
        ("institut", 8),
        ("abteilung", 9),
        ("service", 7),
        ("system", 6),
        ("anlage", 6),
        ("strategie", 9),
        ("ebene", 5),
        ("übertragung", 11),
        ("bestimmung", 10),
        ("vorschrift", 10),
        ("maßnahme", 8),
        ("einrichtung", 11),
        ("berg", 4),
        ("monster", 7),
    ]

    for ending, min_before_ending in known_endings:
        if word_lower.endswith(ending) and len(word) > len(ending) + min_before_ending:
            front_part = word[: -len(ending)]
            # Check if front part can be further decomposed
            if len(front_part) > 15:
                # Simple bisection of front part
                mid = len(front_part) // 2
                return [front_part[:mid], front_part[mid:], word[-len(ending) :]]
            else:
                return [front_part, word[-len(ending) :]]

    # 2. Check for known beginnings
    known_beginnings = [
        ("geschäfts", 9),
        ("projekt", 7),
        ("kunden", 6),
        ("daten", 5),
        ("kommunikation", 13),
        ("information", 11),
        ("arbeitsschutz", 12),
        ("umweltschutz", 12),
        ("verkehr", 7),
        ("bildung", 7),
        ("forschung", 9),
        ("personal", 8),
        ("finanz", 6),
        ("marketing", 9),
        ("vertrieb", 8),
    ]

    for beginning, min_after_beginning in known_beginnings:
        if word_lower.startswith(beginning) and len(word) > len(beginning) + min_after_beginning:
            back_part = word[len(beginning) :]
            # Check if back part can be further decomposed
            if len(back_part) > 15:
                # Simple bisection of back part
                mid = len(back_part) // 2
                return [word[: len(beginning)], back_part[:mid], back_part[mid:]]
            else:
                return [word[: len(beginning)], back_part]

    # 3. Very conservative: only obviously very long compounds
    if len(word) > 20:
        # Smarter bisection - avoid splitting in the middle of syllables
        mid = len(word) // 2
        # Search for better split point near the middle
        for offset in [0, 1, -1, 2, -2, 3, -3]:
            split_point = mid + offset
            if 6 <= split_point <= len(word) - 6:
                part1, part2 = word[:split_point], word[split_point:]
                # Avoid splits with many consonants at end/start
                if not (
                    part1[-3:].count("bcdfghjklmnpqrstvwxyz") >= 3
                    or part2[:3].count("bcdfghjklmnpqrstvwxyz") >= 3
                ):
                    return [part1, part2]

        # Fallback to simple bisection
        return [word[:mid], word[mid:]]

    # 4. Short words are not decomposed
    return [word]


def _intelligent_split_with_germalemma(word: str) -> List[str]:
    """
    Intelligent decomposition that optimally uses GermaLemma.
    GermaLemma is used for word validation, not for compound splitting.

    Args:
        word: The word to decompose (lowercase)

    Returns:
        List[str]: List of components
    """
    # First: use known compound mappings
    if word in KNOWN_COMPOUNDS:
        return KNOWN_COMPOUNDS[word]

    # For unknown words: intelligent heuristic decomposition with GermaLemma validation
    if GERMALEMMA_AVAILABLE:
        return _heuristic_split_with_germalemma_validation(word)
    else:
        # Fallback without GermaLemma
        return _simple_heuristic_split(word)


def _heuristic_split_with_germalemma_validation(word: str) -> List[str]:
    """
    Heuristic decomposition with GermaLemma for validation of word parts.
    """
    if len(word) < 10:  # Short words are not decomposed
        return [word]

    germalemma = _get_germalemma()

    # Try different decomposition strategies
    best_decomposition = [word]

    # Strategy 1: 2-way split at various points
    for i in range(4, len(word) - 3):
        part1 = word[:i]
        rest = word[i:]

        # Try with and without linking sounds
        for rest_candidate in _remove_linking_sound_variants(rest):
            if len(rest_candidate) >= 4:
                if _is_german_word_with_germalemma(
                    part1, germalemma
                ) and _is_german_word_with_germalemma(rest_candidate, germalemma):
                    # Recursive decomposition of second part
                    rest_parts = _heuristic_split_with_germalemma_validation(
                        rest_candidate
                    )
                    candidate = [part1] + rest_parts

                    if len(candidate) > len(best_decomposition):
                        best_decomposition = candidate

    return best_decomposition


def _remove_linking_sound_variants(word: str) -> List[str]:
    """
    Generates variants of a word with removed linking elements.
    """
    variants = [word]  # Original without change

    for link in ["s", "es", "n", "en", "er"]:
        if word.startswith(link) and len(word) > len(link) + 2:
            variants.append(word[len(link) :])

    return variants


def _is_german_word_with_germalemma(word: str, germalemma) -> bool:
    """
    Uses GermaLemma optimally: checks if a word is a valid German word.
    """
    if len(word) < 4:
        return False

    # Check with GermaLemma as different parts of speech
    for pos in ["N", "V", "A"]:  # Noun, Verb, Adjective
        try:
            lemma = germalemma.find_lemma(word, pos)
            if lemma:  # GermaLemma recognized the word
                return True
        except Exception:
            continue

    return False


def _simple_heuristic_split(word: str) -> List[str]:
    """
    Simple fallback decomposition without GermaLemma.
    """
    if len(word) < 12:
        return [word]

    # Very conservative: only obvious cases

    for i in range(6, len(word) - 5):
        part1 = word[:i]
        part2 = word[i:]

        if word_frequency(part1, "de") > 2e-6 and word_frequency(part2, "de") > 2e-6:
            return [part1, part2]

    return [word]


def _split_recursively(word: str) -> List[str]:
    """
    Compatibility function - delegates to the new compound_splitter implementation.
    """
    return _split_with_compound_splitter(word)


def _heuristic_split(word: str) -> List[str]:
    """
    Improved heuristic decomposition for German compounds.
    Tries to find shorter parts for maximum decomposition.
    """
    if len(word) < 8:
        return [word]

    # Recursive decomposition for maximum number of parts
    def _split_greedy_recursive(remaining: str) -> List[str]:
        if len(remaining) <= 4:  # Very short remainders are not further decomposed
            return [remaining] if remaining else []

        # Start with shorter parts for maximum decomposition
        for min_length in [4, 5, 6]:  # Try different minimum lengths
            for i in range(
                min_length, min(len(remaining) - 2, 12)
            ):  # Not too long parts
                candidate = remaining[:i]
                freq = word_frequency(candidate, "de")

                if freq > 1e-7:  # Valid German word found
                    rest_after_candidate = remaining[i:]

                    # Remove possible linking sounds
                    for link in ["s", "es", "n", "en", "er"]:
                        if rest_after_candidate.startswith(link) and len(
                            rest_after_candidate
                        ) > len(link):
                            rest_after_candidate = rest_after_candidate[len(link) :]
                            break

                    # Try recursive decomposition of the rest
                    rest_parts = _split_greedy_recursive(rest_after_candidate)

                    # Only accept if the rest is also decomposable or very short
                    if rest_parts and (
                        len(rest_parts) > 1 or len(rest_after_candidate) <= 6
                    ):
                        return [candidate] + rest_parts

        # No good decomposition found
        return [remaining]

    result = _split_greedy_recursive(word.lower())

    # Filter very short or obviously wrong parts
    filtered_parts = []
    for part in result:
        if len(part) >= 3 and word_frequency(part, "de") > 5e-8:
            filtered_parts.append(part)
        elif len(part) >= 3:  # Keep unknown longer parts too
            filtered_parts.append(part)

    return filtered_parts if len(filtered_parts) > 1 else [word]


def _conservative_heuristic_split(word: str) -> List[str]:
    """
    Conservative decomposition for fallback without GermaLemma.
    Less aggressive than the full heuristic decomposition.
    """
    if len(word) < 12:
        return [word]

    # Try only a simple 2-way split for very long words
    for i in range(6, len(word) - 4):
        part1 = word[:i]
        part2 = word[i:]

        # Remove possible linking sounds from the second part
        for link in ["s", "es", "n", "en", "er"]:
            if part2.startswith(link) and len(part2) > len(link) + 3:
                part2_without_link = part2[len(link) :]
                if (
                    word_frequency(part1, "de") > 1e-7
                    and word_frequency(part2_without_link, "de") > 1e-7
                ):
                    return [part1, part2_without_link]

        # Without linking element
        if word_frequency(part1, "de") > 1e-7 and word_frequency(part2, "de") > 1e-7:
            return [part1, part2]

    return [word]


def _extended_heuristic_split(word: str) -> List[str]:
    """
    Extended decomposition with GermaLemma validation.
    """
    germalemma = _get_germalemma()

    # Start with heuristic decomposition
    heuristic_parts = _heuristic_split(word)

    if len(heuristic_parts) <= 1:
        return heuristic_parts

    # Validate each part with GermaLemma
    validated_parts = []
    for part in heuristic_parts:
        try:
            # Check if the part is recognized as a German word
            lemma = germalemma.find_lemma(part, "N")
            if lemma and lemma != part:
                # Lemmatization found a different word -> use that
                validated_parts.append(lemma)
            else:
                validated_parts.append(part)
        except Exception:
            validated_parts.append(part)

    return validated_parts


def _fallback_compound_check(word: str) -> bool:
    """
    Fallback implementation without GermaLemma for basic compound detection.
    """
    # Simple heuristics for German compounds
    if len(word) < 10:
        return False

    # Check for typical linking elements
    linking_sounds = ["s", "es", "n", "en", "er"]
    for link in linking_sounds:
        if link in word[3:-3]:  # Not at the beginning or end
            return True

    # Very long words are probably compounds
    return len(word) > 20


# --- OLD CORE LOGIC (compatibility) ---

# PERFORMANCE FIX: Cache for word validation
_word_validity_cache = {}


def is_valid_word(word: str, nlp: spacy.Language = None) -> bool:
    """
    OPTIMIZED: Checks if a word is valid with caching for performance.
    """
    if len(word) < 3:
        return False

    word_lower = word.lower()

    # Cache lookup for performance
    if word_lower in _word_validity_cache:
        return _word_validity_cache[word_lower]

    # Check own dictionary first
    if word_lower in OWN_DICTIONARY:
        _word_validity_cache[word_lower] = True
        return True

    # Check word frequency without spaCy (much faster)
    frequency = word_frequency(word_lower, LANGUAGE)
    result = frequency > WORD_FREQ_THRESHOLD

    # Cache the result
    _word_validity_cache[word_lower] = result
    return result


def split_compound_greedy(
    word: str, nlp: spacy.Language = None
) -> Optional[List[str]]:
    """
    Splits a compound by searching for the longest valid part from the left.
    """
    remaining = word.lower()
    resulting_parts = []

    while len(remaining) > 0:
        best_match = ""
        for i in range(len(remaining), 2, -1):
            candidate = remaining[:i]
            if is_valid_word(candidate):
                best_match = candidate
                break

        if best_match:
            resulting_parts.append(best_match)
            remaining = remaining[len(best_match) :]

            for link in LINKING_SOUNDS:
                if remaining.startswith(link) and len(remaining) > len(link):
                    remaining = remaining[len(link) :]
                    break
        else:
            return None

    if len(resulting_parts) > 1:
        return resulting_parts
    return None


# Global nlp object for performance
_global_nlp = None


def _get_nlp():
    global _global_nlp
    if _global_nlp is None:
        _global_nlp = spacy.load("de_core_news_lg")
    return _global_nlp


def check_rule(doc: Doc) -> List[str]:
    """
    Modernized compound detection with german_compound_splitter.
    Detects only truly complex compounds (3+ components).
    """
    errors = []
    processed_tokens = set()

    # MAIN LOGIC: Use new compound_splitter-based detection
    for token in doc:
        if token.i in processed_tokens:
            continue

        # Check only relevant parts of speech (nouns and adjectives)
        if token.pos_ in ["NOUN", "ADJ"] and len(token.text) >= 6:
            # Use the new configurable compound detection
            # Skip words that already contain hyphens - they're already in recommended form
            if "-" in token.text:
                continue

            if is_compound_by_config(token.text):
                # Determine compound type and difficulty
                parts = _split_with_compound_splitter(token.text)
                word_length = len(token.text)
                part_count = len(parts)

                # Type determination based on configuration and properties
                if CompoundConfig.MODE == CompoundConfig.ALL_COMPOUNDS:
                    if part_count >= 4:
                        compound_type = "Komplexes Kompositum"
                        difficulty = "hoch"
                    elif part_count == 3:
                        compound_type = "Zusammengesetztes Wort"
                        difficulty = "mittel"
                    else:
                        compound_type = "Kompositum"
                        difficulty = "niedrig"
                elif word_length > 25:
                    compound_type = "Sehr langes Kompositum"
                    difficulty = "sehr hoch"
                elif word_length > 20:
                    compound_type = "Langes Kompositum"
                    difficulty = "hoch"
                else:
                    compound_type = "Komplexes Kompositum"
                    difficulty = "mittel"

                # Generate improvement suggestion (always both options)
                suggestions = []

                # Always first: replacement with simpler words
                suggestions.append(
                    "Ersetzen Sie durch ein einfacheres Wort oder verwenden Sie Beistriche um das Wort besser leserlich zu machen."
                )

                # If possible: hyphen separation
                if len(parts) >= 2:
                    # Reconstruct parts with linking sounds
                    parts_with_links = _reconstruct_with_linking_sounds(
                        token.text, parts
                    )

                    # DEFENSIVE PROGRAMMING: Quality check of suggestions
                    if _quality_filter_decomposition(token.text, parts_with_links):
                        # Additional safety check: no very short or problematic parts
                        if all(len(part) >= 3 for part in parts_with_links) and not any(
                            part.lower() in ["ung", "sam", "ven", "ner"]
                            for part in parts_with_links
                            if len(part) <= 4
                        ):
                            # Suggest hyphen separation
                            hyphen_form = "-".join(
                                part.lower() for part in parts_with_links
                            )
                            suggestions.append(
                                f'Falls nicht möglich, schreiben Sie "{hyphen_form}"'
                            )

                # Compose the suggestion
                suggestion = ". ".join(suggestions)

                errors.append(
                    f'{compound_type} "{token.text}" ({difficulty}). '
                    f"Besser: {suggestion.capitalize()}."
                )
                processed_tokens.add(token.i)

    return errors


def is_compound_candidate(token: Token) -> bool:
    """
    Detects compound candidates through linguistic features.
    Modern morphological validation.
    """
    # Check word length and frequency
    if len(token.text) < MIN_COMPOUND_LENGTH:
        return False

    # Check if word is rare enough (compounds are often rarer)
    frequency = word_frequency(token.lemma_.lower(), LANGUAGE)

    # Very frequent words are probably not decomposable compounds
    if frequency > WORD_FREQ_THRESHOLD * 10:
        return False

    # Check morphological hints for compound
    if token.pos_ == "NOUN":
        # German nominal compounds are usually long and have certain patterns
        return len(token.text) >= 12 or has_compound_pattern(token.text)
    elif token.pos_ == "ADJ":
        # Adjective compounds are often shorter but have special patterns
        return len(token.text) >= 10 and has_adjective_compound_pattern(token.text)

    return False


def has_compound_pattern(word: str) -> bool:
    """Detects typical German compound patterns."""
    # Check for typical linking elements
    for link in LINKING_SOUNDS:
        if link in word[2:-2]:  # Not at beginning or end
            return True

    # Check for double consonants (often at linking points)
    import re

    if re.search(r"([bcdfghjklmnpqrstvwxyz])\1", word.lower()):
        return True

    return False


def has_adjective_compound_pattern(word: str) -> bool:
    """Special patterns for adjective compounds."""
    # Typical adjective compounds often end in -ig, -lich, -bar, -los
    suffixes = ["ig", "lich", "bar", "los", "reich", "arm", "frei"]
    return any(word.lower().endswith(suffix) for suffix in suffixes)


def analyze_semantic_decomposition(token: Token, nlp) -> Dict:
    """
    Performs semantic compound decomposition.
    State-of-the-art NLP-based word formation analysis.
    """
    word = token.text.lower()

    # Check own dictionary first
    if word in OWN_DICTIONARY:
        return {
            "ist_kompositum": True,
            "methode": "Wörterbuch",
            "teile": None,
            "konfidenz": "hoch",
        }

    # Try greedy decomposition
    decomposition = split_compound_greedy(word, nlp)

    if decomposition and len(decomposition) >= 2:
        # Check semantic coherence of parts
        semantics_score = evaluate_semantic_coherence(decomposition, nlp)

        return {
            "ist_kompositum": True,
            "methode": "Semantische Zerlegung",
            "teile": decomposition,
            "konfidenz": "hoch" if semantics_score > 0.7 else "mittel",
            "semantik_score": semantics_score,
        }

    # Check if very long word (probably compound even without decomposition)
    if len(word) > 20:
        return {
            "ist_kompositum": True,
            "methode": "Längen-Heuristik",
            "teile": None,
            "konfidenz": "niedrig",
        }

    return {"ist_kompositum": False}


def evaluate_semantic_coherence(parts: List[str], nlp) -> float:
    """OPTIMIZED: Evaluates semantic coherence of compound parts."""
    if len(parts) < 2:
        return 0.0

    # PERFORMANCE FIX: Check if all parts are valid words without nlp
    valid_parts = sum(1 for part in parts if is_valid_word(part))
    valid_ratio = valid_parts / len(parts)

    # Bonus for typical compound structures
    structure_bonus = 0.0
    if len(parts) == 2:  # Binary compounds are typical
        structure_bonus = 0.2
    elif parts[-1] in ["ung", "heit", "keit", "schaft"]:  # Nominalizations
        structure_bonus = 0.3

    return min(1.0, valid_ratio + structure_bonus)


def evaluate_readability(token: Token, decomposition_info: Dict) -> Dict[str, str]:
    """
    Evaluates readability and generates context-specific suggestions.
    """
    word_length = len(token.text)
    parts = decomposition_info.get("teile", [])

    # Determine compound type
    if decomposition_info["methode"] == "Wörterbuch":
        compound_type = "Bekanntes Kompositum"
        difficulty = "hoch"
    elif word_length > 25:
        compound_type = "Sehr langes Kompositum"
        difficulty = "sehr hoch"
    elif word_length > 20:
        compound_type = "Langes Kompositum"
        difficulty = "hoch"
    else:
        compound_type = "Kompositum"
        difficulty = "mittel"

    # Generate suggestion
    if parts and len(parts) >= 2:
        if len(parts) == 2:
            suggestion = (
                f'"{parts[0].capitalize()}-{parts[1]}" oder "{parts[0]} für {parts[1]}"'
            )
        else:
            # Multi-part compounds split up
            suggestion = (
                f'Teilen Sie auf: "{" ".join(part.capitalize() for part in parts)}"'
            )
    else:
        suggestion = "Verwenden Sie einfachere Wörter oder erklären Sie die Bedeutung."

    return {"typ": compound_type, "schwierigkeit": difficulty, "vorschlag": suggestion}


def detect_hidden_compounds(doc: Doc, nlp) -> List[Dict]:
    """
    Detects hidden compounds with state-of-the-art linguistic patterns.
    """
    hidden = []

    # Detect adjective compounds with typical patterns
    for token in doc:
        if (
            token.pos_ == "ADJ"
            and len(token.text) >= 8
            and has_adjective_compound_pattern(token.text)
            and not is_valid_word(token.text)
        ):
            hidden.append(
                {
                    "token": token,
                    "nachricht": f'Verstecktes Adjektiv-Kompositum "{token.text}" ist schwer lesbar. Besser: Beschreiben Sie mit einfachen Wörtern.',
                }
            )

    return hidden


def analyze_word_formation_complexity(doc: Doc) -> List[str]:
    """
    Analyzes the word formation complexity of the entire text.
    """
    warnings = []

    # Count compounds per sentence
    for i, sent in enumerate(doc.sents, 1):
        compound_count = sum(
            1
            for token in sent
            if token.pos_ in ["NOUN", "ADJ"]
            and len(token.text) >= MIN_COMPOUND_LENGTH
        )

        if compound_count >= 3:
            warnings.append(
                f"Satz {i} enthält {compound_count} potentielle Komposita. "
                f"Besser: Verwenden Sie max. 2 zusammengesetzte Wörter pro Satz."
            )

    return warnings


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Test of the modernized compound_splitter-based compound rule

    logger.info("--- Test: german_compound_splitter-based is_complex_compound function ---")

    # Test cases according to requirements
    test_words = [
        "Haus",  # 1 component -> False
        "Haustür",  # 2 components -> False
        "Festplattenrekorder",  # 3 components -> True
        "Bundespräsidentenstichwahl",  # 4+ components -> True
        "Kommunikation",  # Not decomposable -> False
        "Datenschutzgrundverordnung",  # Very long compound -> True
        "Projektentwicklungsleiter",  # 3+ components -> True
        "Auto",  # Simple word -> False
        "Computerbildschirm",  # 3 components -> True
        "Sonnenstrahlen",  # 2 components -> False
    ]

    for word in test_words:
        result = is_complex_compound(word)
        parts = _split_with_compound_splitter(word)
        logger.info("%30 -> %5 (Parts: %s)", word, result, parts)

    logger.info("\n--- Test: Compound rule in spaCy Doc ---")

    try:
        nlp = spacy.load("de_core_news_lg")

        test_text = """Die Regierungskommission diskutierte über Datenschutzgrundverordnungen.
        Der Projektentwicklungsleiter präsentierte Softwareentwicklungsrichtlinien.
        Das Auto parkte vor dem Haus. Die Haustür war offen."""

        doc = nlp(test_text)
        results = check_rule(doc)

        if results:
            logger.info("Found: %s complex compounds (3+ parts)", len(results))
            for i, error in enumerate(results, 1):
                logger.error("%s. %s", i, error)
        else:
            logger.info(" No complex compounds (3+ parts) found.")

    except OSError:
        logger.info(" German spaCy model not available. Install with: python -m spacy download de_core_news_lg")

    logger.info("\nCompound Splitter available: %s", COMPOUND_SPLITTER_AVAILABLE)

    # Show current configuration
    logger.info("\n" + "=" * 80)
    logger.info(" CURRENT CONFIGURATION")
    logger.info("=" * 80)
    logger.info("Mode: %s", CompoundConfig.MODE)
    logger.info("Min. word length: %s characters", CompoundConfig.MIN_WORD_LENGTH)
    logger.info("Min. part count: %s parts", CompoundConfig.MIN_PART_COUNT)
    logger.info("Min. compound length: %s characters", CompoundConfig.MIN_COMPOUND_LENGTH)
    logger.info("Hyphen suggestion: %s", CompoundConfig.TRENNZEICHEN_VORSCHLAG)
    logger.info("Replacement suggestion: %s", CompoundConfig.ERSETZUNG_VORSCHLAG)
    logger.info("")

    logger.info(" AVAILABLE CONFIGURATIONS:")
    logger.info("- CompoundConfig.config_strict() # All compounds (2+ parts)")
    logger.info("- CompoundConfig.config_moderate() # Only 3+ parts (default)")
    logger.info("- CompoundConfig.config_length() # Only very long words")
    logger.info("- CompoundConfig.config_liberal() # Only 4+ parts")
    logger.info("")

    logger.info(" CHANGE CONFIGURATION:")
    logger.info("Edit lines 37-55 in this file:")
    logger.info("- MODE = CompoundConfig.ALL_COMPOUNDS # For all compounds")
    logger.info("- MODE = CompoundConfig.BY_LENGTH # For length-based detection")
    logger.info("- MODE = CompoundConfig.BY_PARTS # For parts-based detection")
    logger.info("")
    logger.info("Example for strict detection:")
    logger.info(" MODE = ALL_COMPOUNDS")
    logger.info(" MIN_PART_COUNT = 2")
    logger.info(" TRENNZEICHEN_VORSCHLAG = True")
    logger.info("")
    logger.info(" Save the file and run 'python regel_checker.py'!")
