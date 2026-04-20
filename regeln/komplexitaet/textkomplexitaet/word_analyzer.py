"""
Word-level complexity analysis.

Identifies which specific words contribute to text complexity.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from spacy.tokens import Doc, Token

from .config import (
    DEFAULT_CONFIG,
    ComplexityConfig,
    get_suggestion,
    is_whitelisted,
)
from .models import predict_complexity

logger = logging.getLogger(__name__)


@dataclass
class WordComplexity:
    """Result of word complexity analysis."""

    word: str
    complexity_score: float
    complexity_category: str  # "simple", "medium", "complex"
    reason: str
    suggestion: Optional[str] = None
    category: Optional[str] = None  # "Fremdwort", "Fachsprache", etc.
    position: int = 0  # Character position in text
    ablation_impact: float = 0.0  # How much removing this word reduces complexity


@dataclass
class SentenceComplexity:
    """Result of sentence complexity analysis."""

    sentence: str
    complexity_score: float
    complexity_category: str
    complex_words: List[WordComplexity] = field(default_factory=list)


class WordComplexityAnalyzer:
    """Analyzes word-level complexity using multiple signals."""

    def __init__(
        self,
        config: Optional[ComplexityConfig] = None,
        model_manager=None,
    ):
        """
        Initialize the word complexity analyzer.

        Args:
            config: Configuration settings
            model_manager: Deprecated, ignored. Kept for backward compatibility.
        """
        self.config = config or DEFAULT_CONFIG

    def _categorize_score(self, score: float) -> str:
        """Convert numeric score to category."""
        if score < self.config.simple_threshold:
            return "simple"
        elif score < self.config.medium_threshold:
            return "medium"
        return "complex"

    def _check_known_complex_word(
        self, word: str, lemma: Optional[str] = None
    ) -> Optional[WordComplexity]:
        """Check if word is in the known complex words dictionary."""
        # Try the word itself first
        info = get_suggestion(word)

        # If not found, try the lemma (base form)
        if not info and lemma:
            info = get_suggestion(lemma)

        if info:
            return WordComplexity(
                word=word,
                complexity_score=6.0,  # High score for known complex words
                complexity_category="complex",
                reason=f"{info['category']}: bekanntes schwieriges Wort",
                suggestion=info["simple"],
                category=info["category"],
            )
        return None

    def _check_morphological_complexity(
        self, token: Token
    ) -> Optional[Tuple[float, str]]:
        """
        Check morphological complexity using spaCy features.

        Returns:
            Tuple of (complexity_bonus, reason) or None
        """
        reasons = []
        bonus = 0.0

        # Very long words
        if len(token.text) > 15:
            bonus += 1.5
            reasons.append(f"sehr lang ({len(token.text)} Zeichen)")
        elif len(token.text) > 12:
            bonus += 0.8
            reasons.append(f"lang ({len(token.text)} Zeichen)")

        # Foreign word patterns (suffixes)
        foreign_suffixes = [
            "tion",
            "sion",
            "ment",
            "ismus",
            "ität",
            "enz",
            "anz",
            "ieren",
            "ierung",
            "ik",
            "ie",
            "ur",
            "or",
            "är",
        ]
        word_lower = token.text.lower()
        for suffix in foreign_suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 3:
                bonus += 1.0
                reasons.append(f"Fremdwort-Endung '-{suffix}'")
                break

        # Foreign word patterns (prefixes)
        foreign_prefixes = [
            "pre",
            "pro",
            "anti",
            "inter",
            "trans",
            "multi",
            "super",
            "sub",
            "ex",
            "re",
            "de",
            "dis",
            "un",
            "in",
        ]
        for prefix in foreign_prefixes:
            if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 4:
                # Check it's not a common German word
                if not is_whitelisted(word_lower):
                    bonus += 0.5
                    reasons.append(f"Fremdwort-Vorsilbe '{prefix}-'")
                    break

        if reasons:
            return bonus, "; ".join(reasons)
        return None

    def _analyze_word_with_ablation(
        self, word: str, sentence: str, base_score: float
    ) -> float:
        """
        Analyze word impact by removing it and measuring complexity change.

        Args:
            word: Word to analyze
            sentence: Full sentence
            base_score: Complexity score of full sentence

        Returns:
            Impact score (how much complexity drops when word is removed)
        """
        # Create sentence without the word
        pattern = r"\b" + re.escape(word) + r"\b"
        sentence_without = re.sub(pattern, "", sentence, count=1)
        sentence_without = " ".join(sentence_without.split())  # Clean whitespace

        if not sentence_without.strip():
            return 0.0

        # Get complexity of sentence without the word
        score_without, _ = predict_complexity(sentence_without)

        # Impact is how much complexity drops
        return max(0, base_score - score_without)

    def analyze_sentence(
        self, sentence: str, doc: Optional[Doc] = None
    ) -> SentenceComplexity:
        """
        Analyze a sentence for complexity at word level.

        Args:
            sentence: Sentence text
            doc: Optional spaCy Doc for additional features

        Returns:
            SentenceComplexity with overall score and complex words
        """
        # Get baseline sentence complexity
        base_score, _ = predict_complexity(sentence)
        base_category = self._categorize_score(base_score)

        complex_words = []

        # If we have a spaCy doc, use it for better analysis
        if doc is not None:
            for token in doc:
                if not token.is_alpha or token.is_stop or token.is_punct:
                    continue

                if len(token.text) < self.config.min_word_length:
                    continue

                word = token.text
                word_lower = word.lower()

                # Skip whitelisted words (check both word and lemma)
                lemma = token.lemma_.lower()
                if is_whitelisted(word_lower) or is_whitelisted(lemma):
                    continue

                # Check known complex words first (using both word and lemma)
                known_result = self._check_known_complex_word(word_lower, lemma)
                if known_result:
                    known_result.position = token.idx
                    complex_words.append(known_result)
                    continue

                # Check morphological complexity
                morph_result = self._check_morphological_complexity(token)
                morph_bonus = morph_result[0] if morph_result else 0.0
                morph_reason = morph_result[1] if morph_result else ""

                # Perform ablation analysis for potentially complex words
                if morph_bonus > 0.5 or base_score > self.config.simple_threshold:
                    ablation_impact = self._analyze_word_with_ablation(
                        word, sentence, base_score
                    )

                    # Combine signals
                    word_score = base_score + morph_bonus

                    if (
                        ablation_impact >= self.config.word_ablation_threshold
                        or morph_bonus >= 1.0
                    ):
                        reasons = []
                        if ablation_impact >= self.config.word_ablation_threshold:
                            reasons.append(
                                f"erhöht Satzkomplexität um {ablation_impact:.1f}"
                            )
                        if morph_reason:
                            reasons.append(morph_reason)

                        word_complexity = WordComplexity(
                            word=word,
                            complexity_score=word_score,
                            complexity_category=self._categorize_score(word_score),
                            reason="; ".join(reasons)
                            if reasons
                            else "potenziell schwierig",
                            position=token.idx,
                            ablation_impact=ablation_impact,
                        )

                        # Try to find suggestion
                        suggestion_info = get_suggestion(word_lower)
                        if suggestion_info:
                            word_complexity.suggestion = suggestion_info["simple"]
                            word_complexity.category = suggestion_info["category"]

                        complex_words.append(word_complexity)

        else:
            # Fallback: analyze without spaCy features
            words = sentence.split()
            position = 0

            for word in words:
                # Clean punctuation
                clean_word = re.sub(r"[^\w]", "", word)
                if not clean_word or len(clean_word) < self.config.min_word_length:
                    position += len(word) + 1
                    continue

                word_lower = clean_word.lower()

                # Skip whitelisted
                if is_whitelisted(word_lower):
                    position += len(word) + 1
                    continue

                # Check known complex words
                known_result = self._check_known_complex_word(word_lower)
                if known_result:
                    known_result.position = position
                    complex_words.append(known_result)
                    position += len(word) + 1
                    continue

                # Simple length-based check
                if len(clean_word) > 12:
                    ablation_impact = self._analyze_word_with_ablation(
                        clean_word, sentence, base_score
                    )

                    if ablation_impact >= self.config.word_ablation_threshold:
                        complex_words.append(
                            WordComplexity(
                                word=clean_word,
                                complexity_score=base_score + 1.0,
                                complexity_category="complex",
                                reason=f"langes Wort ({len(clean_word)} Zeichen), erhöht Komplexität um {ablation_impact:.1f}",
                                position=position,
                                ablation_impact=ablation_impact,
                            )
                        )

                position += len(word) + 1

        # Sort by complexity score (highest first)
        complex_words.sort(key=lambda w: w.complexity_score, reverse=True)

        return SentenceComplexity(
            sentence=sentence,
            complexity_score=base_score,
            complexity_category=base_category,
            complex_words=complex_words,
        )

    def get_complex_words(
        self, text: str, doc: Optional[Doc] = None, threshold: Optional[float] = None
    ) -> List[WordComplexity]:
        """
        Get all complex words from text.

        Args:
            text: Input text
            doc: Optional spaCy Doc
            threshold: Minimum complexity score (default from config)

        Returns:
            List of WordComplexity objects for complex words
        """
        threshold = threshold or self.config.medium_threshold

        # Split into sentences if needed
        if doc is not None:
            sentences = [(sent.text, sent.as_doc()) for sent in doc.sents]
        else:
            # Simple sentence splitting
            sentences = [
                (s.strip(), None) for s in re.split(r"[.!?]+", text) if s.strip()
            ]

        all_complex_words = []

        for sent_text, sent_doc in sentences:
            result = self.analyze_sentence(sent_text, sent_doc)

            for word in result.complex_words:
                if word.complexity_score >= threshold:
                    all_complex_words.append(word)

        return all_complex_words


def analyze_words(text: str, doc: Optional[Doc] = None) -> List[WordComplexity]:
    """
    Convenience function to analyze words in text.

    Args:
        text: Input text
        doc: Optional spaCy Doc

    Returns:
        List of complex words found
    """
    analyzer = WordComplexityAnalyzer()
    return analyzer.get_complex_words(text, doc)
