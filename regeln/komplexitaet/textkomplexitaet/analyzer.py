"""
Main API for TextKomplexitaet analyzer.

Provides a unified interface for text complexity analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from spacy.tokens import Doc

from .config import DEFAULT_CONFIG, ComplexityConfig
from .models import predict_complexity
from .word_analyzer import SentenceComplexity, WordComplexity, WordComplexityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class ComplexityResult:
    """Complete result of text complexity analysis."""

    text: str
    overall_score: float
    overall_category: str  # "simple", "medium", "complex"
    sentence_count: int
    complex_word_count: int
    sentences: List[SentenceComplexity] = field(default_factory=list)
    complex_words: List[WordComplexity] = field(default_factory=list)

    @property
    def is_simple(self) -> bool:
        """Check if text is considered simple."""
        return self.overall_category == "simple"

    @property
    def is_complex(self) -> bool:
        """Check if text has complex elements."""
        return self.overall_category == "complex" or self.complex_word_count > 0

    def get_violations(self) -> List[str]:
        """
        Get violation messages in the format used by the rule system.

        Returns:
            List of violation messages
        """
        violations = []

        for word in self.complex_words:
            if word.suggestion:
                msg = (
                    f"{word.category or 'Komplexes Wort'}: "
                    f'"{word.word}" - {word.reason}. '
                    f'Vorschlag: "{word.suggestion}"'
                )
            else:
                msg = (
                    f'Komplexes Wort: "{word.word}" - {word.reason}. '
                    f"Bitte prüfen Sie, ob ein einfacheres Wort möglich ist."
                )

            violations.append(msg)

        return violations

    def get_annotated_text(self) -> str:
        """
        Get text with complex words annotated.

        Returns:
            Text with annotations in format: word[issue: suggestion]
        """
        if not self.complex_words:
            return self.text

        # Sort words by position (reverse to process from end)
        sorted_words = sorted(
            self.complex_words, key=lambda w: w.position, reverse=True
        )

        annotated = self.text

        for word in sorted_words:
            if word.suggestion:
                annotation = f"[{word.category or 'komplex'}: {word.suggestion}]"
            else:
                annotation = "[komplex: prüfen]"

            # Find and annotate the word
            start = word.position
            end = start + len(word.word)

            if start >= 0 and end <= len(annotated):
                annotated = annotated[:end] + annotation + annotated[end:]

        return annotated


class TextComplexityAnalyzer:
    """
    Main analyzer for German text complexity.

    This analyzer uses a pre-trained model to assess text complexity
    and identifies specific words that contribute to complexity.

    Example:
        >>> analyzer = TextComplexityAnalyzer()
        >>> result = analyzer.analyze("Die Administration evaluiert das Konzept.")
        >>> print(result.overall_score)
        5.2
        >>> for word in result.complex_words:
        ...     print(f"{word.word}: {word.suggestion}")
        Administration: Verwaltung
        evaluiert: prüft

    """

    def __init__(self, config: Optional[ComplexityConfig] = None):
        """
        Initialize the analyzer.

        Args:
            config: Optional configuration settings
        """
        self.config = config or DEFAULT_CONFIG
        self._word_analyzer: Optional[WordComplexityAnalyzer] = None

    @property
    def word_analyzer(self) -> WordComplexityAnalyzer:
        """Get the word analyzer (lazy initialization)."""
        if self._word_analyzer is None:
            self._word_analyzer = WordComplexityAnalyzer(config=self.config)
        return self._word_analyzer

    def _categorize_score(self, score: float) -> str:
        """Convert numeric score to category."""
        if score < self.config.simple_threshold:
            return "simple"
        elif score < self.config.medium_threshold:
            return "medium"
        return "complex"

    def analyze(
        self,
        text: str,
        doc: Optional[Doc] = None,
        include_sentence_analysis: bool = True,
    ) -> ComplexityResult:
        """
        Analyze text complexity.

        Args:
            text: Input text to analyze
            doc: Optional spaCy Doc for enhanced analysis
            include_sentence_analysis: Whether to analyze individual sentences

        Returns:
            ComplexityResult with overall score and complex words
        """
        if not text or not text.strip():
            return ComplexityResult(
                text=text,
                overall_score=1.0,
                overall_category="simple",
                sentence_count=0,
                complex_word_count=0,
            )

        # Get overall text complexity
        overall_score, _ = predict_complexity(text)
        overall_category = self._categorize_score(overall_score)

        sentences = []
        all_complex_words = []

        if include_sentence_analysis:
            # Analyze by sentence
            if doc is not None:
                sent_list = list(doc.sents)
            else:
                # Simple sentence splitting
                import re

                sent_texts = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
                sent_list = sent_texts

            for sent in sent_list:
                if doc is not None:
                    sent_text = sent.text
                    sent_doc = sent.as_doc()
                else:
                    sent_text = sent
                    sent_doc = None

                sent_result = self.word_analyzer.analyze_sentence(sent_text, sent_doc)
                sentences.append(sent_result)
                all_complex_words.extend(sent_result.complex_words)

        else:
            # Just get complex words without sentence breakdown
            all_complex_words = self.word_analyzer.get_complex_words(text, doc)

        # Deduplicate complex words (same word might appear multiple times)
        seen_words = set()
        unique_complex_words = []
        for word in all_complex_words:
            key = (word.word.lower(), word.position)
            if key not in seen_words:
                seen_words.add(key)
                unique_complex_words.append(word)

        return ComplexityResult(
            text=text,
            overall_score=overall_score,
            overall_category=overall_category,
            sentence_count=len(sentences),
            complex_word_count=len(unique_complex_words),
            sentences=sentences,
            complex_words=unique_complex_words,
        )

    def analyze_word(self, word: str, context: Optional[str] = None) -> WordComplexity:
        """
        Analyze a single word's complexity.

        Args:
            word: Word to analyze
            context: Optional sentence context

        Returns:
            WordComplexity result
        """
        if context:
            result = self.word_analyzer.analyze_sentence(context)
            for w in result.complex_words:
                if w.word.lower() == word.lower():
                    return w

        # Analyze word in isolation
        from .config import get_suggestion, is_whitelisted

        if is_whitelisted(word.lower()):
            return WordComplexity(
                word=word,
                complexity_score=1.0,
                complexity_category="simple",
                reason="bekanntes einfaches Wort",
            )

        suggestion_info = get_suggestion(word.lower())
        if suggestion_info:
            return WordComplexity(
                word=word,
                complexity_score=6.0,
                complexity_category="complex",
                reason=f"{suggestion_info['category']}: bekanntes schwieriges Wort",
                suggestion=suggestion_info["simple"],
                category=suggestion_info["category"],
            )

        # Use model to get complexity
        score, _ = predict_complexity(word)
        category = self._categorize_score(score)

        return WordComplexity(
            word=word,
            complexity_score=score,
            complexity_category=category,
            reason="basierend auf Sprachmodell-Analyse",
        )

    def get_violations_for_rule(
        self, doc: Doc, min_score: Optional[float] = None
    ) -> List[str]:
        """
        Get violation messages in format compatible with the rule system.

        This method is designed to be called from regel_bert_komplexitaet.py.

        Args:
            doc: spaCy Doc object
            min_score: Minimum complexity score to report

        Returns:
            List of violation messages
        """
        result = self.analyze(doc.text, doc)

        min_score = min_score or self.config.medium_threshold
        violations = []

        for word in result.complex_words:
            if word.complexity_score >= min_score:
                if word.suggestion:
                    msg = (
                        f"{word.category or 'Komplexes Wort'}: "
                        f'"{word.word}" ({word.reason}). '
                        f'Besser: "{word.suggestion}"'
                    )
                else:
                    msg = (
                        f'Komplexes Wort: "{word.word}" ({word.reason}). '
                        f"Prüfen Sie, ob ein einfacheres Wort möglich ist."
                    )
                violations.append(msg)

        return violations

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        from .models import ComplexityModel

        return ComplexityModel._loaded and ComplexityModel._model is not None

    def warmup(self) -> None:
        """Pre-load the model for faster first analysis."""
        from .models import ComplexityModel

        ComplexityModel.get_model()
        logger.info("TextComplexityAnalyzer warmup complete")


# Convenience function for simple usage
def analyze_text(text: str, doc: Optional[Doc] = None) -> ComplexityResult:
    """
    Analyze text complexity.

    Convenience function that creates an analyzer and analyzes text.

    Args:
        text: Input text
        doc: Optional spaCy Doc

    Returns:
        ComplexityResult
    """
    analyzer = TextComplexityAnalyzer()
    return analyzer.analyze(text, doc)
