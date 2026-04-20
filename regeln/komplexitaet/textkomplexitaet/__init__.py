"""
TextKomplexitaet - German Text Complexity Analyzer
"""

from .analyzer import (
    ComplexityResult,
    TextComplexityAnalyzer,
    analyze_text,
)
from .config import (
    DEFAULT_CONFIG,
    LENIENT_CONFIG,
    STRICT_CONFIG,
    ComplexityConfig,
)
from .word_analyzer import (
    SentenceComplexity,
    WordComplexity,
    WordComplexityAnalyzer,
    analyze_words,
)

__all__ = [
    "TextComplexityAnalyzer",
    "ComplexityResult",
    "analyze_text",
    "ComplexityConfig",
    "DEFAULT_CONFIG",
    "STRICT_CONFIG",
    "LENIENT_CONFIG",
    "WordComplexityAnalyzer",
    "WordComplexity",
    "SentenceComplexity",
    "analyze_words",
]
