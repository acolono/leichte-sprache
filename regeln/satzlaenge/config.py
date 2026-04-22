"""Configuration for the satzlaenge rule.

Thresholds target DIN SPEC 33429:2025 §5.4 sentence-length recommendation
of 8-10 words per sentence. The mean threshold of 10 replaces a historical
12-word mean (which was more permissive than the 2025 standard).
"""

# Per-sentence word-count thresholds (adaptive by syntactic complexity)
MAX_WORDS_DEFAULT = 10  # applied when complexity_score <= 3.0
MAX_WORDS_MEDIUM = 9    # applied when 3.0 < complexity_score <= 5.0
MAX_WORDS_HIGH = 8      # applied when complexity_score > 5.0

# Hard limit for per-sentence word count regardless of complexity
MAX_WORDS_HARD = 15

# Mean sentence length across the document (DIN SPEC 33429 target: 8-10)
MAX_MEAN_WORDS = 10

# Complexity-only trigger: flag even short sentences if syntactically very complex
COMPLEXITY_TRIGGER = 6.0
