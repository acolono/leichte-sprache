"""
Configuration for the BERT-based number words rule.

This rule uses a fine-tuned BERT token-classification model
for precise detection of written-out numbers in German texts.
"""

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Hugging Face Model ID - UPDATE after upload!
# Format: "username/model-name"
# Example: "felix-ml/leichte-sprache-zahlwoerter"
HUGGINGFACE_MODEL_ID = "fefeefef/leichte-sprache-zahlwoerter"

# Confidence Threshold: minimum confidence for error detection (0.0 - 1.0)
# - 0.5: Very sensitive (many detections, possible false positives)
# - 0.6: Balanced (RECOMMENDED)
# - 0.8: Conservative (only very confident detections)
CONFIDENCE_THRESHOLD = 0.6

# Fallback: local model paths (for development without Hugging Face upload)
# The system tries these paths if Hugging Face download fails
LOCAL_MODEL_PATHS = [
    "/Users/felix/Documents/projects/gemini_automation/leichte_sprache_model_final",
    "./leichte_sprache_model_final",
    "../leichte_sprache_model_final",
]

# ============================================================================
# LABEL DESCRIPTIONS
# ============================================================================

# Descriptions for the 4 recognized error types
# These are used in error messages (values are user-facing German strings)
LABEL_DESCRIPTIONS = {
    "BAD_WORD_NUM": "Zahl als Wort geschrieben",
    "BAD_YEAR": "Jahreszahl als Wort geschrieben",
    "BAD_PERCENT": "Prozentangabe als Wort geschrieben",
    "BAD_COMPLEX_NUM": "Komplexe Zahl als Wort geschrieben",
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Show detailed position information in error messages
SHOW_POSITION = True

# Error message format
# Available placeholders: {word}, {label}, {description}, {start}, {end}
ERROR_MESSAGE_TEMPLATE = 'Zahlwort "{word}" sollte als Ziffer geschrieben werden. Text-Position: {start}-{end}'

# Alternative templates (commented out):
# ERROR_MESSAGE_TEMPLATE = 'Zahlwort "{word}" ({description}). Text-Position: {start}-{end}'
# ERROR_MESSAGE_TEMPLATE = '{description}: "{word}" (Position {start}-{end})'

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Device priority for model loading
# "auto" = automatic selection (MPS > CUDA > CPU)
# "mps" = Apple Silicon (M1/M2/M3)
# "cuda" = Nvidia GPU
# "cpu" = CPU (slow, but always available)
DEVICE_PREFERENCE = "auto"

# Cache directory for Hugging Face downloads
# None = default (~/.cache/huggingface/)
CACHE_DIR = None

# Maximum sequence length for tokenization
MAX_SEQUENCE_LENGTH = 512

# Batch processing (for future optimizations)
BATCH_SIZE = 1

# ============================================================================
# HYBRID FILTER CONFIGURATION
# ============================================================================

# Enable rule-based filter (pre-processing for number words 0-50)
# Advantages: fast (~1ms), deterministic, reliable for unambiguous cases
ENABLE_RULE_BASED_FILTER = True

# Enable BERT model (for complex numbers, >50, year numbers, context)
# Advantages: context-aware, distinguishes "ein" as article vs. number
ENABLE_BERT_MODEL = True

# Merge strategy for duplicates (when both filters detect the same word)
# True = prefer BERT result (more accurate position, higher confidence)
# False = prefer rule result (faster, deterministic)
PREFER_BERT_ON_CONFLICT = True

# Exceptions for rule-based filter (ambiguous number words)
# These words are skipped by the rule-based filter and
# only handled by the BERT model (if contextually unambiguous)
NUMBER_WORDS_0_50_EXCEPTIONS = [
    "elf",  # Ambiguous: soccer team (11 players), fantasy creatures (elves)
    "acht",  # Ambiguous: idioms ("Acht geben", "sich in Acht nehmen")
]

# Note: "ein/eine/einer" is handled entirely by the BERT model
# (context-based distinction: article vs. number word)

# ============================================================================
# POST-PROCESSING FILTER CONFIGURATION
# ============================================================================

# Ghost artifacts blacklist: tokenizer fragments that are NOT real number words
# These suffixes are often falsely detected by the model (e.g. "50-jaehriges" -> "iges")
# and must be filtered out
GHOST_ARTIFACTS = {"ige", "iges", "sten", "ten", "tel", "ig", "lich", "fach"}

# Minimum word length for detected number words
# Single characters are usually noise (except digits)
MIN_WORD_LENGTH = 2

# Enable regex filter for Roman numerals
# Detects patterns like: I, II, III, IV, V, VI, X, XIV, MCMXCV etc.
ENABLE_REGEX_FILTER = True

# Regex patterns for Roman numerals
# Patterns detect Roman numerals in various contexts
import re  # noqa: E402

REGEX_RULES = [
    {
        "label": "REGEX_ROMAN",
        "desc": "Römische Zahl (schwer verständlich)",
        # Pattern for Roman numerals 1-39 (I-XXXIX):
        # X{1,3}: 10, 20, 30
        # IX: 9
        # IV: 4
        # V?I{1,3}: I, II, III (without V) or VI, VII, VIII (with V)
        # With optional trailing period
        "pattern": re.compile(r"\b(X{1,3}|IX|IV|V?I{1,3})\.?\b"),
    },
    {
        "label": "REGEX_NUM_50_100",
        "desc": "Zahl 50-100 als Wort geschrieben",
        # Pattern for German number words 50-100:
        # - Base tens: fuenfzig, sechzig, siebzig, achtzig, neunzig
        # - Compound: einundfuenfzig, zweiundfuenfzig, ..., neunundfuenfzig
        # - Hundred: hundert, einhundert
        # Case-insensitive for flexibility
        "pattern": re.compile(
            r"\b("
            # Compound numbers (51-99)
            r"(ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun)und(fünfzig|sechzig|siebzig|achtzig|neunzig)|"
            # Base tens (50, 60, 70, 80, 90)
            r"fünfzig|sechzig|siebzig|achtzig|neunzig|"
            # Hundred (with or without "ein")
            r"(ein)?hundert"
            r")\b",
            re.IGNORECASE,
        ),
    },
]

# Extended label descriptions (including regex labels)
# Values are user-facing German strings
LABEL_DESCRIPTIONS_EXTENDED = {
    "BAD_WORD_NUM": "Zahl als Wort geschrieben",
    "BAD_YEAR": "Jahreszahl schwer lesbar",
    "BAD_PERCENT": "Prozentzeichen vermeiden",
    "BAD_COMPLEX_NUM": "Komplexe Zahl / Ziffernfolge",
    "REGEX_ROMAN": "Römische Zahl (schwer verständlich)",
    "REGEX_NUM_50_100": "Zahl 50-100 als Wort geschrieben",
}
