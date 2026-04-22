"""Configuration for the synonyms rule."""

# Similarity thresholds for synonym detection
SIMILARITY_HIGH = 0.75
SIMILARITY_MEDIUM = 0.65
SIMILARITY_LOW = 0.55

# Preference for shorter words
PREFER_SHORTER_WORDS = True

# OpenThesaurus-backed synset check (LGPL/CC-BY-SA).
# Default False — raw OpenThesaurus treats e.g. 'Haus/Familie' as synonyms
# (Habsburg-house reading) and produces too many false positives on ordinary
# German text. Enable only after curating a domain-specific overlay.
USE_OPENTHESAURUS = False
