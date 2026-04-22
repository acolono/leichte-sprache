"""
Configuration for the sentence structure complexity rule.

These thresholds are based on Leichte Sprache (Easy Language) guidelines.
"""

# --- SENTENCE LENGTH ---
# Maximum words per sentence (soft limit - only in combination with other factors)
MAX_WORDS_PER_SENTENCE = 12

# Hard limit - sentences above this are always too long
MAX_WORDS_HARD_LIMIT = 18


# --- SUBORDINATE CLAUSES ---
# Above this number of subordinate clauses, the sentence is marked as complex
MAX_SUBORDINATE_CLAUSES = 1


# --- SENTENCE STRUCTURE ---
# Maximum depth of the dependency tree
MAX_DEPENDENCY_DEPTH = 5

# Maximum distance between subject and verb (in tokens)
MAX_SUBJECT_VERB_DISTANCE = 6


# --- FLAGGING THRESHOLD ---
# Minimum number of issues for a sentence to be flagged as complex
# 1 = sensitive (many reports)
# 2 = balanced (recommended)
# 3 = strict (only severe cases)
MIN_ISSUES_FOR_FLAG = 1


# --- PERPLEXITY SIGNAL (supplementary, info-level) ---
# An NLTK Kneser-Ney LM was trained on the Leichte-Sprache corpus, but the
# full .perplexity() path takes ~1 s/sentence. Calibration showed that 98%+
# of the off-register signal is captured by OOV ratio alone (tokens absent
# from the LS-corpus vocabulary), so we use the model's vocab directly as a
# lightweight O(tokens) proxy. When a sentence is already caught by the
# structural check (nebensätze / length / passive / SVD / depth), the signal
# is suppressed to avoid 4x redundant flags.
ENABLE_PERPLEXITY_SIGNAL = True

# Flag sentences where at least this fraction of alphabetic content tokens
# are NOT in the LS-corpus vocabulary. Calibrated against a 200-sentence
# sample of data/leichte_sprache_korpus.txt (0% OOV by construction) vs
# tools/eval_corpus/*.txt (20-60% OOV for complex domain text). See
# CALIBRATION.md.
THRESHOLD_OOV_RATIO = 0.15

# Minimum alphabetic tokens required before the OOV ratio is considered stable.
MIN_TOKENS_FOR_PERPLEXITY = 3

# Prefix used on the emitted info message. Downstream consumers can filter by
# this marker to route the signal differently from the DIN-SPEC-aligned warnings.
PERPLEXITY_INFO_PREFIX = "[Hinweis Perplexität]"
