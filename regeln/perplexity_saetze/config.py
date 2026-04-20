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
