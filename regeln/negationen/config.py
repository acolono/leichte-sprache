"""Configuration for the negation rule."""

# Detect different negation types
DETECT_NICHT = True  # "nicht"
DETECT_KEIN = True  # "kein", "keine"
DETECT_OHNE = True  # "ohne"
DETECT_WEDER_NOCH = True  # "weder...noch"

# Specially mark double negations
WARN_DOUBLE_NEGATION = True

# Prefix-based negations (un-, miss-, -los, -frei) close a DIN SPEC 33429
# §5.4 gap that the original morphology-only detector left open.
# Default True (gap closure). Set False to match the old behavior.
DETECT_PREFIX_NEGATION = True

# Words that LOOK like un-/miss- prefixes but are not negations in German.
# These are lexicalized words whose un-/miss- is part of the stem, not a
# negation prefix. Matches are by lowercase lemma.
PREFIX_NEGATION_STOPLIST = {
    # un- as stem, not prefix
    "unfall", "unruhe", "union", "unikum", "universum", "unschuld",
    "unterricht", "unternehmen", "unter", "unten", "universität", "uniform",
    "unbill", "unke", "unze",
    # miss- as stem, not prefix (Missionar/Mission borrow Latin mitt-)
    "mission", "missionar", "mister", "mistel", "mist",
}
