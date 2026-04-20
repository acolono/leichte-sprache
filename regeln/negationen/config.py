"""Configuration for the negation rule."""

# Detect different negation types
DETECT_NICHT = True  # "nicht"
DETECT_KEIN = True  # "kein", "keine"
DETECT_OHNE = True  # "ohne"
DETECT_WEDER_NOCH = True  # "weder...noch"

# Specially mark double negations
WARN_DOUBLE_NEGATION = True
