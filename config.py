"""
Central configuration for the Leichte Sprache API.

Values can be overridden via environment variables (prefixed with LS_).
"""

import logging
import os

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Default log level for the /generate endpoint.
# Callers can still override per-request via the "debug" field.
# Valid values: "DEBUG", "INFO", "WARN"
DEFAULT_LOG_LEVEL: str = os.environ.get("LS_DEFAULT_LOG_LEVEL", "INFO")

logging.basicConfig(
    level=getattr(logging, DEFAULT_LOG_LEVEL),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Generator defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS: int = int(os.environ.get("LS_MAX_ITERATIONS", "10"))
DEFAULT_TARGET_VIOLATIONS: int = int(os.environ.get("LS_TARGET_VIOLATIONS", "2"))
