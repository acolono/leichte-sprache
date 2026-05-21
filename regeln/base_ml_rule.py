"""
Abstract base class for ML-based rules with thread-safe lazy loading.

Each subclass automatically receives its own _lock, _model, _loaded, _error
class attributes via __init_subclass__. This prevents different rules from
sharing the same lock or loading state.

Usage:
    from regeln.base_ml_rule import BaseMLRule

    class MyModel(BaseMLRule):
        @classmethod
        def _load_model(cls):
            # Load and return the model
            return pipeline(...)

    # In check_rule():
    model = MyModel.get_model()
    if model is None:
        error = MyModel.get_error() or "Model unavailable"
        return [f"Model not available: {error}"]
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseMLRule(ABC):
    """Base class for ML-based rules with thread-safe lazy model loading.

    Each subclass gets its OWN _lock, _model, _loaded, _error class attributes
    via __init_subclass__. Subclasses must implement _load_model() -> Any.
    """

    _lock: threading.Lock
    _model: Any = None
    _loaded: bool = False
    _error: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """Give each subclass its own lock and state."""
        super().__init_subclass__(**kwargs)
        cls._lock = threading.Lock()
        cls._model = None
        cls._loaded = False
        cls._error = None

    @classmethod
    @abstractmethod
    def _load_model(cls) -> Any:
        """Load the ML model. Called once, under lock.

        Returns:
            The loaded model (pipeline, tuple, etc.)

        Raises:
            Any exception -- will be caught and logged by get_model().
        """
        ...

    @classmethod
    def get_model(cls) -> Optional[Any]:
        """Get the model, loading lazily with thread safety.

        Uses double-checked locking:
        1. Fast path: if already loaded, return immediately (no lock)
        2. Slow path: acquire lock, check again, load if needed

        Returns:
            The model object, or None if loading failed.
        """
        if cls._loaded:
            return cls._model

        with cls._lock:
            # Double-check after acquiring lock
            if cls._loaded:
                return cls._model

            try:
                cls._model = cls._load_model()
            except Exception as e:
                cls._error = f"{cls.__name__}: {type(e).__name__}: {e}"
                # logger.exception emits the full traceback at WARNING level —
                # opaque "stat: path … NoneType" wrappers are useless without it.
                logger.exception(
                    "Failed to load model for %s", cls.__name__
                )
                cls._model = None
            finally:
                cls._loaded = True

        return cls._model

    @classmethod
    def get_error(cls) -> Optional[str]:
        """Return the error message if model loading failed."""
        return cls._error
