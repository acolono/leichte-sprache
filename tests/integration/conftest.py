"""Pytest configuration for integration tests.

Provides session-scoped TestClient and API key skip decorators.
"""
import os

import pytest
from fastapi.testclient import TestClient

from api_main import app


@pytest.fixture(scope="session")
def client():
    """Session-scoped TestClient. Loads FastAPI app with lifespan (model warmup)."""
    with TestClient(app) as c:
        yield c


# Reusable skip decorators for LLM-dependent tests
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)

requires_mistral = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY not set",
)


def _ollama_available() -> bool:
    """Check if Ollama server is reachable and OLLAMA_BASE_URL is set."""
    if not os.environ.get("OLLAMA_BASE_URL"):
        return False
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return resp.status_code == 200
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama server not running on localhost:11434",
)
