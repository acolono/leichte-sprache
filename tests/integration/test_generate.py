"""Integration tests for /generate endpoint with real LLM calls.

Tests 5 German text domains x 4 LLM providers = up to 20 combinations.
Each test asserts: violation reduction, HIX improvement, faithfulness >= 3, and response schema.
"""
import pytest

from tests.integration.conftest import (
    requires_anthropic,
    requires_mistral,
    requires_ollama,
    requires_openai,
)
from tests.integration.test_analyse import (
    TestAnalyseEducation as _Education,
    TestAnalyseGovernment as _Government,
    TestAnalyseHealth as _Health,
    TestAnalyseNews as _News,
    TestAnalyseTechnical as _Technical,
)
from tools.hix import compute_hix_from_text

DOMAIN_TEXTS = [
    pytest.param("government", _Government.GOVERNMENT_TEXT, id="government"),
    pytest.param("health", _Health.HEALTH_TEXT, id="health"),
    pytest.param("education", _Education.EDUCATION_TEXT, id="education"),
    pytest.param("news", _News.NEWS_TEXT, id="news"),
    pytest.param("technical", _Technical.TECHNICAL_TEXT, id="technical"),
]

PROVIDERS = [
    pytest.param("openai", marks=requires_openai, id="openai"),
    pytest.param("anthropic", marks=requires_anthropic, id="anthropic"),
    pytest.param("mistral", marks=requires_mistral, id="mistral"),
    pytest.param("ollama", marks=requires_ollama, id="ollama"),
]


def get_initial_violations(client, text: str) -> int:
    """Get violation count for original text via /analyse endpoint."""
    resp = client.post("/analyse", json={"text": text})
    assert resp.status_code == 200
    return resp.json()["statistics"]["total_violations"]


def get_initial_hix(text: str) -> float:
    """Compute HIX score for original text using tools.hix module."""
    import spacy

    nlp = spacy.load("de_core_news_lg")
    result = compute_hix_from_text(text, nlp)
    return result.hix


def assert_generate_response_schema(data: dict):
    """Validate GenerateResponse schema fields (GENR-06)."""
    assert isinstance(data["original"], str)
    assert isinstance(data["result"], str)
    assert len(data["result"]) > 0, "result must be non-empty"
    assert isinstance(data["success"], bool)
    assert isinstance(data["iterations"], int)
    assert data["iterations"] >= 1
    assert isinstance(data["final_violations"], int)
    assert isinstance(data["iterations_detail"], list)
    assert len(data["iterations_detail"]) >= 1
    assert "provider" in data
    assert "model" in data
    assert "hix" in data
    assert "faithfulness_score" in data
    assert "stop_reason" in data


@pytest.mark.integration
@pytest.mark.llm
class TestGenerateDomains:
    """Test /generate across all domains and providers.

    Asserts three quality metrics per user decision:
    1. Violation reduction: final_violations < initial violations
    2. HIX improvement: response hix > original text hix
    3. Faithfulness: faithfulness_score >= 3 (meaning preserved)
    """

    @pytest.mark.timeout(600)
    @pytest.mark.parametrize("provider", PROVIDERS)
    @pytest.mark.parametrize("domain,text", DOMAIN_TEXTS)
    def test_generate_simplifies_text(self, client, domain, text, provider):
        """Verify /generate reduces violations, improves HIX, and preserves meaning."""
        # Get baselines from original text
        initial_violations = get_initial_violations(client, text)
        initial_hix = get_initial_hix(text)

        # Call /generate with 5 iterations per user decision
        response = client.post(
            "/generate",
            json={
                "text": text,
                "max_iterations": 5,
                "target_violations": 2,
                "provider": provider,
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Schema validation (GENR-06)
        assert_generate_response_schema(data)

        # Violation reduction (GENR-01 through GENR-05)
        assert data["final_violations"] < initial_violations, (
            f"{domain}/{provider}: violations not reduced "
            f"({data['final_violations']} >= {initial_violations})"
        )

        # HIX improvement
        assert data["hix"] is not None, f"{domain}/{provider}: hix is None"
        assert data["hix"] > initial_hix, (
            f"{domain}/{provider}: HIX not improved "
            f"({data['hix']:.1f} <= {initial_hix:.1f})"
        )

        # Faithfulness (meaning preserved)
        if data["faithfulness_score"] is not None:
            assert data["faithfulness_score"] >= 3, (
                f"{domain}/{provider}: faithfulness too low "
                f"({data['faithfulness_score']})"
            )
