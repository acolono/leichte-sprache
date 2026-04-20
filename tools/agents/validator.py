"""Validator Agent — assesses quality and provides actionable feedback for the generator.

Tools: compute_hix
"""

import json
import logging

from pydantic_ai import Agent, RunContext

from tools.agents.deps import AgentDeps, ValidationResult

logger = logging.getLogger(__name__)

validator_agent = Agent(
    "test",
    deps_type=AgentDeps,
    output_type=ValidationResult,
)


@validator_agent.system_prompt
def system_prompt(ctx: RunContext[AgentDeps]) -> str:
    return ctx.deps.validator_instructions


@validator_agent.tool
def compute_hix(ctx: RunContext[AgentDeps], text: str) -> str:
    """Berechne den HIX-Score (Hohenheimer Verstaendlichkeitsindex) fuer einen Text.
    Skala 0-20: 0=Fachsprache, 18-20=Leichte Sprache.
    Nutze dies um die Gesamtverstaendlichkeit des Texts zu bewerten.

    Args:
        text: Der zu bewertende Text
    """
    from tools.hix import compute_hix_from_text

    logger.debug("Validator tool: compute_hix")
    hix_result = compute_hix_from_text(text, ctx.deps.nlp)
    return json.dumps(
        {
            "hix": hix_result.hix,
            "rating": hix_result.rating.value,
            "formula_score": hix_result.formula_score,
            "parameter_score": hix_result.parameter_score,
            "formulas": {
                "amstad": hix_result.formulas.amstad.normalized,
                "wiener": hix_result.formulas.wiener_sachtextformel.normalized,
                "smog": hix_result.formulas.smog_de.normalized,
                "lix": hix_result.formulas.lix.normalized,
            },
            "parameters": {
                "avg_sentence_length": hix_result.parameters.avg_sentence_length.normalized,
                "avg_word_length_syllables": hix_result.parameters.avg_word_length_syllables.normalized,
                "pct_long_words": hix_result.parameters.pct_long_words.normalized,
                "pct_polysyllabic": hix_result.parameters.pct_polysyllabic.normalized,
                "pct_monosyllabic": hix_result.parameters.pct_monosyllabic.normalized,
            },
        },
        ensure_ascii=False,
    )
