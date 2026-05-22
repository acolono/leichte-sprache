"""Restructurer Agent — Stage A of the two-stage /generate pipeline.

Single LLM call that takes the full original text and produces a
fully restructured Leichte Sprache version. Has the explicit freedom
(via its system prompt) to split sentences, drop background info,
reorder, introduce lists, and change perspective — the kinds of
structural moves the sentence-scoped refiner (Stage B) cannot make.

No tools. No internal iteration. The orchestrator calls `run_sync` once
per Stage A invocation and either commits the result or, if the
faithfulness judge later rejects it, re-invokes with feedback.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


class RestructuredText(BaseModel):
    """Stage A's structured output."""

    text: str = Field(
        ...,
        description=(
            "Der vollstaendig in Leichte Sprache umformulierte Text. "
            "Endnutzer sehen nur dieses Feld."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Kurze Debug-Notiz, was strukturell veraendert wurde "
            "(z. B. Saetze geteilt, Anrede eingefuehrt, Liste erstellt). "
            "Nicht fuer den Endnutzer gedacht."
        ),
    )


def _load_system_prompt() -> str:
    """Load the Stage A system prompt from prompts/restructurer_system.md.

    Falls back to a minimal inline prompt if the file is missing so the
    container doesn't refuse to boot during a partial install.
    """
    prompt_path = (
        Path(__file__).parent.parent.parent / "prompts" / "restructurer_system.md"
    )
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    logger.warning(
        "Restructurer system prompt not found at %s — using minimal fallback",
        prompt_path,
    )
    return (
        "Du bist Experte fuer Leichte Sprache. Formuliere den folgenden Text "
        "in Leichte Sprache um. Du darfst Saetze teilen, Reihenfolge aendern, "
        "Hintergrundinformation weglassen und Listen einfuehren. Bewahre "
        "konkrete Fakten (Datum, Zahlen, Namen, Gesetze)."
    )


# Module-level singleton — the Agent stores prompts but the model is bound
# per-invocation by the orchestrator via run_sync(..., model=...).
restructurer_agent: Agent[None, RestructuredText] = Agent(
    "test",
    output_type=RestructuredText,
)


@restructurer_agent.system_prompt
def _system_prompt() -> str:
    return _load_system_prompt()


def build_restructurer_prompt(
    original_text: str,
    faithfulness_feedback: str | None = None,
) -> str:
    """Build the user message for a Stage A invocation.

    On the first call ``faithfulness_feedback`` is None. On a faithfulness
    retry the orchestrator passes the structured judge feedback so the
    next attempt knows which passages must be preserved.
    """
    if not faithfulness_feedback:
        return (
            "Formuliere den folgenden Text in Leichte Sprache um.\n\n"
            "<ORIGINAL>\n"
            f"{original_text}\n"
            "</ORIGINAL>\n"
        )
    return (
        "Dein vorheriger Versuch hat wichtige Inhalte verloren oder verzerrt. "
        "Formuliere den Original-Text erneut in Leichte Sprache um und bewahre "
        "die unten genannten Passagen.\n\n"
        "<ORIGINAL>\n"
        f"{original_text}\n"
        "</ORIGINAL>\n\n"
        "<TREUE-FEEDBACK>\n"
        f"{faithfulness_feedback}\n"
        "</TREUE-FEEDBACK>\n"
    )
