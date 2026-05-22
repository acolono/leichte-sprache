"""Sentence Refiner Agent — Stage B of the two-stage /generate pipeline.

Each invocation takes a single sentence (the target) plus its immediate
neighbours and a structured list of rule violations + fix suggestions, and
returns a refined version of just the target sentence. The orchestrator
in `tools/agent_optimizer.py` calls this agent at most once per problem
sentence per iteration; the agent itself does not iterate — there are no
tools, no internal analyze loop, no exploration.

Compared to the earlier multi-tool generator agent, the LLM no longer
decides when to stop: the deterministic analyzer plus the orchestrator's
plateau/attempt caps do that. Removing the tools collapses the inner
loop from ``UsageLimits(request_limit=10)`` to effectively one call plus
one retry budget — the dominant source of cost blowup in production.

Stage A (full-text restructuring) is in `tools/agents/restructurer.py`.
Stage C (faithfulness verification) lives in `tools/agents/faithfulness_judge.py`.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent

logger = logging.getLogger(__name__)


# Rule importance weights, lifted from the previous generator implementation
# so prompt sorting still matches. The orchestrator imports these when it
# builds per-sentence fix lists.
RULE_WEIGHTS = {
    "nebensaetze": 0,
    "satzlaenge": 0,
    "passiv_erkennung": 0,
    "fremdwoerter": 0,
    "perplexity_saetze": 0,
    "komposita": 10,
    "abkuerzungen": 10,
    "genitiv": 10,
    "konjunktiv": 10,
    "negationen": 10,
    "komplexitaet": 10,
    "kurze_woerter": 20,
    "redewendungen": 20,
    "zahlwoerter": 20,
    "interpunktion": 20,
    "personalpronomen": 20,
    "synonyme": 20,
    "mehrere_aussagen": 20,
}
DEFAULT_RULE_WEIGHT = 15


class RefinedSentence(BaseModel):
    """Stage B's structured output.

    The candidate is a list because ``mehrere_aussagen`` violations are
    often fixed by splitting one sentence into several. A pure one-to-one
    rewrite is the common case — produce a single-element list.
    """

    sentences: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Der umformulierte Satz. Bei einem 'mehrere_aussagen'-Verstoss "
            "darfst du ihn in mehrere kurze Saetze aufspalten — dann mehr "
            "als ein Listen-Eintrag."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Kurze Debug-Notiz, was geaendert wurde. Nicht fuer den Endnutzer."
        ),
    )


_SYSTEM_PROMPT = """\
Du bist Experte fuer Leichte Sprache nach DIN SPEC 33429. Deine Aufgabe ist
es, **einen einzelnen Satz** so umzuformulieren, dass er die genannten
Verstoesse behebt.

Wichtige Regeln fuer diese Aufgabe:

1. **Nur den Zielsatz aendern.** Der Kontext (Saetze davor/danach) ist
   nur zur Information da, damit du Pronomen und Bezuege erhalten kannst.
   Aendere den Kontext nicht.

2. **Kernbedeutung erhalten.** Drop keine Fakten — Daten, Zahlen, Namen,
   Betraege, Termine, Gesetze muessen erhalten bleiben. Wenn der Zielsatz
   ein Verwaltungsdetail enthaelt, das ohne Bedeutungsverlust weggelassen
   werden kann, darfst du es weglassen.

3. **Behebe genau die genannten Verstoesse.** Erfinde keine zusaetzlichen
   Probleme; deine Aenderungen muessen sich auf die uebergebene Verstoss-
   Liste beziehen.

4. **Bei `mehrere_aussagen`** darfst (und sollst) du den Zielsatz in
   mehrere kurze Saetze aufteilen. Gib dann mehrere Listen-Eintraege
   zurueck.

5. **Keine Platzhalter, keine Kommentare, keine Ellipsen.** Niemals "[...]",
   "unveraendert", "Rest bleibt gleich" o. ae. ausgeben. Wenn du den Satz
   nicht verbessern kannst, gib ihn unveraendert zurueck — kein Platzhalter.

6. **Stilregeln Leichte Sprache** (Pflicht in der Ausgabe):
   - max. ca. 10 Woerter pro Satz
   - eine Aussage pro Satz
   - aktiv statt passiv
   - keine Genitive, keine Konjunktive, wenn vermeidbar
   - lange Komposita aufteilen ("Bundes-Regierung") oder ersetzen
   - Zahlen als Ziffern

Gib **ausschliesslich** die strukturierte Ausgabe zurueck — eine Liste
mit dem umformulierten Satz (oder mehreren Saetzen bei Aufspaltung) plus
optional eine kurze Debug-Notiz.
"""


sentence_refiner_agent: Agent[None, RefinedSentence] = Agent(
    "test",
    output_type=RefinedSentence,
)


@sentence_refiner_agent.system_prompt
def _system_prompt() -> str:
    return _SYSTEM_PROMPT


def build_refiner_prompt(
    target_sentence: str,
    before_context: str,
    after_context: str,
    rules: List[dict],
    fixes: List[dict],
) -> str:
    """Compose the user message for one Stage B call.

    The orchestrator pre-computes:

    * ``rules``  — one entry per rule whose violations land in the target
      sentence: ``{"rule_id", "title", "anweisung", "falsch", "richtig"}``
      pulled from the existing per-rule prompt.md files via
      ``deps.rule_prompts``.
    * ``fixes`` — concrete per-violation instructions built by
      ``AgentOptimizer._build_specific_instruction``, sorted by rule weight.

    Both go straight into the prompt — the LLM does not call any tool to
    fetch them.
    """
    parts = ["<KONTEXT>"]
    if before_context.strip():
        parts.append("Davor:")
        parts.append(before_context.rstrip())
    parts.append("Zielsatz (NUR DIESEN aendern):")
    parts.append(target_sentence)
    if after_context.strip():
        parts.append("Danach:")
        parts.append(after_context.lstrip())
    parts.append("</KONTEXT>\n")

    if fixes:
        parts.append("<KONKRETE-FIXES>")
        for i, fix in enumerate(fixes, 1):
            rule = fix.get("rule_id", "")
            instr = fix.get("instruction", "")
            parts.append(f"{i}. [{rule}] {instr}")
        parts.append("</KONKRETE-FIXES>\n")

    if rules:
        parts.append("<REGEL-HINWEISE>")
        for rule in rules:
            rid = rule.get("rule_id", "")
            anweisung = rule.get("anweisung") or rule.get("regel") or ""
            falsch = rule.get("falsch") or []
            richtig = rule.get("richtig") or []
            parts.append(f"## {rid}")
            if anweisung:
                parts.append(anweisung.strip()[:400])
            if falsch:
                parts.append("Falsch: " + " | ".join(str(x)[:80] for x in falsch[:2]))
            if richtig:
                parts.append("Richtig: " + " | ".join(str(x)[:80] for x in richtig[:2]))
        parts.append("</REGEL-HINWEISE>\n")

    parts.append(
        "Gib den umformulierten Zielsatz zurueck. Behebe die genannten "
        "Verstoesse, bewahre die Kernaussage, aendere nur den Zielsatz."
    )
    return "\n".join(parts)


__all__ = [
    "RefinedSentence",
    "sentence_refiner_agent",
    "build_refiner_prompt",
    "RULE_WEIGHTS",
    "DEFAULT_RULE_WEIGHT",
]
