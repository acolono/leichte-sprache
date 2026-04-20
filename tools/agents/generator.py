"""Generator Agent — produces and improves Leichte Sprache text.

Tools: analyze_text, get_rule_guidance, get_fix_suggestions
"""

import json
import logging
from typing import Any, Dict, List, Tuple

from pydantic_ai import Agent, RunContext

from tools.agents.deps import AgentDeps, GeneratedText

logger = logging.getLogger(__name__)

# Rule importance weights (imported here for get_fix_suggestions sorting)
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

generator_agent = Agent(
    "test",
    deps_type=AgentDeps,
    output_type=GeneratedText,
)


@generator_agent.system_prompt
def system_prompt(ctx: RunContext[AgentDeps]) -> str:
    instructions = ctx.deps.generator_instructions.format(
        target_violations=ctx.deps.target_violations,
    )
    return ctx.deps.system_prompt + "\n" + instructions


@generator_agent.tool
def analyze_text(ctx: RunContext[AgentDeps], text: str) -> str:
    """Analysiere Text auf Leichte-Sprache-Verstoesse.
    Gibt zurueck: Anzahl Verstoesse, Ziel, Verbesserung, Details pro Verstoss.
    MUSS nach jeder Textaenderung aufgerufen werden.

    Args:
        text: Der zu analysierende Text in Leichter Sprache
    """
    deps = ctx.deps
    analysis = deps.analyze_func(text)
    issues = analysis.get("issues", [])
    stats = analysis.get("statistics", {})

    violation_count = stats.get("total_violations", 0)
    prev_violations = deps.best_violations if deps.analyses_run > 0 else None

    # Track best text & stagnation
    deps.analyses_run += 1
    if violation_count < deps.best_violations:
        deps.best_text = text
        deps.best_violations = violation_count
        deps.stagnation_count = 0
    else:
        deps.stagnation_count += 1

    # Track persistent violations
    for issue in issues:
        rule_id = issue.get("rule_id", "").replace("_issue", "")
        deps.persistent_violations[rule_id] = (
            deps.persistent_violations.get(rule_id, 0) + 1
        )

    logger.info(
        "Analysis %d: violations=%d (best=%d, target=%d), stagnation=%d",
        deps.analyses_run,
        violation_count,
        deps.best_violations,
        deps.target_violations,
        deps.stagnation_count,
    )

    # Build enriched result (capped at 15 issues)
    enriched_issues = [
        {
            "rule_id": iss.get("rule_id", "").replace("_issue", ""),
            "text": iss.get("text", ""),
            "message": iss.get("message", ""),
            "fixability": deps.classify_violation_func(iss),
        }
        for iss in issues[:15]
    ]

    result: Dict[str, Any] = {
        "total_violations": violation_count,
        "target": deps.target_violations,
        "violations_by_rule": stats.get("violations_by_rule", {}),
        "issues": enriched_issues,
    }

    if prev_violations is not None:
        delta = prev_violations - violation_count
        if delta > 0:
            result["improvement"] = f"+{delta} weniger Verstoesse"
        elif delta == 0:
            result["improvement"] = "keine Veraenderung"
        else:
            result["improvement"] = f"{-delta} mehr Verstoesse (Verschlechterung)"

    # Stagnation warning
    if deps.stagnation_count >= 3:
        logger.warning(
            "Agent stagnation: no improvement for %d analyses",
            deps.stagnation_count,
        )
        result["warning"] = (
            f"Keine Verbesserung seit {deps.stagnation_count} Analysen. "
            "Akzeptiere verbleibende Verstoesse und gib dein Endergebnis zurueck."
        )

    return json.dumps(result, ensure_ascii=False)


@generator_agent.tool
def get_rule_guidance(ctx: RunContext[AgentDeps], rule_id: str) -> str:
    """Hole Anweisungen, Falsch/Richtig-Beispiele und Muster fuer eine Regel.
    Nutze dies wenn du nicht weisst wie du einen bestimmten Verstoss beheben kannst.

    Args:
        rule_id: Regel-ID, z.B. 'passiv_erkennung', 'nebensaetze', 'fremdwoerter'
    """
    clean_id = rule_id.replace("_issue", "")
    logger.debug("Agent tool: get_rule_guidance — rule_id=%s", clean_id)
    rule_data = ctx.deps.rule_prompts.get(clean_id)
    if not rule_data:
        return json.dumps({"error": f"Regel '{clean_id}' nicht gefunden"})
    return json.dumps(
        {
            "rule_id": clean_id,
            "title": rule_data.get("title", ""),
            "regel": rule_data.get("regel", ""),
            "anweisung": rule_data.get("anweisung", ""),
            "muster": rule_data.get("muster", []),
            "falsch": rule_data.get("falsch", []),
            "richtig": rule_data.get("richtig", []),
        },
        ensure_ascii=False,
    )


@generator_agent.tool
def get_fix_suggestions(ctx: RunContext[AgentDeps], text: str) -> str:
    """Analysiere Text und liefere konkrete, priorisierte Aenderungsvorschlaege.
    Teurer als analyze_text (macht Analyse + baut Anweisungen).
    Nutze dies wenn du gezielte Anweisungen fuer verbleibende Verstoesse brauchst.

    Args:
        text: Der zu analysierende Text
    """
    deps = ctx.deps
    logger.debug("Agent tool: get_fix_suggestions")

    analysis = deps.analyze_func(text)
    issues = analysis.get("issues", [])

    if not issues:
        return json.dumps({"suggestions": [], "message": "Keine Verstoesse gefunden."})

    sorted_issues = sorted(
        issues,
        key=lambda iss: RULE_WEIGHTS.get(
            iss.get("rule_id", "").replace("_issue", ""), DEFAULT_RULE_WEIGHT
        ),
    )

    suggestions: List[Dict[str, str]] = []
    for iss in sorted_issues[:10]:
        rule_id = iss.get("rule_id", "").replace("_issue", "")
        message = iss.get("message", "")
        problematic_text = iss.get("text", "")

        instruction = deps.build_specific_instruction_func(
            rule_id, message, problematic_text, text
        )
        if instruction:
            suggestions.append(
                {
                    "rule_id": rule_id,
                    "problematic_text": problematic_text,
                    "instruction": instruction,
                }
            )

    conflicts_raw: List[Tuple[Dict, Dict]] = deps.detect_conflicts_func(issues)
    conflicts = [
        {
            "rule_1": c[0].get("rule_id", "").replace("_issue", ""),
            "rule_2": c[1].get("rule_id", "").replace("_issue", ""),
            "text": c[0].get("text", ""),
            "hint": "Diese Verstoesse ueberlappen — behebe einen nach dem anderen.",
        }
        for c in conflicts_raw[:5]
    ]

    result: Dict[str, Any] = {"suggestions": suggestions}
    if conflicts:
        result["conflicts"] = conflicts

    return json.dumps(result, ensure_ascii=False)
