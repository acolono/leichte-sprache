"""Shared data structures for the Leichte Sprache agents."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Field


@dataclass
class AgentDeps:
    """Dependencies and mutable state for the agent run."""

    # --- Callables (bound methods from AgentOptimizer) ---
    analyze_func: Callable
    classify_violation_func: Callable
    record_failure_func: Callable
    build_specific_instruction_func: Callable
    detect_conflicts_func: Callable

    # --- spaCy model for direct Doc creation (used by compute_hix) ---
    nlp: Any

    # --- Config ---
    rule_prompts: Dict[str, Dict]
    target_violations: int
    original_text: str
    system_prompt: str

    # --- Prompt strings loaded from files ---
    generator_instructions: str = ""
    validator_instructions: str = ""

    # --- Judge model for faithfulness checking ---
    judge_model: Any = None
    last_faithfulness_score: Optional[int] = None

    # --- Mutable tracking state ---
    best_text: str = ""
    best_violations: int = 999
    stagnation_count: int = 0
    analyses_run: int = 0
    analysis_history: list = field(default_factory=list)
    persistent_violations: Dict[str, int] = field(default_factory=dict)


class GeneratedText(BaseModel):
    """Output from the generator agent."""

    text: str
    reasoning: str


class ValidationResult(BaseModel):
    """Output from the validator agent."""

    feedback: str


class LeichteSpracheResult(BaseModel):
    """Structured output the agent must produce to end the loop (kept for backward compat)."""

    text: str
    reasoning: str
    violations: int


class FaithfulnessIssue(BaseModel):
    """A single semantic faithfulness issue found by the judge."""

    category: (
        str  # "missing_info" | "distortion" | "hallucination" | "oversimplification"
    )
    description: str
    original_passage: str
    simplified_passage: str


class FaithfulnessAssessment(BaseModel):
    """Structured output from the faithfulness judge."""

    score: int = Field(..., ge=1, le=5, description="1-5, where 5 = perfectly faithful")
    summary: str
    issues: list[FaithfulnessIssue]
    recommendation: str
