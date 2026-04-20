"""Leichte Sprache agents package."""

from .deps import AgentDeps, FaithfulnessAssessment, GeneratedText, ValidationResult
from .faithfulness_judge import create_faithfulness_judge, load_judge_prompt
from .generator import generator_agent
from .validator import validator_agent

__all__ = [
    "AgentDeps",
    "FaithfulnessAssessment",
    "GeneratedText",
    "ValidationResult",
    "generator_agent",
    "validator_agent",
    "create_faithfulness_judge",
    "load_judge_prompt",
]
