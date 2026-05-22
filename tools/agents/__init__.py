"""Leichte Sprache agents package."""

from .deps import AgentDeps, FaithfulnessAssessment, GeneratedText, ValidationResult
from .faithfulness_judge import create_faithfulness_judge, load_judge_prompt
from .generator import (
    RefinedSentence,
    build_refiner_prompt,
    sentence_refiner_agent,
)
from .restructurer import (
    RestructuredText,
    build_restructurer_prompt,
    restructurer_agent,
)

__all__ = [
    "AgentDeps",
    "FaithfulnessAssessment",
    "GeneratedText",
    "ValidationResult",
    "RefinedSentence",
    "RestructuredText",
    "sentence_refiner_agent",
    "restructurer_agent",
    "build_refiner_prompt",
    "build_restructurer_prompt",
    "create_faithfulness_judge",
    "load_judge_prompt",
]
