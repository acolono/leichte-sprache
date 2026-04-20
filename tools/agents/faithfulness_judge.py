"""Faithfulness Judge — compares simplified text against original for meaning preservation."""

import logging
from pathlib import Path

from pydantic_ai import Agent

from tools.agents.deps import FaithfulnessAssessment

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def load_judge_prompt() -> str:
    """Load the judge prompt template from prompts/judge_prompt.txt."""
    prompt_file = _PROMPTS_DIR / "judge_prompt.txt"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Judge prompt file not found: {prompt_file}")


def create_faithfulness_judge() -> Agent:
    """Create a fresh faithfulness judge agent to avoid event loop binding issues."""
    return Agent(output_type=FaithfulnessAssessment)
