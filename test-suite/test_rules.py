"""Parametrized pytest tests for Leichte Sprache rules.

Each .txt file in test-suite/<rule_name>/ becomes a test case via
conftest.py's pytest_generate_tests hook.
"""

import importlib
from pathlib import Path
from typing import Optional, Tuple


def parse_test_file(file_path: Path) -> Tuple[Optional[str], Optional[str], str]:
    """Parse a test file to extract expected result, description, and text.

    Returns:
        Tuple of (expected, description, text)
        expected: 'flag' or 'pass'
        description: Optional description string
        text: The test text
    """
    expected = None
    description = None
    text_lines = []
    in_metadata = True

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # Parse metadata lines
            if in_metadata and stripped.startswith("#"):
                if stripped.lower().startswith("# expected:"):
                    expected = stripped.split(":", 1)[1].strip().lower()
                elif stripped.lower().startswith("# description:"):
                    description = stripped.split(":", 1)[1].strip()
            elif in_metadata and stripped == "":
                # Empty line in metadata section, continue
                continue
            else:
                # End of metadata, start collecting text
                in_metadata = False
                text_lines.append(line.rstrip())

    text = "\n".join(text_lines).strip()
    return expected, description, text


def test_rule(rule_name, test_name, test_file, nlp):
    """Run a single .txt rule test case."""
    expected, description, text = parse_test_file(test_file)
    assert expected in ("flag", "pass"), f"Invalid expected value: '{expected}'"
    assert text, "Empty test text"

    module = importlib.import_module(f"regeln.{rule_name}")
    check_rule = getattr(module, "check_rule")
    doc = nlp(text)
    violations = check_rule(doc)

    if expected == "flag":
        assert len(violations) > 0, (
            f"Expected violations for '{description or test_name}' but got none"
        )
    else:
        assert len(violations) == 0, (
            f"Expected no violations for '{description or test_name}' "
            f"but got: {violations}"
        )
