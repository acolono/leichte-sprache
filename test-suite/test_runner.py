#!/usr/bin/env python3
"""
Test runner for Leichte Sprache rules.

Validates rules against test data files containing test cases with expected results.

Usage:
    python test-suite/test_runner.py                # Run all tests
    python test-suite/test_runner.py abkuerzungen   # Test specific rule
    python test-suite/test_runner.py -v             # Verbose output
    python test-suite/test_runner.py --list         # List available rules

Test File Format (in test-suite/<rule_name>/*.txt):
    # expected: flag|pass
    # description: Optional description

    Text to test goes here.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import spacy

# Paths
import logging

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent
TEST_SUITE_DIR = Path(__file__).parent
REGELN_DIR = PROJECT_ROOT / "regeln"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))


def get_available_rules() -> List[str]:
    """Get list of available rule modules."""
    rules = []
    for item in REGELN_DIR.iterdir():
        if item.is_dir() and (item / "regel.py").exists():
            rules.append(item.name)
    return sorted(rules)


def get_test_data_dir(rule_name: str) -> Path:
    """Get test data directory for a rule."""
    return TEST_SUITE_DIR / rule_name


def load_rule(rule_name: str):
    """Load a rule module and return its check_rule function."""
    try:
        module = importlib.import_module(f"regeln.{rule_name}")
        func = getattr(module, "check_rule", None)
        if func is None:
            logger.info("Module 'regeln.%s' loaded but has no 'check_rule' function", rule_name)
            logger.info(" Available attributes: %s", dir(module))
        return func
    except Exception as e:
        import traceback

        logger.error("Error loading rule '%s': %s", rule_name, e)
        traceback.print_exc()
        return None


def parse_test_file(file_path: Path) -> Tuple[Optional[str], Optional[str], str]:
    """
    Parse a test file to extract expected result, description, and text.

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


def run_single_test(
    rule_name: str, check_rule, nlp, test_file: Path, verbose: bool = False
) -> Tuple[bool, str]:
    """
    Run a single test case.

    Returns:
        Tuple of (passed, message)
    """
    expected, description, text = parse_test_file(test_file)

    if expected not in ("flag", "pass"):
        return False, f"Invalid expected value: '{expected}' (must be 'flag' or 'pass')"

    if not text:
        return False, "Empty test text"

    # Run the rule
    doc = nlp(text)
    try:
        violations = check_rule(doc)
    except Exception as e:
        return False, f"Rule error: {e}"

    has_violations = len(violations) > 0

    # Check result
    if expected == "flag" and has_violations:
        passed = True
        status = "PASS"
    elif expected == "pass" and not has_violations:
        passed = True
        status = "PASS"
    else:
        passed = False
        status = "FAIL"

    # Build message
    desc = f" ({description})" if description else ""
    if verbose:
        result_detail = (
            f"got {len(violations)} violations" if has_violations else "no violations"
        )
        msg = f"[{status}] {test_file.name}{desc}\n       Expected: {expected}, Result: {result_detail}"
        if not passed and violations:
            msg += f"\n       Violations: {violations[:3]}{'...' if len(violations) > 3 else ''}"
    else:
        msg = f"[{status}] {test_file.name}{desc}"

    return passed, msg


def run_tests_for_rule(
    rule_name: str, nlp, verbose: bool = False
) -> Tuple[int, int, List[str]]:
    """
    Run all tests for a specific rule.

    Returns:
        Tuple of (passed_count, failed_count, messages)
    """
    rule_test_dir = get_test_data_dir(rule_name)
    messages = []
    passed = 0
    failed = 0

    if not rule_test_dir.exists():
        messages.append(f"No test data directory: {rule_test_dir}")
        return 0, 0, messages

    # Load rule
    check_rule = load_rule(rule_name)
    if check_rule is None:
        messages.append(f"Could not load rule: {rule_name}")
        return 0, 1, messages

    # Find test files
    test_files = sorted(rule_test_dir.glob("*.txt"))
    if not test_files:
        messages.append(f"No test files in: {rule_test_dir}")
        return 0, 0, messages

    # Run tests
    for test_file in test_files:
        test_passed, msg = run_single_test(
            rule_name, check_rule, nlp, test_file, verbose
        )
        messages.append(msg)
        if test_passed:
            passed += 1
        else:
            failed += 1

    return passed, failed, messages


def main():
    parser = argparse.ArgumentParser(
        description="Test runner for Leichte Sprache rules.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test-suite/test_runner.py                  # Run all tests
    python test-suite/test_runner.py abkuerzungen    # Test specific rule
    python test-suite/test_runner.py -v              # Verbose output
    python test-suite/test_runner.py --list          # List available rules
        """,
    )
    parser.add_argument(
        "rules", nargs="*", help="Rule(s) to test (default: all rules with test data)"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output with violation details",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available rules and exit"
    )
    args = parser.parse_args()

    # List rules
    if args.list:
        available = get_available_rules()
        logger.info("Available rules:")
        for rule in available:
            test_dir = get_test_data_dir(rule)
            has_tests = test_dir.exists() and any(test_dir.glob("*.txt"))
            status = "[has tests]" if has_tests else "[no tests]"
            logger.info(" %s %s", rule, status)
        return 0

    # Determine which rules to test
    if args.rules:
        rules_to_test = args.rules
    else:
        # Test all rules that have test data
        rules_to_test = [
            d.name
            for d in TEST_SUITE_DIR.iterdir()
            if d.is_dir() and any(d.glob("*.txt"))
        ]

    if not rules_to_test:
        logger.info("No test data found. Create test files in test-suite/<rule_name>/*.txt")
        logger.info("\nTest file format:")
        logger.info(" # expected: flag|pass")
        logger.info(" # description: Optional description")
        logger.info(" ")
        logger.info(" Text to test goes here.")
        return 1

    # Load spaCy model once
    logger.info("Loading spaCy model...")
    nlp = spacy.load("de_core_news_lg")

    # Run tests
    total_passed = 0
    total_failed = 0

    logger.info("\n" + "=" * 70)
    logger.info("RUNNING TESTS")
    logger.info("=" * 70)

    for rule_name in sorted(rules_to_test):
        logger.info("\n--- %s ---", rule_name)
        passed, failed, messages = run_tests_for_rule(rule_name, nlp, args.verbose)
        for msg in messages:
            logger.info(" %s", msg)
        total_passed += passed
        total_failed += failed

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    total = total_passed + total_failed
    logger.error("Total: %s tests, %s passed, %s failed", total, total_passed, total_failed)

    if total_failed > 0:
        logger.error("\nResult: FAILED")
        return 1
    elif total == 0:
        logger.info("\nResult: NO TESTS")
        return 1
    else:
        logger.info("\nResult: PASSED")
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
