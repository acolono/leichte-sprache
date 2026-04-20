"""Tests for rule Protocol validation (REFAC-03)."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

from regeln.protocol import RuleFunction


def _get_all_rule_names():
    """Discover all rule module names from regeln/ directory."""
    regeln_path = Path(__file__).parent.parent / "regeln"
    return sorted(
        d.name
        for d in regeln_path.iterdir()
        if d.is_dir() and d.name != "__pycache__" and (d / "regel.py").exists()
    )


RULE_NAMES = _get_all_rule_names()


@pytest.mark.parametrize("regel_module", RULE_NAMES)
def test_all_rules_satisfy_protocol(regel_module):
    """Every rule module must export a callable check_rule matching RuleFunction."""
    module = importlib.import_module(f"regeln.{regel_module}")
    assert hasattr(module, "check_rule"), f"{regel_module} has no check_rule"
    fn = module.check_rule
    assert isinstance(fn, RuleFunction), (
        f"{regel_module}.check_rule is not callable (got {type(fn).__name__})"
    )


def test_malformed_module_missing_function(tmp_path, monkeypatch):
    """A module without check_rule should fail the hasattr check."""
    fake_module = ModuleType("fake_rule")
    fake_module.something_else = "not a function"

    assert not hasattr(fake_module, "check_rule")


def test_malformed_module_non_callable(tmp_path, monkeypatch):
    """A module with non-callable check_rule should fail isinstance check."""
    fake_module = ModuleType("fake_rule")
    fake_module.check_rule = "not a function"

    assert not isinstance(fake_module.check_rule, RuleFunction)


def test_protocol_accepts_conforming_function():
    """A function with the right signature passes the Protocol check."""

    def check_rule(doc):
        return []

    assert isinstance(check_rule, RuleFunction)


def test_rule_count():
    """Ensure all 18 expected rules are discovered."""
    assert len(RULE_NAMES) == 18, (
        f"Expected 18 rules, found {len(RULE_NAMES)}: {RULE_NAMES}"
    )


@pytest.mark.parametrize("regel_module", ["personalpronomen", "komposita"])
def test_optional_deps_pass_protocol(regel_module):
    """Rules with optional dependencies must still load and pass Protocol check."""
    module = importlib.import_module(f"regeln.{regel_module}")
    fn = module.check_rule
    assert isinstance(fn, RuleFunction), (
        f"{regel_module} with optional deps failed Protocol check"
    )
