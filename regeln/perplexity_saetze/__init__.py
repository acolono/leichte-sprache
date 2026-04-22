"""
Perplexity-based sentence complexity analysis for Leichte Sprache.

This package provides a rule for detecting complex sentences using
an N-gram language model trained on Leichte Sprache data.

Example usage:
    >>> from regeln.perplexity_saetze import PerplexitySaetzeRule
    >>>
    >>> rule = PerplexitySaetzeRule()
    >>> result = rule.check(doc)
    >>>
    >>> for v in result.violations:
    ...     print(f"{v.type}: {v.message}")

Legacy usage:
    >>> from regeln.perplexity_saetze import check_rule
    >>>
    >>> errors = check_rule(doc)
"""

from .regel import check_rule

pruefe_regel = check_rule

__all__ = ["check_rule", "pruefe_regel"]
