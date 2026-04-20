"""Protocol definition for Leichte Sprache rule modules.

All rule modules must expose a check_rule function matching this protocol.
Validated at rule load time by analysis_service._load_rules().
"""

from typing import List, Protocol, runtime_checkable

from spacy.tokens import Doc


@runtime_checkable
class RuleFunction(Protocol):
    """Interface contract for rule checker functions.

    Every rule module's check_rule must be a callable that accepts
    a spaCy Doc and returns a list of violation message strings.
    """

    def __call__(self, doc: Doc) -> List[str]: ...
