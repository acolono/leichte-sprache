# Rule Module Structure

Every rule lives in `regeln/<rule_name>/` with exactly 4 files:

```
regeln/<rule_name>/
├── __init__.py
├── config.py
├── regel.py
└── README.md
```

All 4 files are mandatory, even if config.py has no parameters yet.

After creating a rule, add its description to `_regel_beschreibungen` in `analysis_service.py`.

Rules are auto-discovered at runtime — no manual registration needed beyond the description.

## File Templates

`__init__.py` (identical for every rule):

```python
"""Rule module."""

from .regel import pruefe_regel

__all__ = ["pruefe_regel"]
```

`regel.py` must import config at the top:

```python
from . import config  # noqa: F401
```

`config.py` — empty placeholder if no config needed:

```python
"""Configuration for the <rule_name> rule."""
```
