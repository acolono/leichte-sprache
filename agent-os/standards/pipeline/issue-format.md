# Issue Dict Format

Each issue in the `issues` list has this structure:

```python
{
    "rule_id": "<rule_name>_issue",  # e.g. "negationen_issue"
    "text": "problematic word or phrase",
    "message": "Full German violation message with Besser: suggestion",
    "start": int | None,  # char offset in original text (0-based)
    "end": int | None      # char offset, exclusive
}
```

- `rule_id`: rule directory name + `_issue` suffix (appended by analyzer)
- `start`/`end`: character positions in the original text; `None` if position couldn't be determined
- `text`: extracted from the violation message (first quoted term)
- `message`: the full string returned by `pruefe_regel`
