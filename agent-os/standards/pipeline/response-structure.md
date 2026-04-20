# Analysis Response Structure

`analyse_text()` returns a dict with two possible shapes:

## Success

```python
{
    "annotated_text": "Text with [rule: word] annotations",
    "statistics": {
        "total_violations": int,
        "unique_violations": int,
        "violations_by_rule": {"rule_name": count}
    },
    "issues": [<Issue dicts>]
}
```

## Error (legacy)

```python
{"error": "description of what went wrong"}
```

Consumers must check for `"error" in result` before accessing other fields.

Note: the dict-based error format is a legacy design. New service-layer code should prefer raising exceptions.
