# pruefe_regel Interface

Every rule must export one function:

```python
def pruefe_regel(doc: spacy.tokens.Doc) -> List[str]:
```

- Input: spaCy `Doc` object (German `de_core_news_lg` model)
- Output: list of violation message strings (empty list = no violations)
- Let exceptions propagate — do not silently swallow errors

Variable names inside rules must be English.

The function is imported via `__init__.py` and called by `analysis_service.py`.
