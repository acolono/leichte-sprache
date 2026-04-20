# Standalone Rule Testing

Every `regel.py` must include an `if __name__ == "__main__":` block for quick iteration.

```python
if __name__ == "__main__":
    import spacy
    nlp = spacy.load("de_core_news_lg")

    test_cases = [
        "Positive case (should flag).",
        "Negative case (should pass).",
    ]

    for text in test_cases:
        doc = nlp(text)
        results = pruefe_regel(doc)
        # print results
```

Run directly: `python regeln/<rule_name>/regel.py`

This is for development feedback. The canonical test suite is `test-suite/`.
