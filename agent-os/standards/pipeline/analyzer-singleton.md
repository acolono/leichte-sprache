# Analyzer Singleton

`analysis_service.py` exposes a module-level `analyse_text(text)` function.

Internally it uses a lazy-initialized singleton `LeichteSpracheAnalyzer`:

```python
_analyzer = None

def analyse_text(text: str) -> dict:
    global _analyzer
    if _analyzer is None:
        _analyzer = LeichteSpracheAnalyzer()
    return _analyzer.analysiere_text(text)
```

- SpaCy model (`de_core_news_lg`) loads on first call — expensive (~2-3s)
- Rules are loaded once and cached on the instance
- All consumers (API, generator, tests) call `analyse_text()` — never instantiate `LeichteSpracheAnalyzer` directly
