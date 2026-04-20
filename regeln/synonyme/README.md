# Regel: Synonyme

Erkennt inkonsistente Synonym-Verwendung. Einheitliche Begriffe sind leichter verständlich.

## Funktionsweise

- Wörterbuch mit deutschen Synonympaaren
- Semantische Vektoranalyse mit spaCy
- Empfiehlt das einfachste/häufigste Wort

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Ähnlichkeits-Schwellenwerte |
| `test.py` | Tests: `python -m regeln.synonyme.test` |

## Konfiguration (config.py)

```python
AEHNLICHKEIT_HOCH = 0.75
AEHNLICHKEIT_MITTEL = 0.65
AEHNLICHKEIT_NIEDRIG = 0.55
BEVORZUGE_KUERZERE_WOERTER = True
```

## Beispiele

**OK:** "Das Auto steht da. Das Auto ist rot."

**Flagged:** "Die Implementierung der Umsetzung dauert."

## Verwendung

```python
import spacy
from regeln.synonyme import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Die Implementierung der Umsetzung.")
errors = pruefe_regel(doc)
```
