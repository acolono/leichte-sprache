# Regel: Redewendungen

Erkennt Redewendungen und bildhafte Ausdrücke. Wörtliche Sprache ist leichter verständlich.

## Funktionsweise

- Wörterbuch mit 50+ deutschen Redewendungen
- Pattern-Matching für Variationen
- Kategorisiert nach Themen: aufgeben, erfolg, gefühl, etc.

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Erkennungs-Typen aktivieren |
| `test.py` | Tests: `python -m regeln.redewendungen.test` |

## Konfiguration (config.py)

```python
ERKENNE_REDEWENDUNGEN = True
ERKENNE_METAPHERN = True
ERKENNE_BILDHAFTE_AUSDRUECKE = True
```

## Beispiele

**OK:** "Er gibt auf.", "Sie ist sehr glücklich."

**Flagged:** "Er wirft die Flinte ins Korn.", "Sie schwebt auf Wolke sieben."

## Verwendung

```python
import spacy
from regeln.redewendungen import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Das ist wörtlich gemeint.")
errors = pruefe_regel(doc)
```
