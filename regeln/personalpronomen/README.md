# Regel: Personalpronomen

Erkennt problematische Pronomen-Verwendung. Klare Bezüge sind leichter verständlich.

## Funktionsweise

- Erkennt Pronomen-Häufung (>3 pro Satz)
- Erkennt unklare Bezüge: "es", "das" ohne klaren Bezug
- Nutzt personalpronomen-v1.0.0 Bibliothek

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen |
| `test.py` | Tests: `python -m regeln.personalpronomen.test` |

## Beispiele

**OK:** "Der Mann geht nach Hause.", "Das Auto ist rot."

**Flagged:** "Er sagte, dass er es ihr geben würde.", "Es ist wichtig."

## Verwendung

```python
import spacy
from regeln.personalpronomen import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Es ist wichtig.")
errors = pruefe_regel(doc)
```
