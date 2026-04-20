# Regel: Satzlänge

Erkennt zu lange Sätze. Ideal sind kurze Sätze mit maximal 10 Wörtern.

## Funktionsweise

- Zählt Wörter pro Satz (ohne Satzzeichen)
- Bewertet Komplexität durch Nebensätze, Passiv, Einschübe
- Kategorisiert: kurz (≤6), mittel (≤10), lang (≤15), sehr lang (>15)

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen (aktuell leer) |
| `test.py` | Tests: `python -m regeln.satzlaenge.test` |

## Beispiele

**OK:** "Das Auto ist rot.", "Der Mann geht nach Hause."

**Flagged:** "Der Mann, der gestern hier war, hat sein Auto vor dem Haus geparkt."

## Verwendung

```python
import spacy
from regeln.satzlaenge import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Das Auto ist rot.")
errors = pruefe_regel(doc)  # [] = OK
```
