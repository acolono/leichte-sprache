# Regel: Interpunktion

Erkennt problematische Satzzeichen. Einfache Interpunktion ist leichter verständlich.

## Funktionsweise

- Erkennt Gedankenstriche und Einschübe
- Prüft auf zu viele Kommata
- Erlaubt: Punkt, Fragezeichen, Ausrufezeichen, Doppelpunkt

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen |
| `test.py` | Tests: `python -m regeln.interpunktion.test` |

## Beispiele

**OK:** "Das ist toll!", "Der Preis: 50 Euro."

**Flagged:** "Das Auto – ein großes Fahrzeug – steht hier."

## Verwendung

```python
import spacy
from regeln.interpunktion import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Das ist ein Satz.")
errors = pruefe_regel(doc)
```
