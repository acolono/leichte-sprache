# Regel: Nebensätze

Erkennt komplexe Nebensätze die das Verständnis erschweren.

## Funktionsweise

- Erkennt subordinierende Konjunktionen (dass, weil, obwohl, wenn, etc.)
- Findet Relativsätze (der, die, das + Verb)
- Analysiert Satzstruktur mit spaCy Dependency Parser

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen (aktuell leer) |
| `test.py` | Tests: `python -m regeln.nebensaetze.test` |

## Beispiele

**OK:** "Der Mann geht spazieren.", "Das Auto ist rot."

**Flagged:** "Ich weiß, dass es regnet.", "Der Mann, der dort steht, ist mein Bruder."

## Verwendung

```python
import spacy
from regeln.nebensaetze import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Ich weiß, dass es regnet.")
errors = pruefe_regel(doc)
```
