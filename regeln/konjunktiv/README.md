# Regel: Konjunktiv

Erkennt Konjunktiv-Formen. Indikativ ist leichter verständlich.

## Funktionsweise

- Erkennt Konjunktiv I (indirekte Rede): "er sei", "sie habe"
- Erkennt Konjunktiv II (Irrealis): "wäre", "hätte", "könnte"
- Erkennt würde-Formen: "würde gehen"

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen |
| `test.py` | Tests: `python -m regeln.konjunktiv.test` |

## Beispiele

**OK:** "Er sagte, dass er kommt.", "Das Auto ist rot."

**Flagged:** "Er sagte, er sei krank.", "Ich würde gerne kommen."

## Verwendung

```python
import spacy
from regeln.konjunktiv import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Ich würde gerne kommen.")
errors = pruefe_regel(doc)
```
