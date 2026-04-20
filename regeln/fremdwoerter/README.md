# Regel: Fremdwörter

Erkennt Fremdwörter die durch deutsche Alternativen ersetzt werden sollten.

## Funktionsweise

- Wörterbuch mit 100+ Fremdwörtern und deutschen Alternativen
- Wortfrequenz-Analyse für Seltenheits-Bewertung
- Erkennt typische Fremdwort-Endungen (-tion, -ismus, etc.)

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen (aktuell leer) |
| `test.py` | Tests: `python -m regeln.fremdwoerter.test` |

## Beispiele

**OK:** "Die Verwaltung ist zuständig.", "Wir brauchen einen Plan."

**Flagged:** "Die Administration evaluiert Konzepte."

## Verwendung

```python
import spacy
from regeln.fremdwoerter import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Die Administration ist zuständig.")
errors = pruefe_regel(doc)
```
