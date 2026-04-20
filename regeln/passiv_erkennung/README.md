# Regel: Passiv-Erkennung

Erkennt Passiv-Konstruktionen. Aktiv-Formulierungen sind leichter verständlich.

## Funktionsweise

- Erkennt Vorgangspassiv: "wird gebaut", "wurde geschrieben"
- Nutzt spaCy Dependency Parser und Morphologie
- Zustandspassiv ("ist geöffnet") wird nicht erkannt (Konfiguration)

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen (aktuell leer) |
| `test.py` | Tests: `python -m regeln.passiv_erkennung.test` |

## Beispiele

**OK:** "Der Mann baut das Haus.", "Die Frau liest ein Buch."

**Flagged:** "Das Haus wird gebaut.", "Der Brief wurde geschrieben."

## Verwendung

```python
import spacy
from regeln.passiv_erkennung import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Das Haus wird gebaut.")
errors = pruefe_regel(doc)
```
