# Regel: Kurze Wörter

Erkennt zu lange Wörter. Kurze Wörter sind leichter lesbar.

## Funktionsweise

- Analysiert Silbenanzahl (max. 3 Silben)
- Analysiert Zeichenlänge (max. 12 Zeichen)
- Nutzt pyphen für deutsche Silbentrennung

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Schwellenwerte |
| `test.py` | Tests: `python -m regeln.kurze_woerter.test` |

## Konfiguration (config.py)

```python
MAX_SILBEN = 3
MAX_ZEICHEN = 12
KOMPLEXE_WOERTER_SCHWELLE = 4
```

## Beispiele

**OK:** "Das Auto steht hier.", "Der Mann geht."

**Flagged:** "Die Bundeskanzlerin sprach.", "Digitalisierungsstrategien"

## Verwendung

```python
import spacy
from regeln.kurze_woerter import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Die Veranstaltungsorganisation.")
errors = pruefe_regel(doc)
```
