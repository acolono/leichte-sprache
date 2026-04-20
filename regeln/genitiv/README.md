# Regel: Genitiv

Erkennt Genitiv-Konstruktionen. "Von"-Konstruktionen sind leichter verständlich.

## Funktionsweise

- Erkennt morphologischen Genitiv: "des Mannes", "der Frau"
- Erkennt Possessiv-s: "Peters Auto"
- Nutzt spaCy Morphologie-Analyse

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Genitiv-Typen aktivieren/deaktivieren |
| `test.py` | Tests: `python -m regeln.genitiv.test` |

## Konfiguration (config.py)

```python
ERKENNE_MORPHOLOGISCHEN_GENITIV = True  # "des Mannes"
ERKENNE_POSSESSIV_S = True              # "Peters Auto"
ERKENNE_VON_KONSTRUKTION = False        # OK in Leichter Sprache
```

## Beispiele

**OK:** "Das Auto von Peter.", "Das Haus von dem Mann."

**Flagged:** "Das Auto des Mannes.", "Peters Fahrrad."

## Verwendung

```python
import spacy
from regeln.genitiv import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Das Auto des Nachbarn.")
errors = pruefe_regel(doc)
```
