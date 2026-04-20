# Regel: Negationen

Erkennt Verneinungen. Positive Formulierungen sind leichter verständlich.

## Funktionsweise

- Erkennt Negationspartikel: "nicht", "nie"
- Erkennt negative Determinanten: "kein", "keine"
- Erkennt präpositionale Negation: "ohne"
- Nutzt spaCy Morphologie (dep_=ng)

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Negations-Typen aktivieren |
| `test.py` | Tests: `python -m regeln.negationen.test` |

## Konfiguration (config.py)

```python
ERKENNE_NICHT = True
ERKENNE_KEIN = True
ERKENNE_OHNE = True
ERKENNE_WEDER_NOCH = True
```

## Beispiele

**OK:** "Das ist möglich.", "Er bleibt zu Hause."

**Flagged:** "Er ist nicht da.", "Er hat kein Auto.", "Er ging ohne Jacke."

## Verwendung

```python
import spacy
from regeln.negationen import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Er ist nicht da.")
errors = pruefe_regel(doc)
```
