# Regel: Komposita

Erkennt lange zusammengesetzte Wörter. Kurze Wörter sind leichter lesbar.

## Funktionsweise

- Nutzt german_compound_splitter für professionelle Zerlegung
- Wörterbuch mit 180+ bekannten Komposita
- Konfigurierbare Modi: alle, nach Länge, nach Teilen

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Modi und Schwellenwerte |
| `test.py` | Tests: `python -m regeln.komposita.test` |

## Konfiguration (config.py)

```python
class KompositumKonfig:
    MODUS = KOMBINIERT        # ALLE_KOMPOSITA, NACH_LAENGE, NACH_TEILEN
    MIN_WORTLAENGE = 12       # Minimale Zeichenlänge
    MIN_TEILE_ANZAHL = 3      # Minimale Bestandteile
```

## Beispiele

**OK:** "Das Haus ist schön.", "Die Haustür ist offen."

**Flagged:** "Die Datenschutzgrundverordnung gilt.", "Qualitätssicherungsabteilung"

## Verwendung

```python
import spacy
from regeln.komposita import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Die Datenschutzgrundverordnung.")
errors = pruefe_regel(doc)
```
