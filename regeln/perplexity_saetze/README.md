# Regel: Perplexität

Erkennt komplexe Sätze mittels Perplexitäts-Analyse. Hohe Perplexität = schwer vorhersagbar = schwer verständlich.

## Funktionsweise

- Kneser-Ney interpoliertes 3-Gramm-Modell trainiert auf Leichte-Sprache-Korpus
- Berechnet normalisierte Perplexität pro Satz (Perplexität / Anzahl Tokens)
- Flaggt Sätze über Schwellenwert als komplex

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Schwellenwerte und N-Gramm-Größe |
| `perplexity_model.pkl` | Trainiertes Kneser-Ney Modell |
| `test.py` | Tests ausführen: `python -m regeln.perplexity_saetze.test` |

## Konfiguration (config.py)

```python
THRESHOLD_KOMPLEX_SATZ = 500.0       # Komplex
THRESHOLD_SEHR_KOMPLEX_SATZ = 1000.0 # Sehr komplex
N_GRAM_SIZE = 3                      # Trigram
```

## Beispiele

**OK:** "Das Haus ist groß.", "Der Mann geht nach Hause."

**Flagged:** "Die Administration evaluiert komplexe strategische Konzepte."

## Verwendung

```python
import spacy
from regeln.perplexity_saetze import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Das Haus ist groß.")
errors = pruefe_regel(doc)  # [] = OK
```
