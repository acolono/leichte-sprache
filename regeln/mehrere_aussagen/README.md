# Regel: Mehrere Aussagen

Erkennt Sätze mit mehreren Aussagen. Ein Satz sollte nur eine Aussage enthalten.

## Funktionsweise

- BERT-Modell trainiert auf StaGE-Dataset (Statement Segmentation)
- Linguistische Heuristiken für Pre-Filtering
- Erkennt koordinierte Verben und multiple Subjekte

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen |
| `model/` | Trainiertes BERT-Modell |
| `test.py` | Tests: `python -m regeln.mehrere_aussagen.test` |

## Beispiele

**OK:** "Das Haus ist groß.", "Der Mann geht."

**Flagged:** "Der Mann geht in den Park und kauft ein Eis."

## Verwendung

```python
import spacy
from regeln.mehrere_aussagen import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Er arbeitet und sie liest.")
errors = pruefe_regel(doc)
```
