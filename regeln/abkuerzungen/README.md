# Regel: Abkürzungen (BERT)

BERT-basierte Abkürzungserkennung mit trainiertem NER-Modell. Erkennt auch unbekannte Abkürzungen durch Kontext-Analyse.

## Funktionsweise

- Feinabgestimmtes BERT-Modell (bert-base-german-cased)
- B-I-O Tagging Schema für Multi-Word Abbreviations
- Post-Processing für fragmentierte Predictions

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen |
| `model/` | Trainiertes BERT-Modell |
| `test.py` | Tests: `python -m regeln.abkuerzungen_bert.test` |
| `train.py` | Training-Script |
| `train_config.py` | Training-Konfiguration |
| `data/` | Training-Daten |

## Performance

- Precision: 97.0%
- Recall: 98.9%
- F1-Score: 97.95%

## Beispiele

**OK:** "Das Haus ist groß."

**Flagged:** "Die GmbH arbeitet.", "Prof. Dr. Schmidt", "ca. 50"

## Verwendung

```python
import spacy
from regeln.abkuerzungen_bert import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Die GmbH arbeitet.")
errors = pruefe_regel(doc)
```

## Training

Re-train the model with:
```bash
python -m regeln.abkuerzungen_bert.train
```

Training data: `data/abbreviation_dataset_final.jsonl` (11,877 examples)
