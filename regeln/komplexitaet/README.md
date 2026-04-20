# Regel: Komplexität (BERT)

BERT-basierte Erkennung komplexer Wörter mit deutschem Textmodell.

## Funktionsweise

- Nutzt MiriUll/distilbert-german-text-complexity Modell
- Bewertet Wort-Komplexität im Kontext
- Schlägt einfachere Alternativen vor

## Dateien

| Datei | Beschreibung |
|-------|--------------|
| `regel.py` | Hauptlogik mit `pruefe_regel(doc)` |
| `config.py` | Einstellungen |
| `textkomplexitaet/` | Analyzer-Paket |
| `test.py` | Tests: `python -m regeln.komplexitaet.test` |

## Beispiele

**OK:** "Das Haus ist groß.", "Der Mann geht nach Hause."

**Flagged:** "Die Administration evaluiert die Implementation."

## Verwendung

```python
import spacy
from regeln.komplexitaet import pruefe_regel

nlp = spacy.load("de_core_news_lg")
doc = nlp("Die Methodologie erfordert Analyse.")
errors = pruefe_regel(doc)
```
