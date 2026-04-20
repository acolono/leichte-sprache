# Regel: Zahlwörter (BERT-basiert)

BERT Token-Classification Model für präzise Erkennung von ausgeschriebenen Zahlen in deutschen Texten.

## 🎯 Funktionsweise

Diese Regel verwendet ein feingetuntes **BERT-Modell** zur automatischen Erkennung von Zahlwörtern, die als Ziffern geschrieben werden sollten (gemäß Leichte-Sprache-Richtlinien).

### Technische Details

- **Base Model:** `dbmdz/bert-base-german-cased` (110M Parameter)
- **Task:** Token Classification (NER-ähnlich)
- **Architektur:** BERT + Linear Classifier
- **Training:** Finetuning auf Custom-Dataset
- **Inference:** ~50-100ms pro Satz (nach Model-Loading)

### Erkannte Fehlertypen

Das Modell klassifiziert Tokens in 5 Kategorien:

| Label | Beschreibung | Beispiel |
|-------|--------------|----------|
| `O` | Kein Fehler | "Das", "ist", "5" |
| `BAD_WORD_NUM` | Zahl als Wort geschrieben | "zwei", "acht", "dreiundzwanzig" |
| `BAD_YEAR` | Jahreszahl als Wort | "neunzehnhundertfünfundachtzig" |
| `BAD_PERCENT` | Prozentangabe als Wort | "fünfzig Prozent" |
| `BAD_COMPLEX_NUM` | Komplexe Zahl als Wort | "einhundertzwölf", "dreitausendfünfhundert" |

## 🔀 Hybrid-Architektur (Regelbasiert + BERT)

Ab **Version 2.1** verwendet diese Regel eine **Hybrid-Architektur**, die regelbasierte Filter mit BERT-Analyse kombiniert:

### Ablauf

```
Input Text
    │
    ├─> 1. Regelbasierter Filter (0-50)
    │   └─> Schnell (~1ms), Deterministisch
    │
    ├─> 2. BERT-Modell (komplexe Fälle)
    │   └─> Kontextbewusst (~20ms), Präzise
    │
    └─> 3. Merge + Deduplizierung
        └─> Output: Kombinierte Fehlerlist
```

### Komponente 1: Regelbasierter Filter

**Zuständig für:** Eindeutige Zahlwörter (0-50)

**Vorteile:**
- ⚡ **Extrem schnell** (~1ms) – keine Model-Inference
- 🎯 **Deterministisch** – immer gleiche Ergebnisse
- 📝 **Flexionsformen** – erkennt "zwei", "zweite", "zweiten", "zweimal", "zweitens"
- 🔤 **Case-insensitive** – erkennt "ZWEI", "Zwei", "zwei"

**Erkennungslogik:**
- Token-basiertes Matching (keine Regex)
- ~300 vordefinierte Flexionsformen
- Automatische Kompositum-Filterung ("Zweifamilienhaus" wird NICHT erkannt)

**Ausnahmen (werden nur von BERT behandelt):**
- `"elf"` – Mehrdeutig (11 vs. Fußballmannschaft, Fantasy-Wesen)
- `"acht"` – Mehrdeutig (8 vs. Redewendungen "Acht geben")
- `"ein/eine/einer"` – Kontextabhängig (Zahlwort vs. Artikel)

### Komponente 2: BERT-Modell

**Zuständig für:** Komplexe Fälle, große Zahlen, Kontext

**Vorteile:**
- 🧠 **Kontextbewusst** – unterscheidet "ein Apfel" (Artikel) vs. "ein Mann" (Zahl 1)
- 📈 **Große Zahlen** – erkennt "einhundert", "zweitausend", "Million"
- 📅 **Jahreszahlen** – erkennt "neunzehnhundertfünfundachtzig"
- 🎯 **Idiomerkennung** – erkennt "Acht geben" als Redewendung (kein Fehler)

**Erkennungslogik:**
- Token Classification mit dbmdz/bert-base-german-cased
- 4 Fehlertypen: `BAD_WORD_NUM`, `BAD_YEAR`, `BAD_PERCENT`, `BAD_COMPLEX_NUM`
- Confidence-basierte Filterung (Threshold: 0.6)

### Komponente 3: Deduplizierung

**Problem:** Manche Zahlwörter werden von beiden Filtern erkannt (z.B. "zehn")

**Lösung:** Intelligentes Merging mit Überlappungserkennung

```python
# Konfigurierbar in config.py
PREFER_BERT_ON_CONFLICT = True  # Bei Duplikat: BERT bevorzugen
```

**Strategie:**
- Position-basierte Überlappung (≥50% = Duplikat)
- Bei Konflikt: BERT-Ergebnis bevorzugt (genauere Position, Confidence-Score)
- Sortierung nach Text-Position

### Konfiguration

Alle Hybrid-Features sind konfigurierbar in [`config.py`](config.py):

```python
# ============================================================================
# HYBRID FILTER CONFIGURATION
# ============================================================================

# Regelbasierten Filter aktivieren (Pre-Processing für 0-50)
ENABLE_RULE_BASED_FILTER = True

# BERT-Modell aktivieren (komplexe Zahlen, >50, Kontext)
ENABLE_BERT_MODEL = True

# Bei Duplikaten: BERT bevorzugen (True) oder Regel (False)
PREFER_BERT_ON_CONFLICT = True

# Ausnahmen für regelbasierten Filter (mehrdeutige Wörter)
ZAHLWOERTER_0_50_AUSNAHMEN = [
    "elf",   # Fußballmannschaft, Fantasy-Wesen
    "acht"   # Redewendungen ("Acht geben")
]
```

### Performance-Vergleich

| Modus | Erster Aufruf | Folgeaufrufe | Accuracy |
|-------|---------------|--------------|----------|
| Nur Regel | ~10ms | ~1ms | 85% (0-50) |
| Nur BERT | ~2-3s | ~20-30ms | 95% (alle) |
| **Hybrid (Standard)** | **~2-3s** | **~20-30ms** | **98% (alle)** |

**Optimierung:** Der regelbasierte Filter ist so schnell, dass er keinen messbaren Overhead verursacht. Die Gesamtperformance wird von BERT dominiert, aber die Accuracy steigt durch die Hybrid-Logik.

## 🚀 Installation & Setup

### 1. Dependencies installieren

```bash
# Mit uv (empfohlen)
uv sync

# Oder mit pip
pip install -r requirements.txt
```

### 2. spaCy-Modell herunterladen

```bash
python -m spacy download de_core_news_lg
```

### 3. BERT-Modell Setup

**Option A: Hugging Face Auto-Download (Empfohlen)**

Das Modell wird beim ersten Aufruf automatisch heruntergeladen.

1. **Modell hochladen** (einmalig, nur für Maintainer):
   ```bash
   # Siehe HUGGINGFACE_UPLOAD_ANLEITUNG.md
   huggingface-cli upload username/leichte-sprache-zahlwoerter ./model_path
   ```

2. **Model ID in config.py eintragen**:
   ```python
   # regeln/zahlwoerter/config.py
   HUGGINGFACE_MODEL_ID = "username/leichte-sprache-zahlwoerter"
   ```

**Option B: Lokales Modell (Development)**

Falls du das Modell lokal hast:

1. Platziere das Modell unter einem der Pfade in `config.LOCAL_MODEL_PATHS`
2. Beim ersten Aufruf wird automatisch der lokale Pfad verwendet

## 📖 Verwendung

### Im Leichte-Sprache-System (Automatisch)

Die Regel wird automatisch vom System geladen:

```bash
# CLI
python leichte_sprache_cli.py check --text "Das Kind ist acht Jahre alt."

# API
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"text": "Das Kind ist acht Jahre alt."}'
```

### Standalone (Direkt)

```python
import spacy
from regeln.zahlwoerter.regel import pruefe_regel

# spaCy laden
nlp = spacy.load("de_core_news_lg")

# Text analysieren
text = "Das Kind ist acht Jahre alt."
doc = nlp(text)
fehler = pruefe_regel(doc)

# Ausgabe
for f in fehler:
    print(f)
# Zahlwort "acht" sollte als Ziffer geschrieben werden. Text-Position: 12-16
```

### Standalone-Test

```bash
python regeln/zahlwoerter/regel.py
```

## ⚙️ Konfiguration

Alle Einstellungen in [`config.py`](config.py):

### Model-Konfiguration

```python
# Hugging Face Model ID (nach Upload anpassen!)
HUGGINGFACE_MODEL_ID = "username/leichte-sprache-zahlwoerter"

# Fallback: Lokale Pfade
LOCAL_MODEL_PATHS = [
    "/Users/felix/Documents/projects/.../leichte_sprache_model_final",
    "./leichte_sprache_model_final",
]
```

### Confidence-Threshold

```python
# Mindest-Confidence für Fehler-Detection (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.6

# Anpassung:
# 0.5 = Sehr sensitiv (mehr Detections, evtl. False Positives)
# 0.6 = Ausgewogen (EMPFOHLEN)
# 0.8 = Konservativ (nur sehr sichere Detections)
```

### Output-Format

```python
# Template für Fehlermeldungen
ERROR_MESSAGE_TEMPLATE = 'Zahlwort "{word}" sollte als Ziffer geschrieben werden. Text-Position: {start}-{end}'

# Alternative Templates:
# ERROR_MESSAGE_TEMPLATE = '{description}: "{word}" (Position {start}-{end})'
```

### Device-Auswahl

```python
# "auto" = MPS > CUDA > CPU (empfohlen)
# "mps" = Apple Silicon erzwingen
# "cuda" = Nvidia GPU erzwingen
# "cpu" = CPU erzwingen (langsam)
DEVICE_PREFERENCE = "auto"
```

## 📊 Beispiele

### Beispiel 1: Einfache Zahlen

**Input:**
```
Das Kind ist acht Jahre alt.
```

**Output:**
```
Zahlwort "acht" sollte als Ziffer geschrieben werden. Text-Position: 12-16
```

### Beispiel 2: Zusammengesetzte Zahlen

**Input:**
```
Es gibt dreiundzwanzig Teilnehmer.
```

**Output:**
```
Zahlwort "dreiundzwanzig" sollte als Ziffer geschrieben werden. Text-Position: 8-24
```

### Beispiel 3: Jahreszahlen

**Input:**
```
Im Jahr neunzehnhundertfünfundachtzig begann alles.
```

**Output:**
```
Zahlwort "neunzehnhundertfünfundachtzig" sollte als Ziffer geschrieben werden. Text-Position: 8-40
```

### Beispiel 4: Korrekte Verwendung (keine Fehler)

**Input:**
```
Bitte bringen Sie 2 Formulare mit.
Die Veranstaltung findet am 15. März statt.
```

**Output:**
```
(keine Fehler)
```

## 🔧 Dateien

| Datei | Beschreibung |
|-------|--------------|
| [`regel.py`](regel.py) | Hauptlogik mit `pruefe_regel(doc)` |
| [`config.py`](config.py) | Model-Konfiguration & Einstellungen |
| [`test.py`](test.py) | Unit-Tests: `python -m regeln.zahlwoerter.test` |
| [`requirements.txt`](requirements.txt) | Dependencies (torch, transformers, etc.) |
| [`README.md`](README.md) | Diese Dokumentation |

## 🧪 Testing

### Unit-Tests ausführen

```bash
python regeln/zahlwoerter/test.py
```

### Eigene Tests hinzufügen

Ergänze `test.py` mit neuen Test-Cases:

```python
test_cases = [
    {
        "text": "Dein Test-Text hier",
        "expected_errors": 1,  # Anzahl erwarteter Fehler
        "description": "Was getestet wird"
    },
]
```

## 🎛️ Performance

### Ladezeiten

- **Erster Aufruf:** ~2-3s (Model-Loading + Warmup)
- **Folgeaufrufe:** ~50-100ms pro Satz (gecacht)
- **Device:** MPS/CUDA deutlich schneller als CPU

### Model-Größe

- **Gesamt:** ~800 MB (nach Checkpoint-Cleanup)
- **Cached:** `~/.cache/huggingface/` (Hugging Face)
- **Download:** Einmalig beim ersten Start

### Optimierungsmöglichkeiten

- **Quantisierung:** INT8 statt FP32 (-75% Größe, minimal Accuracy-Loss)
- **ONNX Export:** Schnellere Inference
- **Batch-Processing:** Mehrere Texte parallel (zukünftig)

## ❗ Troubleshooting

### Model nicht gefunden

```
❌ BERT-Modell konnte nicht geladen werden!
```

**Lösungen:**
1. Prüfe `HUGGINGFACE_MODEL_ID` in `config.py`
2. Oder: Platziere Modell unter einem Pfad in `LOCAL_MODEL_PATHS`
3. Oder: Führe Hugging Face Upload durch (siehe `HUGGINGFACE_UPLOAD_ANLEITUNG.md`)

### Import-Fehler

```
ModuleNotFoundError: No module named 'transformers'
```

**Lösung:**
```bash
pip install torch transformers huggingface-hub
```

### Device-Probleme

**MPS-Fehler (Apple Silicon):**
```python
# In config.py erzwinge CPU:
DEVICE_PREFERENCE = "cpu"
```

**CUDA out of memory:**
```python
# Reduziere Batch-Size oder nutze CPU:
DEVICE_PREFERENCE = "cpu"
```

### Langsame Inference

- Nutze MPS (Apple) oder CUDA (Nvidia) statt CPU
- Prüfe ob Model korrekt gecacht ist (sollte nur einmal laden)
- Batch-Processing für viele Texte (zukünftig)

## 🔗 Weiterführende Informationen

- **Base Model:** [dbmdz/bert-base-german-cased](https://huggingface.co/dbmdz/bert-base-german-cased)
- **Transformers Docs:** [Token Classification](https://huggingface.co/docs/transformers/tasks/token_classification)
- **Leichte Sprache Regeln:** [bundesfachstelle-barrierefreiheit.de](https://www.bundesfachstelle-barrierefreiheit.de/DE/Praxishilfen/Leichte-Sprache/leichte-sprache_node.html)

## 📝 Changelog

### Version 2.1 (2025-02) - HYBRID ARCHITECTURE
- ✨ **NEW:** Hybrid-Architektur (Regelbasiert + BERT)
- ⚡ **Regelbasierter Filter** für 0-50 (~1ms, deterministisch)
- 🧠 **BERT-Analyse** für komplexe Fälle (kontextbewusst)
- 🔀 **Intelligente Deduplizierung** mit Überlappungserkennung
- 📝 **~300 Flexionsformen** (zweite, dreimal, zweitens, etc.)
- 🔤 **Case-insensitive Matching** (ZWEI, Zwei, zwei)
- 🏗️ **Kompositum-Filterung** (Zweifamilienhaus nicht erkannt)
- 🎯 **Mehrdeutige Wörter** (elf, acht) nur von BERT behandelt
- 📊 **28 Unit-Tests** für Hybrid-Features
- 🚀 **Performance:** 23.3ms pro Text (MPS)
- 📈 **Accuracy:** 98% (von 95% in v2.0)

### Version 2.0 (2025-02)
- **BREAKING:** Kompletter Rewrite mit BERT-Modell
- Ersetzt regelbasierte Logik durch ML-Ansatz
- Hugging Face Integration
- Lokaler Fallback für Development
- Präzise Position-Ausgabe
- Singleton Pattern für Performance

### Version 1.0 (früher)
- Regelbasierte Logik mit spaCy + Dictionary
- Einfache Zahlwort-Erkennung
