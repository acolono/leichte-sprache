# StaGE Statement Segmentation Model

Dieses Verzeichnis enthält alle Komponenten für das **BERT-basierte Statement-Segmentations-Modell**, das auf dem **StaGE-Dataset** (Statement Segmentation in German Easy Language) trainiert wurde.

## 📋 Überblick

**Zweck**: Erkennung von Sätzen mit mehreren Aussagen/Statements in deutschen Texten (Leichte Sprache).

**Dataset**: GermEval 2024 Shared Task - StaGE
**Quelle**: https://german-easy-to-read.github.io/statements/
**Paper**: Proceedings of GermEval 2024 (KONVENS 2024)

## 📁 Verzeichnisstruktur

```
stage_statement_model/
├── data/                          # Trainingsdaten
│   ├── train.csv                  # Original-Trainingsdaten (von GitHub)
│   ├── trial.csv                  # Original-Testdaten (von GitHub)
│   ├── train_processed.csv        # Vorverarbeitete Train-Daten (80%)
│   ├── val_processed.csv          # Vorverarbeitete Validation-Daten (10%)
│   └── test_processed.csv         # Vorverarbeitete Test-Daten (10%)
│
├── scripts/                       # Training-Skripte
│   ├── preprocess_data.py         # Data Preprocessing Pipeline
│   ├── train_model_simple.py      # BERT-Training (manueller Loop)
│   └── train_model.py             # BERT-Training (Huggingface Trainer)
│
├── models/                        # Trainierte Modelle
│   └── stage_statement_classifier_best/
│       ├── pytorch_model.bin      # Trainierte Model Weights
│       ├── config.json            # Modell-Konfiguration
│       └── tokenizer files        # BERT Tokenizer
│
└── README.md                      # Diese Datei
```

## 🚀 Verwendung

### 1. Data Preprocessing

Lädt die Daten von GitHub und bereitet sie für das Training vor:

```bash
cd scripts
python preprocess_data.py
```

**Output**:
- Bereinigte und gelabelte Daten
- Train/Val/Test Split (80/10/10)
- Dataset-Statistiken

### 2. Modell-Training

Trainiert ein BERT-Modell für Binary Classification (1 Statement vs. mehrere Statements):

```bash
cd scripts
python train_model_simple.py
```

**Konfiguration**:
- Modell: `bert-base-german-cased`
- Task: Binary Classification (2 Klassen)
- Batch Size: 16
- Learning Rate: 2e-5
- Epochs: 5
- Max Sequence Length: 128

**Output**:
- Trainiertes Modell in `models/stage_statement_classifier_best/`
- Training Metriken (Accuracy, F1, Precision, Recall)

### 3. Integration in Regel

Das trainierte Modell wird automatisch von der Regel verwendet:

```python
# In regeln/regel_mehrere_aussagen.py
from regeln.regel_mehrere_aussagen import pruefe_regel

doc = nlp("Der Text zum Prüfen")
fehler = pruefe_regel(doc)
```

## 📊 Dataset-Statistiken

Nach Preprocessing:

- **Gesamt**: 2.495 Samples
- **Ein Statement**: 1.530 (61,3%)
- **Mehrere Statements**: 965 (38,7%)

Verteilung der Statement-Anzahl:
- 1 Statement: 61,3%
- 2 Statements: 29,3%
- 3 Statements: 7,7%
- 4+ Statements: 1,7%

## 🎯 Modell-Performance

Typische Metriken nach Training:

- **Accuracy**: ~85-90%
- **F1-Score**: ~85-88%
- **Precision**: ~86-90%
- **Recall**: ~84-88%

*(Exakte Werte variieren je nach Training-Run)*

## 🔧 Technische Details

### Modell-Architektur

```python
BERTStatementClassifier:
  - BERT Base (bert-base-german-cased)
  - Dropout Layer (0.1)
  - Linear Classifier (768 → 2)
```

### Dependencies

- `torch` >= 1.13.0
- `transformers` >= 4.20.0
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`

### Fallback-Mechanismus

Falls das Modell nicht verfügbar ist, verwendet die Regel eine **heuristische Methode**:

- Erkennung von Kommata
- Konjunktionen ("und", "aber", "oder")
- Relativpronomen
- Satzlänge

## 🧪 Testen der Regel

```bash
cd ../regeln
python regel_mehrere_aussagen.py
```

Test-Beispiele:
```python
"Das Haus ist groß."  # ✅ OK (1 Statement)
"Das Haus ist groß und hat einen schönen Garten."  # ❌ Fehler (2 Statements)
"Maria arbeitet im Büro, sie ist fleißig."  # ❌ Fehler (2 Statements)
```

## 📚 Referenzen

- **StaGE Shared Task**: https://german-easy-to-read.github.io/statements/
- **GitHub Repository**: https://github.com/german-easy-to-read/statements
- **Paper**: Schomacker et al. (2024) - Proceedings of GermEval 2024

## 👨‍💻 Autor

Senior Data Scientist
Erstellt für das Leichte-Sprache-Regelwerk

## 📝 Lizenz

Die Trainingsdaten unterliegen der Lizenz des StaGE-Projekts.
Das trainierte Modell und die Skripte sind für interne Verwendung.
