# Regel: perplexity_saetze

Hybride Prüfung für komplexe Satzstruktur. Kombiniert einen strikten
syntaktischen Check (DIN-SPEC-33429-konform) mit einem ergänzenden
LM-basierten Signal für Off-Register-Wortwahl.

## Zwei Signale

### 1. Struktur (Warnung, DIN-SPEC-33429-aligned)

SpaCy-Heuristiken erkennen:

- Nebensätze (Zählung, subordinierende Konjunktionen + Relativpronomen)
- Satzlänge (Soft- und Hard-Limit)
- Passiv-Konstruktionen (Vorgangspassiv: `werden` + Partizip)
- Subjekt-Verb-Distanz
- Dependency-Tiefe

Trifft einer dieser Thresholds zu, wird der Satz als strukturell komplex
geflaggt.

### 2. Perplexität (Hinweis, `[Hinweis Perplexität]` prefix)

Nur wenn der strukturelle Check still bleibt, prüft die Regel die
Off-Register-Wortwahl gegen das Vokabular eines Kneser-Ney-3-Gramm-Modells
(`model/perplexity_model.pkl`, trainiert auf
`data/leichte_sprache_korpus.txt`). Liegt der Anteil der nicht im Vokabular
enthaltenen Tokens über `THRESHOLD_OOV_RATIO` (Default 0.15), wird ein
Hinweis-Level-Violation emittiert.

Begründung für OOV statt voller Perplexität: NLTKs `.perplexity()` ist mit
~1 s/Satz zu langsam, und die Kalibrierung in `CALIBRATION.md` zeigt, dass
98 %+ des Off-Register-Signals allein von OOV-Unigrammen getragen werden.
Dieselben Sätze werden erkannt, bei O(tokens) statt O(n²) Aufwand.

### Deduplikation

Wenn der strukturelle Check bereits anschlägt, wird der Perplexitäts-Hinweis
unterdrückt. So vermeiden wir das in der Rule-Research-Phase identifizierte
"4×-redundante-Flags"-Problem.

## Dateien

| Datei | Zweck |
|---|---|
| `regel.py` | Hauptlogik mit `check_rule(doc)` + `pruefe_regel`-Alias |
| `config.py` | Alle Schwellenwerte, inkl. `ENABLE_PERPLEXITY_SIGNAL` |
| `model/perplexity_model.pkl` | Kneser-Ney-Vokabular (17 MB, via GitHub-Release) |
| `train.py` | Trainings-Pipeline (`tools/ml train perplexity_saetze`) |
| `train_config.py` | ML-CLI-Adapter |
| `CALIBRATION.md` | Threshold-Herleitung + Kalibrations-Daten |

## Verwendung

```python
import spacy
from regeln.perplexity_saetze import check_rule

nlp = spacy.load("de_core_news_lg")
doc = nlp("Die Protonenkollision fand statt.")
errors = check_rule(doc)
# → ['[Hinweis Perplexität] Satz enthält Wörter, die in Leichte-Sprache-Texten
#    selten vorkommen: "Die Protonenkollision fand statt.". Prüfen Sie die Wortwahl.']
```

## Modell trainieren

```bash
python -m tools.ml train perplexity_saetze --promote
```

Das Korpus wird aus `data/leichte_sprache_korpus.txt` gelesen. Trainingszeit:
wenige Minuten auf CPU.

## Modell evaluieren

```bash
python -m tools.ml evaluate perplexity_saetze --verbose
```

Gibt avg_perplexity, vocab_size und die Anzahl bewerteter Sätze aus.

## Tests

```bash
python test-suite/test_runner.py perplexity_saetze
```

18 Fixtures decken die typischen Fälle ab (strukturell komplex, off-register,
Dedup-Kanarien, DIN-SPEC-Gegenbeispiele, Länge-Soft-Limit, sub-floor-Sätze).

## Kein standardisiertes Signal

Die Perplexitäts-Komponente ist nicht Teil der in DIN SPEC 33429, dem
Netzwerk-LS-Regelwerk oder dem Hildesheim-Regelbuch geforderten Checks. Sie
läuft bewusst als `[Hinweis]` (nicht als Warning) und soll Redakteur:innen
lexikalische Aufmerksamkeits-Marker liefern, keine blockierenden Fehler.
