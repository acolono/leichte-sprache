# ML Rule Training and Evaluation Guide

The `tools.ml` CLI provides a unified interface for training and evaluating ML-based rule models. It discovers trainable rules automatically by scanning for `train_config.py` modules under `regeln/`, and handles device detection, staging, promotion, and evaluation through a consistent command set.

```bash
python -m tools.ml --help
```

## CLI Command Reference

### `train` command

Train an ML rule model. Output goes to a staging directory (`regeln/<rule>/model/.staging/`) -- never directly to the production model path.

```bash
python -m tools.ml train <rule> [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `rule` | `str` (positional) | `None` | Rule name to train (e.g. `abkuerzungen`) |
| `--list` | `bool` | `False` | List all trainable rules and exit |
| `--device` | `str` | `auto` | Compute device: `auto`, `cpu`, `cuda`, `mps` |
| `--seed` | `int` | `42` | Random seed for reproducibility |
| `--promote` | `bool` | `False` | Promote staged model to production after training |

**Examples:**

```bash
# List all trainable rules with status
python -m tools.ml train --list

# Train abkuerzungen to staging
python -m tools.ml train abkuerzungen

# Train with explicit device and seed
python -m tools.ml train abkuerzungen --device cpu --seed 123

# Train and promote in one step
python -m tools.ml train abkuerzungen --promote
```

### `evaluate` command

Evaluate a trained ML rule model. By default evaluates the deployed (production) model. Automatically compares against a staged model if one exists.

```bash
python -m tools.ml evaluate <rule> [OPTIONS]
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `rule` | `str` (positional) | required | Rule name to evaluate |
| `--model-dir` | `str` | `None` (deployed model) | Path to model directory to evaluate |
| `--verbose` / `-v` | `bool` | `False` | Show per-class metric breakdown |
| `--data` | `str` | `None` (rule default) | Path to custom evaluation dataset |
| `--json` | `bool` | `False` | Output results as JSON |

**Examples:**

```bash
# Evaluate deployed abkuerzungen model
python -m tools.ml evaluate abkuerzungen

# Verbose output with per-class breakdown
python -m tools.ml evaluate abkuerzungen --verbose

# Evaluate a specific model directory
python -m tools.ml evaluate abkuerzungen --model-dir regeln/abkuerzungen/model/.staging

# JSON output for scripting
python -m tools.ml evaluate abkuerzungen --json

# Evaluate with custom data
python -m tools.ml evaluate abkuerzungen --data path/to/test_data.jsonl
```

## Data Preparation

Training data is rule-specific and lives under `regeln/<rule>/` or `regeln/<rule>/data/`. The format varies by rule type:

- **NER rules** (abkuerzungen, zahlwoerter): JSONL with tokenized text and BIO labels
- **Classification rules** (mehrere_aussagen): Format defined in the rule's training script
- **N-gram rules** (perplexity_saetze): Plain text corpus (`leichte_sprache_korpus.txt`)
- **Custom PyTorch** (personalpronomen): JSONL with pronoun annotations and multi-label targets

Check each rule's `train_config.py` for the expected data path and format. The `train()` function documents its data expectations and raises `FileNotFoundError` with format details when data is missing.

## Training Workflow

A typical workflow using `abkuerzungen` as an example:

**1. List available rules:**

```bash
python -m tools.ml train --list
```

This shows each rule's name, model type, framework, and training status (trained/untrained).

**2. Train to staging:**

```bash
python -m tools.ml train abkuerzungen
```

Training output goes to `regeln/abkuerzungen/model/.staging/`. The production model at `regeln/abkuerzungen/model/` is not touched.

**3. Evaluate the staged model:**

```bash
python -m tools.ml evaluate abkuerzungen --verbose
```

When a staged model exists, evaluate automatically compares deployed vs. staged metrics side-by-side with color-coded deltas.

**4. Promote the staged model:**

If satisfied with the staged model's metrics, train with `--promote` (or re-train):

```bash
python -m tools.ml train abkuerzungen --promote
```

Promotion does two things:
1. **Backup**: Copies the current production model to `regeln/abkuerzungen/model/.backup/<timestamp>/`
2. **Promote**: Copies all files from `.staging/` into the `model/` directory, overwriting existing files

After promotion, restart the API server to load the new model.

You can also train and promote in separate steps -- train first, evaluate, then train again with `--promote` when ready.

## Evaluation

The `evaluate` command reports different metrics depending on the rule type:

**Supervised models (NER / classification):**
- Precision, Recall, F1, Accuracy, Support
- `--verbose` shows per-class breakdown (per-entity for NER, per-label for classification)

**Unsupervised models (n-gram):**
- Average perplexity, vocabulary size, sentences evaluated
- No P/R/F1 (unsupervised model)

**All rules:**
- Test-suite pass/fail results (runs `test-suite/test_runner.py <rule>` automatically)
- Staged vs. deployed comparison table (when a staged model exists and `--model-dir` is not explicitly set)

The `--json` flag outputs all metrics as a JSON object with keys: `rule`, `model_metrics`, `staged_metrics`, `test_suite`. Useful for CI pipelines or scripted comparisons.

## Rule Type Variations

The CLI supports four training paradigms. Each rule's `train_config.py` exports `METADATA`, `train()`, and `evaluate()`.

| Rule | Paradigm | Framework | Notes |
|------|----------|-----------|-------|
| abkuerzungen | HuggingFace Trainer NER | transformers | Token classification (BIO), standard HF TrainingArguments |
| zahlwoerter | HuggingFace Trainer NER | transformers | Token classification (4 error types), same pattern as abkuerzungen |
| mehrere_aussagen | HuggingFace classification | pytorch | Sequence classification (StaGE), `torch.save` format |
| personalpronomen | Custom PyTorch | pytorch | 6-head multi-label classifier, custom training loop with OneCycleLR scheduler |
| perplexity_saetze | NLTK n-gram | nltk | Kneser-Ney trigram model, CPU-only, deterministic, no P/R/F1 metrics |

Key differences:
- **HuggingFace Trainer** rules (abkuerzungen, zahlwoerter) use standard HF training args and produce safetensors/config.json outputs
- **Custom PyTorch** (personalpronomen) has its own training loop, saves `best_model.pt`, and trains 6 classification heads simultaneously
- **NLTK n-gram** (perplexity_saetze) ignores `--device` and `--seed` flags (CPU-only, deterministic interpolation), outputs a pickle file

## Device Management

The CLI auto-detects the best available compute device using this priority:

1. **CUDA** (NVIDIA GPU) -- if `torch.cuda.is_available()`
2. **MPS** (Apple Silicon) -- if `torch.backends.mps.is_available()` and built
3. **CPU** -- fallback

Override with `--device`:

```bash
python -m tools.ml train abkuerzungen --device cpu
python -m tools.ml train abkuerzungen --device mps
python -m tools.ml train abkuerzungen --device cuda
```

Requesting an unavailable device raises a `RuntimeError`.

The `--seed` flag (default: 42) sets seeds across `random`, `numpy`, and `torch` (CPU + CUDA + MPS) and enables deterministic CuDNN behavior for reproducible training runs.

## Troubleshooting

| Problem | Cause | Solution |
|---------|-------|----------|
| "Rule not found" error | Rule directory missing `train_config.py` | Create `train_config.py` with `METADATA`, `train()`, and `evaluate()` exports |
| "Device 'cuda' requested but not available" | CUDA not installed or no NVIDIA GPU | Use `--device cpu` or `--device mps` (Apple Silicon) |
| Model not loading after promote | API server still using old model in memory | Restart the API server (`uvicorn` / `python api_main.py`) |
| Empty staging directory error | Training did not complete or `.staging/` was cleaned | Re-run training: `python -m tools.ml train <rule>` |
| Missing training data | Data files not present under `regeln/<rule>/data/` | Check the rule's `train_config.py` for expected data paths and format |
| Training OOM (out of memory) | Batch size too large for available memory | Reduce `TRAIN_BATCH_SIZE` in the rule's `train_config.py`, or use `--device cpu` |
| "No staged model found" on promote | `--promote` used but `.staging/` is empty | Train without `--promote` first, verify `.staging/` has files, then re-train with `--promote` |
| Evaluation returns no metrics | `evaluate()` not implemented in `train_config.py` | Check that the rule's `train_config.py` exports an `evaluate()` function |
