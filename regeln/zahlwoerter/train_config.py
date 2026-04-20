"""Training adapter for the BERT-based German number word NER model.

Provides the CLI contract (METADATA, train, evaluate) for integration
with ``python -m tools.ml train zahlwoerter`` and ``evaluate zahlwoerter``.

The adapter is self-contained -- it does NOT import from other rule adapters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# CLI DISCOVERY METADATA
# ============================================================================

METADATA = {
    "model_type": "bert-token-classification",
    "framework": "transformers",
    "description": "BERT-based German number word detection (4 error types)",
}

# ============================================================================
# CONSTANTS
# ============================================================================

MODULE_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"

# Base model for fine-tuning
MODEL_CHECKPOINT = "dbmdz/bert-base-german-cased"

# HuggingFace Hub model (used as fallback for evaluate when no local model)
HUB_MODEL_ID = "fefeefef/leichte-sprache-zahlwoerter"

# BIO label set (4 error types + O)
LABEL_LIST = [
    "O",
    "B-BAD_WORD_NUM",
    "I-BAD_WORD_NUM",
    "B-BAD_YEAR",
    "I-BAD_YEAR",
    "B-BAD_PERCENT",
    "I-BAD_PERCENT",
    "B-BAD_COMPLEX_NUM",
    "I-BAD_COMPLEX_NUM",
]

# Training hyperparameters
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_LENGTH = 128

# ============================================================================
# HELPER: Label mappings
# ============================================================================

_LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}
_ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_dataset(records: list[dict]):
    """Build a HuggingFace Dataset from JSONL records."""
    from datasets import Dataset

    tokens_list = [r["tokens"] for r in records]
    labels_list = [r["labels"] for r in records]
    # Convert string labels to integer IDs
    label_ids = [
        [_LABEL2ID.get(lbl, 0) for lbl in labels]
        for labels in labels_list
    ]
    return Dataset.from_dict({"tokens": tokens_list, "ner_tags": label_ids})


def _tokenize_and_align(examples, tokenizer):
    """Tokenize pre-tokenized input and align NER labels to sub-tokens.

    Strategy: first sub-token gets the original label, rest get -100.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )
    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word = None
        for wid in word_ids:
            if wid is None:
                label_ids.append(-100)
            elif wid >= len(labels):
                label_ids.append(-100)
            elif wid != prev_word:
                label_ids.append(labels[wid])
            else:
                label_ids.append(-100)
            prev_word = wid
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


# ============================================================================
# CLI CONTRACT: train()
# ============================================================================


def train(output_dir: Path, device: str, seed: int) -> dict:
    """Train the zahlwoerter NER model via the unified CLI contract.

    Expects JSONL training data in ``regeln/zahlwoerter/data/``.
    Each line: ``{"tokens": [...], "labels": [...]}``.

    Args:
        output_dir: Directory for training output (staging area).
        device: Device string (cpu, cuda, mps).
        seed: Random seed for reproducibility.

    Returns:
        Dict with status, f1, precision, recall on completion.

    Raises:
        FileNotFoundError: When no training data exists, with format docs.
    """
    # --- Check for training data -------------------------------------------
    jsonl_files = list(DEFAULT_DATA_DIR.glob("*.jsonl")) if DEFAULT_DATA_DIR.exists() else []
    if not jsonl_files:
        raise FileNotFoundError(
            f"Training data not found at {DEFAULT_DATA_DIR}\n"
            f"\n"
            f"Expected format: JSONL file (one example per line) at {DEFAULT_DATA_DIR}/train.jsonl\n"
            f'Each line: {{"tokens": ["Das", "Kind", "ist", "acht", "Jahre", "alt", "."], '
            f'"labels": ["O", "O", "O", "B-BAD_WORD_NUM", "O", "O", "O"]}}\n'
            f"\n"
            f"Label set: {', '.join(LABEL_LIST)}"
        )

    # --- Lazy imports (avoid loading transformers at module level) ----------
    import numpy as np
    from datasets import DatasetDict
    from seqeval.metrics import f1_score, precision_score, recall_score
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    # --- Load data ---------------------------------------------------------
    all_records: list[dict] = []
    for f in sorted(jsonl_files):
        all_records.extend(_load_jsonl(f))

    logger.info("Loaded %d training examples from %d files", len(all_records), len(jsonl_files))

    # Split 80/10/10
    import random

    random.seed(seed)
    random.shuffle(all_records)
    n = len(all_records)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    train_ds = _build_dataset(all_records[:train_end])
    val_ds = _build_dataset(all_records[train_end:val_end])
    test_ds = _build_dataset(all_records[val_end:])

    datasets = DatasetDict({"train": train_ds, "validation": val_ds, "test": test_ds})

    # --- Load tokenizer and model ------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=_ID2LABEL,
        label2id=_LABEL2ID,
    )

    # --- Tokenize datasets -------------------------------------------------
    tokenized = datasets.map(
        lambda ex: _tokenize_and_align(ex, tokenizer),
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- Compute metrics callback ------------------------------------------
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=2)

        true_labels_list = []
        pred_labels_list = []
        for pred_row, label_row in zip(predictions, labels):
            p_seq, l_seq = [], []
            for p, lb in zip(pred_row, label_row):
                if lb == -100:
                    continue
                p_seq.append(_ID2LABEL.get(int(p), "O"))
                l_seq.append(_ID2LABEL.get(int(lb), "O"))
            if p_seq:
                pred_labels_list.append(p_seq)
                true_labels_list.append(l_seq)

        return {
            "precision": precision_score(true_labels_list, pred_labels_list),
            "recall": recall_score(true_labels_list, pred_labels_list),
            "f1": f1_score(true_labels_list, pred_labels_list),
        }

    # --- Training arguments ------------------------------------------------
    output_dir = Path(output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "training_output"),
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=seed,
        use_mps_device=(device == "mps"),
    )

    # --- Train -------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # --- Save best model ---------------------------------------------------
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save label mappings
    label_mappings = {
        "label_list": LABEL_LIST,
        "id2label": _ID2LABEL,
        "label2id": _LABEL2ID,
    }
    with open(output_dir / "label_mappings.json", "w", encoding="utf-8") as fh:
        json.dump(label_mappings, fh, indent=2, ensure_ascii=False)

    # --- Evaluate on test set ----------------------------------------------
    test_metrics = trainer.evaluate(tokenized["test"])
    f1 = test_metrics.get("eval_f1", 0.0)
    precision = test_metrics.get("eval_precision", 0.0)
    recall = test_metrics.get("eval_recall", 0.0)

    return {
        "status": "completed",
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ============================================================================
# CLI CONTRACT: evaluate()
# ============================================================================


def evaluate(model_dir: Path, **kwargs) -> dict:
    """Evaluate the zahlwoerter NER model on labeled data or verify model loads.

    Loads the model from *model_dir* (or falls back to Hub), runs seqeval
    evaluation if test data exists, otherwise returns model info.

    Args:
        model_dir: Path to the saved model directory.
        **kwargs:  Accepts optional ``data_path`` for a custom eval set.

    Returns:
        Dict with seqeval metrics or model-info summary.
    """
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        model_dir = Path(model_dir)

        # --- Determine model source ----------------------------------------
        has_local = model_dir.exists() and any(
            (model_dir / f).exists()
            for f in ("model.safetensors", "pytorch_model.bin", "config.json")
        )

        model_source = str(model_dir) if has_local else HUB_MODEL_ID
        if not has_local:
            logger.info("Local model not found at %s, falling back to Hub: %s", model_dir, HUB_MODEL_ID)

        # --- Load label mappings -------------------------------------------
        label_map_path = model_dir / "label_mappings.json" if has_local else None
        if label_map_path and label_map_path.exists():
            with open(label_map_path, encoding="utf-8") as fh:
                mappings = json.load(fh)
            label_list = mappings["label_list"]
            id2label = {int(k): v for k, v in mappings["id2label"].items()}
            label2id = mappings["label2id"]
        else:
            label_list = LABEL_LIST
            id2label = _ID2LABEL
            label2id = _LABEL2ID

        # --- Load model and tokenizer --------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_source)
        model = AutoModelForTokenClassification.from_pretrained(model_source)
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        # --- Check for eval data -------------------------------------------
        data_path = kwargs.get("data_path")
        if data_path is not None:
            data_path = Path(data_path)
        else:
            candidate = DEFAULT_DATA_DIR / "test.jsonl"
            if candidate.exists():
                data_path = candidate

        if data_path is None or not data_path.exists():
            return {
                "status": "completed",
                "model_loaded": True,
                "label_count": len(label_list),
                "note": "No labeled eval data available. Use test-suite for functional validation.",
            }

        # --- Tokenize eval data --------------------------------------------
        from datasets import Dataset
        from seqeval.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
            precision_score,
            recall_score,
        )
        from transformers import DataCollatorForTokenClassification

        records = _load_jsonl(data_path)
        tokens_list = [r["tokens"] for r in records]
        labels_list = [r["labels"] for r in records]
        label_ids = [[label2id.get(lbl, 0) for lbl in lbls] for lbls in labels_list]

        ds = Dataset.from_dict({"tokens": tokens_list, "ner_tags": label_ids})
        ds_tok = ds.map(
            lambda ex: _tokenize_and_align(ex, tokenizer),
            batched=True,
            remove_columns=ds.column_names,
        )
        ds_tok.set_format("torch")

        collator = DataCollatorForTokenClassification(tokenizer=tokenizer, return_tensors="pt")
        loader = torch.utils.data.DataLoader(ds_tok, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

        # --- Inference -----------------------------------------------------
        true_labels_all: list[list[str]] = []
        pred_labels_all: list[list[str]] = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = np.argmax(outputs.logits.cpu().numpy(), axis=2)
                labels_np = batch["labels"].cpu().numpy()

                for pred_row, label_row in zip(preds, labels_np):
                    p_seq, l_seq = [], []
                    for p, lb in zip(pred_row, label_row):
                        if lb == -100:
                            continue
                        p_seq.append(id2label.get(int(p), "O"))
                        l_seq.append(id2label.get(int(lb), "O"))
                    if p_seq:
                        pred_labels_all.append(p_seq)
                        true_labels_all.append(l_seq)

        # --- Compute metrics -----------------------------------------------
        precision = precision_score(true_labels_all, pred_labels_all)
        recall = recall_score(true_labels_all, pred_labels_all)
        f1 = f1_score(true_labels_all, pred_labels_all)
        accuracy = accuracy_score(true_labels_all, pred_labels_all)
        support = sum(len(s) for s in true_labels_all)
        report = classification_report(true_labels_all, pred_labels_all, digits=4)

        return {
            "status": "completed",
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "support": support,
            "classification_report": report,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
