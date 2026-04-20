"""
Configuration file for BERT German Abbreviation NER Training
Optimized for Apple M1 Mac with 64GB Unified Memory

Usage:
    python -m regeln.abkuerzungen_bert.train
"""

from pathlib import Path

# Module directory (where this file lives)
import logging

# METADATA for CLI discovery (tools/ml --list)
METADATA = {
    "model_type": "bert-token-classification",
    "framework": "transformers",
    "description": "BERT-based German abbreviation NER model",
}

logger = logging.getLogger(__name__)
MODULE_DIR = Path(__file__).parent

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
DATA_FILE = str(MODULE_DIR / "data" / "abbreviation_dataset_final.jsonl")
SEED = 42  # For reproducibility

# Train/Val/Test split ratios (80/10/10)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_CHECKPOINT = "bert-base-german-cased"

# Labels (will be validated and cleaned from data)
# NOTE: Initially the data only has O and B-ABK, but after B-I-O correction
# we will have O, B-ABK, and I-ABK
EXPECTED_LABELS = ["O", "B-ABK"]  # Before correction (for cleaning invalid labels)

# ============================================================================
# TRAINING HYPERPARAMETERS (M1-Optimized)
# ============================================================================
# Batch sizes optimized for M1 Unified Memory
TRAIN_BATCH_SIZE = 32  # Increased from 16 for M1 efficiency
EVAL_BATCH_SIZE = 32

# Learning rate and optimization
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1  # 10% of training steps for warmup

# Training duration
NUM_EPOCHS = 3

# Gradient settings
MAX_GRAD_NORM = 1.0
FP16 = False  # Will be auto-enabled for MPS/CUDA if available

# ============================================================================
# EARLY STOPPING CONFIGURATION
# ============================================================================
ENABLE_EARLY_STOPPING = True  # Set to False to disable
EARLY_STOPPING_PATIENCE = 2  # Stop if no improvement after N epochs
EARLY_STOPPING_THRESHOLD = 0.0001  # Minimum improvement considered significant

# ============================================================================
# CLASS WEIGHTS CONFIGURATION
# ============================================================================
USE_CLASS_WEIGHTS = True  # ENABLED: Critical for handling O/B-ABK/I-ABK imbalance
# If None, will be calculated automatically from training data
# After B-I-O correction, the imbalance between O, B-ABK, and I-ABK is significant
CLASS_WEIGHTS = None  # Auto-calculate from data

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================
OUTPUT_DIR = str(MODULE_DIR / "training_output")
FINAL_MODEL_DIR = str(MODULE_DIR / "model")
LOGS_DIR = str(MODULE_DIR / "training_logs")

# ============================================================================
# EVALUATION & LOGGING
# ============================================================================
EVALUATION_STRATEGY = "epoch"  # Evaluate after each epoch
SAVE_STRATEGY = "epoch"  # Save checkpoint after each epoch
LOGGING_STRATEGY = "steps"
LOGGING_STEPS = 50  # Log every N steps

LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "f1"  # Use F1-score to select best model
GREATER_IS_BETTER = True

# Save only the best model to save disk space
SAVE_TOTAL_LIMIT = 2

# ============================================================================
# HARDWARE OPTIMIZATION
# ============================================================================
# Auto-detect available hardware (MPS for M1, CUDA for NVIDIA, CPU fallback)
DATALOADER_NUM_WORKERS = 4  # Parallel data loading
DATALOADER_PIN_MEMORY = True

# ============================================================================
# TOKENIZATION
# ============================================================================
MAX_LENGTH = 128  # Maximum sequence length (most examples are ~9 tokens)
TRUNCATION = True
PADDING = "max_length"

# Label alignment strategy for WordPiece sub-tokens
# "first": First sub-token gets original label, rest get -100
# This is the recommended approach for NER tasks
LABEL_ALIGNMENT_STRATEGY = "first"


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
def set_seed(seed=SEED):
    """Set random seed for reproducibility across all libraries"""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_device():
    """Auto-detect best available device (MPS > CUDA > CPU)"""
    import torch

    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"  # Apple Silicon
    elif torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    else:
        return "cpu"


def print_config():
    """Print current configuration"""
    logger.info("=" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 80)
    logger.info("Model: %s", MODEL_CHECKPOINT)
    logger.info("Data: %s", DATA_FILE)
    logger.info("Device: %s", get_device())
    logger.info("Seed: %s", SEED)
    logger.info("\nTraining:")
    logger.info(" Batch Size (Train/Eval): %s/%s", TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE)
    logger.info(" Learning Rate: %s", LEARNING_RATE)
    logger.info(" Epochs: %s", NUM_EPOCHS)
    logger.info(" Weight Decay: %s", WEIGHT_DECAY)
    logger.info("\nOptimization:")
    logger.info(" Early Stopping: %s (patience=%s)", ENABLE_EARLY_STOPPING, EARLY_STOPPING_PATIENCE)
    logger.info(" Early Stopping: %s (patience=%s)", ENABLE_EARLY_STOPPING, EARLY_STOPPING_PATIENCE)
    logger.info(" Class Weights: %s", USE_CLASS_WEIGHTS)
    logger.info(" Best Model Metric: %s", METRIC_FOR_BEST_MODEL)
    logger.info("\nOutput:")
    logger.info(" Model Output: %s", OUTPUT_DIR)
    logger.info(" Final Model: %s", FINAL_MODEL_DIR)
    logger.info("=" * 80)


# ============================================================================
# CLI CONTRACT FUNCTIONS
# ============================================================================
def train(output_dir: Path, device: str, seed: int) -> dict:
    """Train the abbreviation NER model via the CLI contract.

    Delegates to the existing train.py main() function, temporarily
    overriding module-level config constants for the staging workflow.

    Args:
        output_dir: Directory for training output (staging area).
        device: Device string (cpu, cuda, mps).
        seed: Random seed for reproducibility.

    Returns:
        Dict with at least a "status" key.
    """
    # Lazy import to avoid loading transformers at module level
    from regeln.abkuerzungen.train import main as run_training

    import regeln.abkuerzungen.train_config as cfg

    original_output = cfg.OUTPUT_DIR
    original_final = cfg.FINAL_MODEL_DIR
    original_seed = cfg.SEED
    original_get_device = cfg.get_device
    try:
        cfg.OUTPUT_DIR = str(output_dir / "training_output")
        cfg.FINAL_MODEL_DIR = str(output_dir)
        cfg.SEED = seed
        cfg.get_device = lambda: device  # Pass through CLI device
        result = run_training()
        return result if isinstance(result, dict) else {"status": "completed"}
    finally:
        cfg.OUTPUT_DIR = original_output
        cfg.FINAL_MODEL_DIR = original_final
        cfg.SEED = original_seed
        cfg.get_device = original_get_device


def evaluate(model_dir: Path, **kwargs) -> dict:
    """Evaluate the abbreviation NER model on the held-out test split.

    Loads the model from *model_dir*, reproduces the same train/test split
    used during training (80/10/10, seed 42), runs inference on the test
    set, and returns aggregate + per-class NER metrics via seqeval.

    Args:
        model_dir: Path to the saved model directory (must contain
                   model.safetensors or pytorch_model.bin, tokenizer
                   files, and label_mappings.json).
        **kwargs:  Accepts optional ``data_path`` for a custom eval set.

    Returns:
        Dict with precision, recall, f1, accuracy, support, and
        a ``classification_report`` string for ``--verbose`` output.
    """
    try:
        import json as _json

        import numpy as _np
        import torch as _torch
        from datasets import load_dataset as _load_dataset
        from seqeval.metrics import (
            accuracy_score as _acc,
            classification_report as _cls_report,
            f1_score as _f1,
            precision_score as _prec,
            recall_score as _rec,
        )
        from transformers import (
            AutoModelForTokenClassification as _AutoModel,
            AutoTokenizer as _AutoTok,
            DataCollatorForTokenClassification as _Collator,
        )

        # --- resolve paths ---------------------------------------------------
        model_dir = Path(model_dir)
        data_path = kwargs.get("data_path", Path(DATA_FILE))

        if not model_dir.exists():
            return {"status": "error", "message": f"Model directory not found: {model_dir}"}

        label_map_path = model_dir / "label_mappings.json"
        if not label_map_path.exists():
            return {"status": "error", "message": f"label_mappings.json not found in {model_dir}"}

        with open(label_map_path) as fh:
            mappings = _json.load(fh)
        label_list = mappings["label_list"]
        id2label = {int(k): v for k, v in mappings["id2label"].items()}
        label2id = mappings["label2id"]

        # --- load model + tokenizer ------------------------------------------
        tokenizer = _AutoTok.from_pretrained(str(model_dir))
        model = _AutoModel.from_pretrained(str(model_dir))
        device = _torch.device("cpu")
        model.to(device)
        model.eval()

        # --- reproduce test split (same logic as train.py) --------------------
        from regeln.abkuerzungen.train import correct_bio_labels, load_and_clean_data, split_dataset

        dataset = load_and_clean_data(str(data_path))
        splits = split_dataset(dataset)
        test_ds = splits["test"]

        # --- tokenize with label alignment ------------------------------------
        def _align_labels(examples):
            tokenized = tokenizer(
                examples["tokens"],
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                is_split_into_words=True,
            )
            all_labels = []
            for i, labels in enumerate(examples["labels"]):
                word_ids = tokenized.word_ids(batch_index=i)
                label_ids = []
                prev_word = None
                for wid in word_ids:
                    if wid is None:
                        label_ids.append(-100)
                    elif wid >= len(labels):
                        # Data quality: token/label length mismatch
                        label_ids.append(-100)
                    elif wid != prev_word:
                        label_ids.append(label2id.get(labels[wid], 0))
                    else:
                        label_ids.append(-100)
                    prev_word = wid
                all_labels.append(label_ids)
            tokenized["labels"] = all_labels
            return tokenized

        test_tok = test_ds.map(_align_labels, batched=True, remove_columns=test_ds.column_names)
        test_tok.set_format("torch")

        collator = _Collator(tokenizer=tokenizer, return_tensors="pt")
        loader = _torch.utils.data.DataLoader(test_tok, batch_size=EVAL_BATCH_SIZE, collate_fn=collator)

        # --- inference --------------------------------------------------------
        true_labels = []
        true_preds = []

        with _torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = _np.argmax(outputs.logits.cpu().numpy(), axis=2)
                labels_np = batch["labels"].cpu().numpy()

                for pred_row, label_row in zip(preds, labels_np):
                    p_seq, l_seq = [], []
                    for p, lb in zip(pred_row, label_row):
                        if lb == -100:
                            continue
                        p_seq.append(id2label.get(int(p), "O"))
                        l_seq.append(id2label.get(int(lb), "O"))
                    if p_seq:
                        true_preds.append(p_seq)
                        true_labels.append(l_seq)

        # --- compute metrics --------------------------------------------------
        precision = _prec(true_labels, true_preds)
        recall = _rec(true_labels, true_preds)
        f1 = _f1(true_labels, true_preds)
        accuracy = _acc(true_labels, true_preds)
        support = sum(len(s) for s in true_labels)
        report = _cls_report(true_labels, true_preds, digits=4)

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
