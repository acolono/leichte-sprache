"""Training adapter for personalpronomen (multi-head pronoun classifier).

Custom PyTorch training loop for the 6-head pronoun classifier
(person, gender, number, case, polite, reflex).

Usage:
    python -m tools.ml train personalpronomen [--device auto] [--seed 42]
    python -m tools.ml evaluate personalpronomen [--verbose]
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

from .pronoun_model import ModelConfig, create_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI discovery metadata
# ---------------------------------------------------------------------------

METADATA = {
    "model_type": "pytorch-custom-multihead",
    "framework": "pytorch",
    "description": "Custom 6-head pronoun classifier (person, gender, number, case, polite, reflex)",
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODULE_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = MODULE_DIR / "data"
HEADS = ["person", "gender", "number", "case", "polite", "reflex"]
TOKENIZER_NAME = "bert-base-german-cased"
MAX_LENGTH = 128


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PronounDataset(Dataset):
    """Dataset for pronoun classification training.

    Expects JSONL files where each line is a JSON object:
    ``{"text": "Er ging nach Hause.", "pronoun_index": 0,
       "labels": {"person": "3", "gender": "Masc", "number": "Sing",
                  "case": "Nom", "polite": "NONE", "reflex": "false"}}``

    ``pronoun_index`` is the **word** index (0-based) of the pronoun in
    the whitespace-split text.
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: AutoTokenizer,
        label_encoders: dict,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.tokenizer = tokenizer
        self.label_encoders = label_encoders
        self.max_length = max_length
        self.samples: List[dict] = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text = sample["text"]
        pronoun_index = sample["pronoun_index"]
        labels = sample["labels"]

        # Tokenize with offset mapping so we can locate the pronoun tokens.
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

        # Build pronoun_mask: map word-level pronoun_index -> token positions.
        words = text.split()
        if pronoun_index < len(words):
            char_start = sum(len(w) + 1 for w in words[:pronoun_index])
            char_end = char_start + len(words[pronoun_index])
        else:
            # Fallback: treat first word as pronoun
            char_start = 0
            char_end = len(words[0]) if words else 1

        pronoun_mask = torch.zeros(self.max_length, dtype=torch.float32)
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start is not None and end is not None and end > 0:
                if start >= char_start and end <= char_end + 1:
                    pronoun_mask[token_idx] = 1.0

        # Fallback: if no token matched, use approximate position
        if pronoun_mask.sum() == 0:
            approx_pos = min(pronoun_index + 1, self.max_length - 2)
            pronoun_mask[approx_pos] = 1.0

        # Encode labels
        item: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pronoun_mask": pronoun_mask,
        }
        for head in HEADS:
            label_str = str(labels.get(head, "NONE"))
            label_to_id = self.label_encoders[head]["label_to_id"]
            label_id = label_to_id.get(label_str, 0)
            item[f"label_{head}"] = torch.tensor(label_id, dtype=torch.long)

        return item


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------


def train(output_dir: Path, device: str, seed: int) -> dict:
    """Train the 6-head pronoun classifier via the CLI contract.

    Expects JSONL training data at ``regeln/personalpronomen/data/train.jsonl``.
    Raises ``FileNotFoundError`` with documented data format when absent.

    Args:
        output_dir: Staging directory for training output.
        device: Device string (cpu, cuda, mps).
        seed: Random seed for reproducibility.

    Returns:
        Dict with training metrics.
    """
    # ------------------------------------------------------------------
    # 1. Check for training data
    # ------------------------------------------------------------------
    train_file = DEFAULT_DATA_DIR / "train.jsonl"
    has_data = DEFAULT_DATA_DIR.exists() and any(DEFAULT_DATA_DIR.glob("*.jsonl"))

    if not has_data:
        raise FileNotFoundError(
            f"Training data not found at {DEFAULT_DATA_DIR}\n"
            "\n"
            f"Expected format: JSONL file (one example per line) at {DEFAULT_DATA_DIR}/train.jsonl\n"
            'Each line: {"text": "Er ging nach Hause.", "pronoun_index": 0, '
            '"labels": {"person": "3", "gender": "Masc", "number": "Sing", '
            '"case": "Nom", "polite": "NONE", "reflex": "false"}}\n'
            "\n"
            "Label values (from label_encoders.json):\n"
            "  person:  NONE, 1, 2, 3\n"
            "  gender:  NONE, Fem, Masc, Neut\n"
            "  number:  NONE, Plur, Sing\n"
            "  case:    NONE, Acc, Dat, Nom\n"
            "  polite:  NONE, Form\n"
            "  reflex:  false, true"
        )

    # ------------------------------------------------------------------
    # 2. Load label encoders
    # ------------------------------------------------------------------
    label_encoders_path = MODULE_DIR / "model" / "label_encoders.json"
    with open(label_encoders_path, "r", encoding="utf-8") as f:
        label_encoders = json.load(f)

    # ------------------------------------------------------------------
    # 3. Load tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # ------------------------------------------------------------------
    # 4. Create dataset with 80/20 train/val split
    # ------------------------------------------------------------------
    # Use train.jsonl if it exists, otherwise first .jsonl found
    if train_file.exists():
        data_path = train_file
    else:
        data_path = next(DEFAULT_DATA_DIR.glob("*.jsonl"))

    full_dataset = PronounDataset(data_path, tokenizer, label_encoders, MAX_LENGTH)

    val_size = max(1, int(len(full_dataset) * 0.2))
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    # ------------------------------------------------------------------
    # 5. Create DataLoaders
    # ------------------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # ------------------------------------------------------------------
    # 6. Create model
    # ------------------------------------------------------------------
    config = ModelConfig(
        num_person_classes=label_encoders["person"]["num_classes"],
        num_gender_classes=label_encoders["gender"]["num_classes"],
        num_number_classes=label_encoders["number"]["num_classes"],
        num_case_classes=label_encoders["case"]["num_classes"],
        num_polite_classes=label_encoders["polite"]["num_classes"],
        num_reflex_classes=label_encoders["reflex"]["num_classes"],
    )
    model = create_model(config)
    model.to(device)
    model.train()

    # ------------------------------------------------------------------
    # 7. Optimizer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # ------------------------------------------------------------------
    # 8. Scheduler: linear warmup + decay via OneCycleLR
    # ------------------------------------------------------------------
    num_epochs = 10
    total_steps = num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=2e-5,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="linear",
    )

    # ------------------------------------------------------------------
    # 9. Training loop
    # ------------------------------------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_accuracy: Optional[float] = None
    best_train_loss: Optional[float] = None
    final_loss = 0.0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pronoun_mask = batch["pronoun_mask"].to(device)

            logits = model(input_ids, attention_mask, pronoun_mask)

            # Per-head cross-entropy loss, summed
            loss = torch.tensor(0.0, device=device)
            for head in HEADS:
                head_logits = logits[head]
                head_labels = batch[f"label_{head}"].to(device)
                loss = loss + F.cross_entropy(head_logits, head_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        final_loss = avg_epoch_loss

        # Validation accuracy
        val_accuracy = _compute_val_accuracy(model, val_loader, device)
        logger.info(
            "Epoch %d/%d  loss=%.4f  val_accuracy=%.4f",
            epoch + 1, num_epochs, avg_epoch_loss, val_accuracy,
        )

        # Save best model by val accuracy
        is_best = best_val_accuracy is None or val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    # ------------------------------------------------------------------
    # 10. Copy label_encoders.json to output
    # ------------------------------------------------------------------
    shutil.copy2(label_encoders_path, output_dir / "label_encoders.json")

    return {
        "status": "completed",
        "epochs": num_epochs,
        "final_loss": final_loss,
        "best_val_accuracy": best_val_accuracy,
    }


def _compute_val_accuracy(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: str,
) -> float:
    """Compute average accuracy across all 6 heads on the validation set."""
    model.eval()
    correct_per_head = {head: 0 for head in HEADS}
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pronoun_mask = batch["pronoun_mask"].to(device)
            batch_size = input_ids.size(0)
            total += batch_size

            logits = model(input_ids, attention_mask, pronoun_mask)

            for head in HEADS:
                preds = torch.argmax(logits[head], dim=-1)
                labels = batch[f"label_{head}"].to(device)
                correct_per_head[head] += (preds == labels).sum().item()

    if total == 0:
        return 0.0

    # Average accuracy across heads
    per_head_acc = [correct_per_head[h] / total for h in HEADS]
    return sum(per_head_acc) / len(per_head_acc)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def evaluate(model_dir: Path, **kwargs) -> dict:
    """Evaluate the deployed pronoun classifier model.

    Loads the model from ``model_dir/best_model.pt`` and, if test data
    exists, computes per-head accuracy and macro F1. Otherwise returns
    model metadata.

    Args:
        model_dir: Path to model directory containing best_model.pt.
        **kwargs:  Accepts optional ``data_path`` for a custom test JSONL.

    Returns:
        Dict with evaluation metrics and a ``classification_report`` string.
    """
    try:
        model_dir = Path(model_dir)

        # Locate model weights
        model_path = model_dir / "best_model.pt"
        if not model_path.exists():
            return {
                "status": "error",
                "message": f"Model not found at {model_path}",
            }

        # Load label encoders (try model_dir first, fall back to deployed)
        le_path = model_dir / "label_encoders.json"
        if not le_path.exists():
            le_path = MODULE_DIR / "model" / "label_encoders.json"

        with open(le_path, "r", encoding="utf-8") as f:
            label_encoders = json.load(f)

        # Create model with correct config from label encoders
        config = ModelConfig(
            num_person_classes=label_encoders["person"]["num_classes"],
            num_gender_classes=label_encoders["gender"]["num_classes"],
            num_number_classes=label_encoders["number"]["num_classes"],
            num_case_classes=label_encoders["case"]["num_classes"],
            num_polite_classes=label_encoders["polite"]["num_classes"],
            num_reflex_classes=label_encoders["reflex"]["num_classes"],
        )
        model = create_model(config)
        model.load_state_dict(
            torch.load(str(model_path), map_location="cpu", weights_only=True),
            strict=True,
        )
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())

        # Check for eval data
        data_path = kwargs.get("data_path")
        if data_path is not None:
            test_path = Path(data_path)
        else:
            test_path = DEFAULT_DATA_DIR / "test.jsonl"

        if test_path is not None and test_path.exists():
            return _evaluate_with_data(model, test_path, label_encoders, param_count)

        # No eval data -- return model info
        return {
            "status": "completed",
            "model_loaded": True,
            "heads": len(HEADS),
            "parameters": param_count,
            "note": "No labeled eval data available. Use test-suite for functional validation.",
            "classification_report": "N/A -- no labeled eval data. Use test-suite for functional validation.",
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _evaluate_with_data(
    model: torch.nn.Module,
    test_path: Path,
    label_encoders: dict,
    param_count: int,
) -> dict:
    """Run evaluation on labeled JSONL test data with per-head metrics."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = PronounDataset(test_path, tokenizer, label_encoders, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds: Dict[str, list] = {h: [] for h in HEADS}
    all_labels: Dict[str, list] = {h: [] for h in HEADS}

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            pronoun_mask = batch["pronoun_mask"]

            logits = model(input_ids, attention_mask, pronoun_mask)

            for head in HEADS:
                preds = torch.argmax(logits[head], dim=-1).tolist()
                labels = batch[f"label_{head}"].tolist()
                all_preds[head].extend(preds)
                all_labels[head].extend(labels)

    # Per-head accuracy and macro F1
    head_metrics = {}
    for head in HEADS:
        acc = accuracy_score(all_labels[head], all_preds[head])
        f1 = f1_score(all_labels[head], all_preds[head], average="macro", zero_division=0)
        head_metrics[head] = {"accuracy": acc, "f1": f1}

    avg_accuracy = sum(m["accuracy"] for m in head_metrics.values()) / len(HEADS)
    avg_f1 = sum(m["f1"] for m in head_metrics.values()) / len(HEADS)

    # Build classification report string
    report_lines = ["Per-Head Metrics:", ""]
    for head in HEADS:
        m = head_metrics[head]
        report_lines.append(f"  {head:8s}  accuracy={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    report_str = "\n".join(report_lines)

    return {
        "status": "completed",
        "model_loaded": True,
        "heads": len(HEADS),
        "parameters": param_count,
        "avg_accuracy": avg_accuracy,
        "avg_f1": avg_f1,
        "per_head": head_metrics,
        "classification_report": report_str,
    }
