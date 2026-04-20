"""Training adapter for mehrere_aussagen (StaGE statement segmentation).

Wraps the manual PyTorch training loop from train_model_simple.py
to conform to the CLI contract (tools/ml train <rule>).

Usage:
    python -m tools.ml train mehrere_aussagen [--device auto] [--seed 42]
"""

from pathlib import Path

# METADATA for CLI discovery (tools/ml --list)
METADATA = {
    "model_type": "bert-sequence-classification",
    "framework": "pytorch",
    "description": "BERT statement segmentation classifier (StaGE)",
}

MODULE_DIR = Path(__file__).parent


def train(output_dir: Path, device: str, seed: int) -> dict:
    """Train the StaGE statement classifier via the CLI contract.

    Wraps StaGEModelTrainer from train_model_simple.py directly
    (not main(), which uses argparse and hardcoded paths).

    Args:
        output_dir: Staging directory for training output.
        device: Device string (cpu, cuda, mps).
        seed: Random seed for reproducibility.

    Returns:
        Dict with training metrics.
    """
    import torch
    from regeln.mehrere_aussagen.model.scripts.train_model_simple import (
        StaGEModelTrainer,
    )

    # Reproducibility
    torch.manual_seed(seed)

    # Validate training data exists
    data_dir = MODULE_DIR / "model" / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data not found: {data_dir}")

    # CRITICAL: output_dir from CLI is rule_dir/model/.staging/
    # After promote, .staging/* copies to model/.
    # regel.py loads from model/models/stage_statement_classifier_best/
    # So trainer output_dir must be output_dir/"models" so that
    # save_model("stage_statement_classifier_best") creates:
    #   output_dir/models/stage_statement_classifier_best/pytorch_model.bin
    # After promote this becomes:
    #   model/models/stage_statement_classifier_best/pytorch_model.bin
    trainer = StaGEModelTrainer(
        model_name="bert-base-german-cased",
        num_labels=2,
        output_dir=str(output_dir / "models"),
        max_length=128,
    )

    # Override auto-detected device with CLI-provided device
    trainer.device = torch.device(device)
    trainer.model.to(trainer.device)

    # Load pre-split CSV data
    data = trainer.load_data(data_dir=str(data_dir))

    # Create dataloaders
    dataloaders = trainer.create_dataloaders(data, batch_size=16)

    # Train
    train_history, val_history = trainer.train(
        dataloaders=dataloaders, num_epochs=5, learning_rate=2e-5
    )

    # Extract final validation metrics
    final_metrics = val_history[-1] if val_history else {}
    return {
        "status": "completed",
        "f1": final_metrics.get("f1", 0.0),
        "accuracy": final_metrics.get("accuracy", 0.0),
    }


def evaluate(model_dir: Path, **kwargs) -> dict:
    """Evaluate the StaGE statement classifier on the test split.

    Loads the saved model weights, runs inference on test_processed.csv,
    and returns aggregate P/R/F1 plus a classification report.

    Args:
        model_dir: Path to model directory (parent of ``models/``).
        **kwargs:  Accepts optional ``data_path`` for a custom test CSV.

    Returns:
        Dict with precision, recall, f1, accuracy, loss, and
        ``classification_report`` string.
    """
    try:
        import torch
        from sklearn.metrics import classification_report as _cls_report
        from regeln.mehrere_aussagen.model.scripts.train_model_simple import (
            StaGEModelTrainer,
        )

        model_dir = Path(model_dir)

        # Locate saved weights -- try two conventions:
        # 1) model_dir/models/stage_statement_classifier_best/pytorch_model.bin
        # 2) model_dir/stage_statement_classifier_best/pytorch_model.bin (staging)
        weight_candidates = [
            model_dir / "models" / "stage_statement_classifier_best" / "pytorch_model.bin",
            model_dir / "stage_statement_classifier_best" / "pytorch_model.bin",
        ]
        weight_path = None
        for candidate in weight_candidates:
            if candidate.exists():
                weight_path = candidate
                break

        if weight_path is None:
            return {
                "status": "error",
                "message": f"No pytorch_model.bin found in {model_dir}. Tried: {[str(c) for c in weight_candidates]}",
            }

        # Locate test data
        data_path = kwargs.get("data_path")
        if data_path is not None:
            test_csv = Path(data_path)
        else:
            test_csv = MODULE_DIR / "model" / "data" / "test_processed.csv"

        if not test_csv.exists():
            return {"status": "error", "message": f"Test data not found: {test_csv}"}

        # Load checkpoint to read config
        checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        model_name = checkpoint.get("model_name", "bert-base-german-cased")
        num_labels = checkpoint.get("num_labels", 2)
        max_length = checkpoint.get("max_length", 128)

        # Instantiate trainer and load weights
        trainer = StaGEModelTrainer(
            model_name=model_name,
            num_labels=num_labels,
            output_dir=str(model_dir),
            max_length=max_length,
        )
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.device = torch.device("cpu")
        trainer.model.to(trainer.device)
        trainer.model.eval()

        # Load test data and create dataloader
        import pandas as _pd
        from regeln.mehrere_aussagen.model.scripts.train_model_simple import StaGEDataset
        from torch.utils.data import DataLoader

        test_df = _pd.read_csv(test_csv)
        test_dataset = StaGEDataset(
            texts=test_df["phrase"].tolist(),
            labels=test_df["has_multiple_statements"].tolist(),
            tokenizer=trainer.tokenizer,
            max_length=max_length,
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

        # Run evaluation using trainer's evaluate method
        metrics = trainer.evaluate(test_loader)

        # Build classification report for --verbose
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(trainer.device)
                attention_mask = batch["attention_mask"].to(trainer.device)
                labels = batch["label"].to(trainer.device)
                logits = trainer.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        target_names = ["single_statement", "multiple_statements"]
        report = _cls_report(all_labels, all_preds, target_names=target_names, digits=4)

        return {
            "status": "completed",
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "loss": metrics["loss"],
            "classification_report": report,
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc)}
