"""
BERT Training Script für StaGE Statement Segmentation
Trainiert einen deutschen BERT-Classifier für Multi-Statement-Erkennung.

Autor: Senior Data Scientist
"""

logger = logging.getLogger(__name__)
import logging

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


class StaGEDataset(Dataset):
    """PyTorch Dataset für StaGE Statement Segmentation."""

    def __init__(
        self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class StaGEModelTrainer:
    """Trainer-Klasse für BERT-basierte Statement-Klassifikation."""

    def __init__(
        self,
        model_name: str = "bert-base-german-cased",
        num_labels: int = 2,
        output_dir: str = "../models",
        max_length: int = 128,
    ):
        """
        Args:
            model_name: Name des vortrainierten Modells (Hugging Face)
            num_labels: Anzahl der Klassen (2 für binary, 4 für multiclass)
            output_dir: Verzeichnis für Modell-Checkpoints
            max_length: Maximale Sequenzlänge
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = Path(output_dir)
        self.max_length = max_length

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(" Device: %s", self.device)

        # Initialisiere Tokenizer und Model
        logger.info(" Lade Modell: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def load_data(self, data_dir: str = "../data") -> Dict[str, pd.DataFrame]:
        """Lädt vorverarbeitete Daten."""
        data_path = Path(data_dir)

        logger.info("\n Lade Daten von: %s", data_path)

        data = {
            "train": pd.read_csv(data_path / "train_processed.csv"),
            "val": pd.read_csv(data_path / "val_processed.csv"),
            "test": pd.read_csv(data_path / "test_processed.csv"),
        }

        for split, df in data.items():
            logger.info(" %s: %s samples", split, len(df))

        return data

    def create_datasets(
        self,
        data: Dict[str, pd.DataFrame],
        label_column: str = "has_multiple_statements",
    ) -> Dict[str, Dataset]:
        """Erstellt PyTorch Datasets."""
        logger.info("\n Erstelle PyTorch Datasets (Label: %s)", label_column)

        datasets = {}
        for split, df in data.items():
            texts = df["phrase"].tolist()
            labels = df[label_column].tolist()

            datasets[split] = StaGEDataset(
                texts=texts,
                labels=labels,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

            logger.info(" %s: %s samples", split, len(datasets[split]))

        return datasets

    def compute_metrics(self, pred):
        """Berechnet Metriken für Evaluation."""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="weighted"
        )
        acc = accuracy_score(labels, preds)

        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    def train(
        self,
        datasets: Dict[str, Dataset],
        num_epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
    ):
        """Trainiert das Modell."""
        logger.info("\n Starte Training...")
        logger.info("=" * 60)

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None,  # Kein W&B oder TensorBoard
        )

        # Trainer mit Early Stopping
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["val"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Training
        logger.info(" Hyperparameters:")
        logger.info(" Epochs: %s", num_epochs)
        logger.info(" Batch Size: %s", batch_size)
        logger.info(" Learning Rate: %s", learning_rate)
        logger.info(" Warmup Steps: %s", warmup_steps)
        logger.info(" Weight Decay: %s", weight_decay)
        logger.info("=" * 60)

        trainer.train()

        logger.info("\n Training abgeschlossen!")

        return trainer

    def evaluate(self, trainer: Trainer, dataset: Dataset, split_name: str = "test"):
        """Evaluiert das Modell auf einem Dataset."""
        logger.info("\n Evaluiere auf %s Set...", split_name)

        results = trainer.evaluate(dataset)

        logger.info("\n %s RESULTS:", split_name.upper())
        logger.info("=" * 60)
        for key, value in results.items():
            if key.startswith("eval_"):
                metric_name = key.replace("eval_", "")
                logger.info(" %s: %.4f", metric_name, value)
        logger.info("=" * 60)

        return results

    def save_model(self, output_name: str = "stage_statement_classifier"):
        """Speichert das trainierte Modell."""
        save_path = self.output_dir / output_name
        save_path.mkdir(exist_ok=True, parents=True)

        logger.info("\n Speichere Modell: %s", save_path)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Speichere Konfiguration
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(" Modell gespeichert!")

    def predict(self, texts: List[str]) -> np.ndarray:
        """Macht Vorhersagen auf neuen Texten."""
        self.model.eval()
        self.model.to(self.device)

        predictions = []

        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )

                input_ids = encoding["input_ids"].to(self.device)
                attention_mask = encoding["attention_mask"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1).cpu().numpy()[0]

                predictions.append(pred)

        return np.array(predictions)


def main():
    """Hauptfunktion für Modell-Training."""
    logger.info(" StaGE Statement Segmentation - BERT Model Training")
    logger.info("=" * 60)

    # Konfiguration
    CONFIG = {
        "model_name": "bert-base-german-cased",  # Deutsches BERT-Modell
        "num_labels": 2,  # Binary Classification: 1 vs. mehrere Statements
        "max_length": 128,  # Sätze in Leichter Sprache sind kurz
        "num_epochs": 5,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "warmup_steps": 500,
        "weight_decay": 0.01,
    }

    logger.info("⚙ Konfiguration:")
    for key, value in CONFIG.items():
        logger.info(" %s: %s", key, value)
    logger.info("=" * 60)

    # Initialisiere Trainer
    trainer_obj = StaGEModelTrainer(
        model_name=CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
        max_length=CONFIG["max_length"],
    )

    # Lade Daten
    data = trainer_obj.load_data()

    # Erstelle Datasets
    datasets = trainer_obj.create_datasets(data, label_column="has_multiple_statements")

    # Training
    trainer = trainer_obj.train(
        datasets=datasets,
        num_epochs=CONFIG["num_epochs"],
        batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=CONFIG["warmup_steps"],
        weight_decay=CONFIG["weight_decay"],
    )

    # Evaluation
    trainer_obj.evaluate(trainer, datasets["val"], split_name="validation")
    trainer_obj.evaluate(trainer, datasets["test"], split_name="test")

    # Speichere Modell
    trainer_obj.save_model("stage_statement_classifier")

    # Test Predictions
    logger.info("\n Test Predictions:")
    logger.info("=" * 60)

    test_texts = [
        "Das Haus ist groß.",  # 1 Statement
        "Das Haus ist groß und schön.",  # 2 Statements (potentiell)
        "Der Mann geht in den Park und kauft ein Eis.",  # 2 Statements
        "Es regnet heute.",  # 1 Statement
        "Maria arbeitet im Büro, sie ist sehr fleißig und macht ihre Arbeit gut.",  # 3 Statements
    ]

    predictions = trainer_obj.predict(test_texts)

    for text, pred in zip(test_texts, predictions, strict=False):
        label = "Mehrere Statements" if pred == 1 else "Ein Statement"
        logger.info(" '%s'", text)
        logger.info(" → %s (Klasse: %s)\n", label, pred)

    logger.info("=" * 60)
    logger.info(" Training und Evaluation abgeschlossen!")
    logger.info(" Modell gespeichert in: %s", trainer_obj.output_dir / 'stage_statement_classifier')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
