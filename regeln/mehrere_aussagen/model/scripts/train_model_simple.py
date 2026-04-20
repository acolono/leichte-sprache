"""
Vereinfachter BERT Training Script für StaGE Statement Segmentation
Verwendet manuelles Training-Loop (kein Huggingface Trainer) für bessere Kompatibilität.

Autor: Senior Data Scientist
"""

import logging

logger = logging.getLogger(__name__)

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Unterdrücke TensorFlow Warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class StaGEDataset(Dataset):
    """PyTorch Dataset für StaGE Statement Segmentation."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            "label": torch.tensor(label, dtype=torch.long),
        }


class BERTStatementClassifier(nn.Module):
    """BERT-basierter Classifier für Statement-Segmentation."""

    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class StaGEModelTrainer:
    """Trainer-Klasse mit manuellem Training-Loop."""

    def __init__(
        self,
        model_name="bert-base-german-cased",
        num_labels=2,
        output_dir="../models",
        max_length=128,
    ):
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
        self.model = BERTStatementClassifier(model_name, num_labels=num_labels)
        self.model.to(self.device)

        # Training state
        self.best_val_f1 = 0.0

    def load_data(self, data_dir="../data"):
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

    def create_dataloaders(
        self, data, batch_size=16, label_column="has_multiple_statements"
    ):
        """Erstellt DataLoaders."""
        logger.info("\n Erstelle DataLoaders (Label: %s)", label_column)

        dataloaders = {}
        for split, df in data.items():
            dataset = StaGEDataset(
                texts=df["phrase"].tolist(),
                labels=df[label_column].tolist(),
                tokenizer=self.tokenizer,
                max_length=self.max_length,
            )

            shuffle = split == "train"
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,  # Für Kompatibilität
            )

            logger.info(" %s: %s samples, %s batches", split, len(dataset), len(dataloaders[split]))

        return dataloaders

    def train_epoch(self, dataloader, optimizer, criterion):
        """Trainiert eine Epoche."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, dataloader):
        """Evaluiert das Modell."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = criterion(logits, labels)

                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        # Metriken
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self, dataloaders, num_epochs=5, learning_rate=2e-5):
        """Trainiert das Modell."""
        logger.info("\n Starte Training...")
        logger.info("=" * 60)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_history = []
        val_history = []

        for epoch in range(num_epochs):
            logger.info("\n Epoch %s/%s", epoch + 1, num_epochs)
            logger.info("-" * 60)

            # Training
            train_loss, train_acc = self.train_epoch(
                dataloaders["train"], optimizer, criterion
            )
            logger.info(" Train Loss: %.4f | Train Acc: %.4f", train_loss, train_acc)

            # Validation
            val_metrics = self.evaluate(dataloaders["val"])
            logger.info(" Val Loss: %.4f | Val Acc: %.4f", val_metrics['loss'], val_metrics['accuracy'])
            logger.info(" Val F1: %.4f | Val Precision: %.4f | Val Recall: %.4f", val_metrics['f1'], val_metrics['precision'], val_metrics['recall'])

            train_history.append({"loss": train_loss, "accuracy": train_acc})
            val_history.append(val_metrics)

            # Save best model
            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.save_model("stage_statement_classifier_best")
                logger.info("  Neues bestes Modell gespeichert! (F1: %.4f)", self.best_val_f1)

        logger.info("\n Training abgeschlossen!")
        return train_history, val_history

    def save_model(self, output_name="stage_statement_classifier"):
        """Speichert das trainierte Modell."""
        save_path = self.output_dir / output_name
        save_path.mkdir(exist_ok=True, parents=True)

        # Speichere Modell
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "max_length": self.max_length,
            },
            save_path / "pytorch_model.bin",
        )

        # Speichere Tokenizer
        self.tokenizer.save_pretrained(save_path)

        # Speichere Config
        config = {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "max_length": self.max_length,
        }

        with open(save_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def predict(self, texts):
        """Macht Vorhersagen auf neuen Texten."""
        self.model.eval()
        predictions = []
        probabilities = []

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

                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().numpy()[0]

                predictions.append(pred)
                probabilities.append(probs.cpu().numpy()[0])

        return np.array(predictions), np.array(probabilities)


def main():
    """Hauptfunktion für Modell-Training."""
    logger.info(" StaGE Statement Segmentation - BERT Model Training")
    logger.info("=" * 60)

    # Konfiguration
    CONFIG = {
        "model_name": "bert-base-german-cased",
        "num_labels": 2,
        "max_length": 128,
        "num_epochs": 5,
        "batch_size": 16,
        "learning_rate": 2e-5,
    }

    logger.info("⚙ Konfiguration:")
    for key, value in CONFIG.items():
        logger.info(" %s: %s", key, value)
    logger.info("=" * 60)

    # Initialisiere Trainer
    trainer = StaGEModelTrainer(
        model_name=CONFIG["model_name"],
        num_labels=CONFIG["num_labels"],
        max_length=CONFIG["max_length"],
    )

    # Lade Daten
    data = trainer.load_data()

    # Erstelle DataLoaders
    dataloaders = trainer.create_dataloaders(
        data, batch_size=CONFIG["batch_size"], label_column="has_multiple_statements"
    )

    # Training
    train_history, val_history = trainer.train(
        dataloaders=dataloaders,
        num_epochs=CONFIG["num_epochs"],
        learning_rate=CONFIG["learning_rate"],
    )

    # Test Evaluation
    logger.info("\n Finale Test Evaluation:")
    logger.info("=" * 60)
    test_metrics = trainer.evaluate(dataloaders["test"])

    for metric, value in test_metrics.items():
        logger.info(" Test %s: %.4f", metric, value)

    # Test Predictions
    logger.info("\n Test Predictions:")
    logger.info("=" * 60)

    test_texts = [
        "Das Haus ist groß.",
        "Das Haus ist groß und schön.",
        "Der Mann geht in den Park und kauft ein Eis.",
        "Es regnet heute.",
        "Maria arbeitet im Büro, sie ist sehr fleißig und macht ihre Arbeit gut.",
    ]

    predictions, probabilities = trainer.predict(test_texts)

    for text, pred, prob in zip(test_texts, predictions, probabilities, strict=False):
        label = "Mehrere Statements" if pred == 1 else "Ein Statement"
        confidence = prob[pred] * 100
        logger.info(" '%s'", text)
        logger.info(" → %s (Konfidenz: %.1f%)\n", label, confidence)

    logger.info("=" * 60)
    logger.info(" Training und Evaluation abgeschlossen!")
    logger.info(" Modell gespeichert in: %s", trainer.output_dir / 'stage_statement_classifier_best')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
