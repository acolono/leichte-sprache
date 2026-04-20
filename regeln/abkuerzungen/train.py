#!/usr/bin/env python3
"""
BERT-based NER Training Script for German Abbreviation Detection
WITH B-I-O LABEL CORRECTION AND CLASS WEIGHTING

This script trains a token classification model to identify German abbreviations
in text using BERT (bert-base-german-cased).

KEY IMPROVEMENTS:
1. B-I-O Label Correction: Converts sequential B-ABK labels to B-ABK, I-ABK, I-ABK...
2. Weighted Loss: Uses custom trainer with class weights to handle imbalance
3. Robust to multi-word abbreviations: e.g., "GmbH & Co. KG"

Optimized for Apple M1 Mac with 64GB Unified Memory.

Author: Generated for Leichte Sprache Project
Date: 2025-11-12 (Updated with B-I-O corrections)
"""

import logging

logger = logging.getLogger(__name__)

import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from seqeval.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

# Import configuration (from same module directory)
from . import train_config as config

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================


def correct_bio_labels(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Corrects incomplete BIO labels.

    Problem: The dataset labels multi-word abbreviations as [B-ABK, B-ABK, B-ABK, B-ABK]
    Solution: Convert sequential B-ABK labels to B-ABK, I-ABK, I-ABK, I-ABK...

    This teaches the model to distinguish between:
    - B-ABK: Beginning of an abbreviation
    - I-ABK: Inside/Continuation of an abbreviation

    Example:
        Input:  tokens=["GmbH", "&", "Co.", "KG"], labels=["B-ABK", "B-ABK", "B-ABK", "B-ABK"]
        Output: tokens=["GmbH", "&", "Co.", "KG"], labels=["B-ABK", "I-ABK", "I-ABK", "I-ABK"]

    Args:
        example: Dataset example with 'labels' field

    Returns:
        Example with corrected labels
    """
    labels = example["labels"]
    corrected_labels = []
    previous_label = "O"  # Start with O context

    for label in labels:
        if label == "B-ABK":
            # If previous label was ABK-related, this is a continuation → I-ABK
            if previous_label in ["B-ABK", "I-ABK"]:
                corrected_labels.append("I-ABK")
                previous_label = "I-ABK"
            else:
                # This is a new abbreviation → B-ABK
                corrected_labels.append("B-ABK")
                previous_label = "B-ABK"
        else:
            # Keep O labels as-is
            corrected_labels.append(label)
            previous_label = label

    example["labels"] = corrected_labels
    return example


def load_and_clean_data(data_file: str) -> Dataset:
    """
    Load JSONL dataset and clean invalid labels.

    Args:
        data_file: Path to JSONL file

    Returns:
        Cleaned dataset
    """
    logger.info("\n%s", '=' * 80)
    logger.info("LOADING AND CLEANING DATA")
    logger.info("%s", '=' * 80)
    logger.info("Loading data from: %s", data_file)

    # Load JSONL file
    dataset = load_dataset("json", data_files=data_file, split="train")

    logger.info("Initial dataset size: %s examples", len(dataset))

    # Analyze label distribution BEFORE cleaning
    all_labels = []
    for example in dataset:
        all_labels.extend(example["labels"])

    label_counts = Counter(all_labels)
    logger.info("\nInitial label distribution (BEFORE cleaning):")
    for label, count in sorted(label_counts.items()):
        logger.info(" %s: %s (%.2f%)", label, count, count / len(all_labels) * 100)

    # Clean invalid labels (e.g., "." label found in data)
    invalid_labels = set(label_counts.keys()) - set(config.EXPECTED_LABELS)

    if invalid_labels:
        logger.info("\n Found invalid labels: %s", invalid_labels)
        logger.info("Cleaning dataset by replacing invalid labels with 'O'...")

        def clean_labels(example):
            """Replace invalid labels with 'O'"""
            cleaned_labels = [
                label if label in config.EXPECTED_LABELS else "O"
                for label in example["labels"]
            ]
            example["labels"] = cleaned_labels
            return example

        dataset = dataset.map(clean_labels)

        # Re-analyze after cleaning
        all_labels = []
        for example in dataset:
            all_labels.extend(example["labels"])

        label_counts = Counter(all_labels)
        logger.info("\nCleaned label distribution (AFTER removing invalid labels):")
        for label, count in sorted(label_counts.items()):
            logger.info(" %s: %s (%.2f%)", label, count, count / len(all_labels) * 100)

    # CRITICAL: Apply B-I-O label correction
    logger.info("\n%s", '=' * 80)
    logger.info("APPLYING B-I-O LABEL CORRECTION")
    logger.info("%s", '=' * 80)
    logger.info("Converting sequential B-ABK labels to B-ABK, I-ABK, I-ABK...")
    logger.info("Example: [B-ABK, B-ABK, B-ABK] → [B-ABK, I-ABK, I-ABK]")

    dataset = dataset.map(correct_bio_labels)

    # Re-analyze AFTER B-I-O correction
    all_labels = []
    for example in dataset:
        all_labels.extend(example["labels"])

    label_counts = Counter(all_labels)
    logger.info("\nCorrected B-I-O label distribution (AFTER B-I-O correction):")
    for label, count in sorted(label_counts.items()):
        logger.info(" %s: %s (%.2f%)", label, count, count / len(all_labels) * 100)

    # Calculate and display imbalance ratios
    o_count = label_counts.get("O", 0)
    b_abk_count = label_counts.get("B-ABK", 0)
    i_abk_count = label_counts.get("I-ABK", 0)

    logger.info("\nClass imbalance analysis:")
    if b_abk_count > 0:
        logger.info(" O:B-ABK ratio: %.2f:1", o_count / b_abk_count)
    if i_abk_count > 0:
        logger.info(" O:I-ABK ratio: %.2f:1", o_count / i_abk_count)
    if b_abk_count > 0 and i_abk_count > 0:
        logger.info(" B-ABK:I-ABK ratio: %.2f:1", b_abk_count / i_abk_count)

    logger.info("\n Dataset loaded, cleaned, and B-I-O corrected successfully")

    return dataset


def split_dataset(dataset: Dataset) -> DatasetDict:
    """
    Split dataset into train/validation/test (80/10/10).

    Args:
        dataset: Full dataset

    Returns:
        DatasetDict with train, validation, and test splits
    """
    logger.info("\n%s", '=' * 80)
    logger.info("SPLITTING DATASET (80/10/10)")
    logger.info("%s", '=' * 80)

    # First split: 80% train, 20% temp (for val+test)
    train_test_split = dataset.train_test_split(
        test_size=1 - config.TRAIN_RATIO, seed=config.SEED
    )

    # Second split: Split the 20% into 10% val and 10% test (50/50 of the 20%)
    val_test_split = train_test_split["test"].train_test_split(
        test_size=0.5, seed=config.SEED
    )

    # Create final DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": train_test_split["train"],
            "validation": val_test_split["train"],
            "test": val_test_split["test"],
        }
    )

    logger.info("Train size: %s examples (%.0f%)", len(dataset_dict['train']), config.TRAIN_RATIO * 100)
    logger.info("Validation size: %s examples (%.0f%)", len(dataset_dict['validation']), config.VAL_RATIO * 100)
    logger.info("Test size: %s examples (%.0f%)", len(dataset_dict['test']), config.TEST_RATIO * 100)

    return dataset_dict


# ============================================================================
# LABEL PROCESSING
# ============================================================================


def create_label_mappings(dataset: Dataset) -> tuple:
    """
    Create label2id and id2label mappings from dataset.

    IMPORTANT: This function dynamically creates mappings from the data,
    so it will automatically pick up O, B-ABK, and I-ABK after B-I-O correction.

    Args:
        dataset: Training dataset

    Returns:
        Tuple of (label_list, label2id, id2label)
    """
    logger.info("\n%s", '=' * 80)
    logger.info("CREATING LABEL MAPPINGS")
    logger.info("%s", '=' * 80)

    # Get unique labels from training data
    all_labels = set()
    for example in dataset:
        all_labels.update(example["labels"])

    # Sort for consistency (important for reproducibility)
    label_list = sorted(list(all_labels))

    # Create mappings
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}

    logger.info("Found %s unique labels:", len(label_list))
    for label, idx in label2id.items():
        logger.info(" %s -> %s", label, idx)

    # Verify we have the expected labels after B-I-O correction
    expected_after_correction = {"O", "B-ABK", "I-ABK"}
    actual_labels = set(label_list)
    if actual_labels == expected_after_correction:
        logger.info("\n Label set is correct: %s", expected_after_correction)
    else:
        logger.warning("\n WARNING: Unexpected label set!")
        logger.info(" Expected: %s", expected_after_correction)
        logger.info(" Actual: %s", actual_labels)

    return label_list, label2id, id2label


def calculate_class_weights(
    dataset: Dataset, label2id: Dict[str, int]
) -> Optional[torch.Tensor]:
    """
    Calculate class weights for imbalanced data using inverse frequency.

    Args:
        dataset: Training dataset
        label2id: Label to ID mapping

    Returns:
        Tensor of class weights or None if not using class weights
    """
    if not config.USE_CLASS_WEIGHTS:
        return None

    logger.info("\n%s", '=' * 80)
    logger.info("CALCULATING CLASS WEIGHTS")
    logger.info("%s", '=' * 80)

    # Count label occurrences
    label_counts = Counter()
    for example in dataset:
        label_counts.update(example["labels"])

    # Calculate inverse frequency weights
    total_samples = sum(label_counts.values())
    num_classes = len(label2id)

    weights = []
    for label in sorted(label2id.keys(), key=lambda x: label2id[x]):
        count = label_counts[label]
        # Inverse frequency: total / (num_classes * count)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
        logger.info(" %s: count=%s, weight=%.4f", label, count, weight)

    class_weights = torch.tensor(weights, dtype=torch.float32)
    logger.info("\n Class weights calculated: %s", class_weights.tolist())

    return class_weights


# ============================================================================
# TOKENIZATION AND ALIGNMENT
# ============================================================================


def create_tokenize_function(tokenizer, label2id):
    """
    Create tokenization function with proper label alignment for WordPiece tokens.

    CRITICAL: This function handles the complex task of aligning labels with
    BERT's WordPiece sub-tokens. When a word is split into multiple sub-tokens:
    - The first sub-token gets the original label
    - All subsequent sub-tokens get -100 (ignored by loss function)
    - Special tokens ([CLS], [SEP]) also get -100

    Args:
        tokenizer: Hugging Face tokenizer
        label2id: Mapping from label strings to IDs

    Returns:
        Function that tokenizes and aligns labels
    """

    def tokenize_and_align_labels(examples):
        """
        Tokenize text and align labels with sub-tokens.

        Process:
        1. Tokenize pre-tokenized words with is_split_into_words=True
        2. Use word_ids() to map each sub-token back to its original word
        3. Assign labels:
           - Special tokens ([CLS], [SEP], [PAD]): -100
           - First sub-token of a word: original label ID
           - Subsequent sub-tokens of same word: -100

        Example:
            Input: tokens=["Dipl.-Ing.", "Weber"], labels=["B-ABK", "O"]
            After WordPiece: ["[CLS]", "Dipl", ".", "-", "Ing", ".", "Weber", "[SEP]"]
            word_ids: [None, 0, 0, 0, 0, 0, 1, None]
            Output labels: [-100, label_id("B-ABK"), -100, -100, -100, -100, label_id("O"), -100]
        """
        # Tokenize the texts
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=config.TRUNCATION,
            padding=config.PADDING,
            max_length=config.MAX_LENGTH,
            is_split_into_words=True,  # CRITICAL: Tell tokenizer input is pre-tokenized
        )

        all_labels = []

        # Process each example in the batch
        for i, labels in enumerate(examples["labels"]):
            # Get word IDs for this example
            # word_ids() returns the index of the word that each token comes from
            # None indicates special tokens ([CLS], [SEP], [PAD])
            word_ids = tokenized_inputs.word_ids(batch_index=i)

            previous_word_idx = None
            label_ids = []

            # Iterate through each token in the tokenized sequence
            for word_idx in word_ids:
                # Special tokens (None) get -100
                if word_idx is None:
                    label_ids.append(-100)

                # First sub-token of a word: assign the original label
                elif word_idx != previous_word_idx:
                    # Safety check: ensure word_idx is within bounds
                    if word_idx < len(labels):
                        # Convert string label to ID
                        label_ids.append(label2id[labels[word_idx]])
                    else:
                        # If out of bounds, assign -100 (ignore)
                        label_ids.append(-100)

                # Subsequent sub-tokens of the same word: assign -100
                # This ensures only the first sub-token contributes to loss
                else:
                    label_ids.append(-100)

                previous_word_idx = word_idx

            all_labels.append(label_ids)

        tokenized_inputs["labels"] = all_labels
        return tokenized_inputs

    return tokenize_and_align_labels


def tokenize_datasets(
    dataset_dict: DatasetDict, tokenizer, label2id: Dict[str, int]
) -> DatasetDict:
    """
    Apply tokenization to all dataset splits.

    Args:
        dataset_dict: Dictionary containing train/val/test splits
        tokenizer: Hugging Face tokenizer
        label2id: Label to ID mapping

    Returns:
        Tokenized dataset dictionary
    """
    logger.info("\n%s", '=' * 80)
    logger.info("TOKENIZING DATASETS")
    logger.info("%s", '=' * 80)

    tokenize_function = create_tokenize_function(tokenizer, label2id)

    logger.info("Tokenizing train, validation, and test sets...")
    tokenized_datasets = dataset_dict.map(
        tokenize_function, batched=True, desc="Tokenizing datasets"
    )

    logger.info(" Tokenization complete")

    # Display example to verify correct alignment
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE: Tokenization and Label Alignment")
    logger.info("=" * 80)
    example = tokenized_datasets["train"][0]
    tokens = tokenizer.convert_ids_to_tokens(example["input_ids"])
    labels = example["labels"]

    logger.info("%<20 %<10 %s", 'Token', 'Label ID', 'Label')
    logger.info("-" * 50)
    for token, label_id in zip(tokens, labels, strict=False):
        if label_id == -100:
            label_name = "[IGNORE]"
        else:
            label_name = [k for k, v in label2id.items() if v == label_id][0]
        logger.info("%<20 %<10 %s", token, label_id, label_name)
    logger.info("=" * 80)

    return tokenized_datasets


# ============================================================================
# METRICS
# ============================================================================


def create_compute_metrics(label_list: List[str]):
    """
    Create metrics computation function using seqeval.

    seqeval correctly handles sequence labeling metrics by considering
    entity-level performance rather than token-level.

    Args:
        label_list: List of label names

    Returns:
        Function that computes metrics from predictions
    """

    def compute_metrics(eval_prediction):
        """
        Compute precision, recall, F1, and accuracy.

        Process:
        1. Extract predictions (logits) and true labels
        2. Convert logits to predicted label IDs using argmax
        3. Remove -100 labels (ignored tokens) from both predictions and labels
        4. Convert IDs back to label strings
        5. Compute metrics using seqeval

        Args:
            eval_prediction: EvalPrediction object with predictions and label_ids

        Returns:
            Dictionary with precision, recall, f1, and accuracy
        """
        predictions, labels = eval_prediction

        # Convert logits to predictions (argmax)
        # predictions shape: (batch_size, sequence_length, num_labels)
        predictions = np.argmax(predictions, axis=2)

        # Remove -100 labels and convert to label strings
        true_labels = []
        true_predictions = []

        for prediction, label in zip(predictions, labels, strict=False):
            # Filter out -100 (ignored) labels
            # Zip together prediction and label, keep only where label != -100
            filtered = [
                (p, lb) for p, lb in zip(prediction, label, strict=False) if lb != -100
            ]

            if filtered:
                pred_ids, label_ids = zip(*filtered, strict=False)

                # Convert IDs to label strings
                true_labels.append([label_list[lb] for lb in label_ids])
                true_predictions.append([label_list[p] for p in pred_ids])

        # Compute metrics using seqeval
        # seqeval handles sequence labeling correctly (entity-level metrics)
        results = {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

        return results

    return compute_metrics


# ============================================================================
# CUSTOM TRAINER WITH CLASS WEIGHTS
# ============================================================================


class WeightedLossTrainer(Trainer):
    """
    A custom trainer that applies class weights to the loss function.

    This is critical for handling class imbalance between O, B-ABK, and I-ABK.
    Without class weights, the model might learn to predict only the majority class (O).

    The weighted loss ensures that:
    - Rare classes (B-ABK, I-ABK) contribute more to the loss
    - The model learns to recognize abbreviations despite their low frequency
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        Compute loss with class weights - compatible with transformers 4.x+.

        Args:
            model: The model being trained
            inputs: Batch of inputs
            return_outputs: Whether to return model outputs
            num_items_in_batch: Optional parameter for transformers 4.x+ compatibility

        Returns:
            Loss (and outputs if return_outputs=True)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Manually define loss function with class weights
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(self.args.device)
            if self.class_weights is not None
            else None
        )

        # Flatten logits and labels for loss calculation
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# MODEL AND TRAINING
# ============================================================================


def setup_model(
    label_list: List[str], label2id: Dict[str, int], id2label: Dict[int, str]
):
    """
    Load pre-trained BERT model for token classification.

    Args:
        label_list: List of label names
        label2id: Label to ID mapping
        id2label: ID to label mapping

    Returns:
        Model instance
    """
    logger.info("\n%s", '=' * 80)
    logger.info("LOADING MODEL")
    logger.info("%s", '=' * 80)
    logger.info("Model checkpoint: %s", config.MODEL_CHECKPOINT)
    logger.info("Number of labels: %s", len(label_list))

    model = AutoModelForTokenClassification.from_pretrained(
        config.MODEL_CHECKPOINT,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    logger.info(" Model loaded successfully")

    # Display model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("\nModel parameters:")
    logger.info(" Total: %,", total_params)
    logger.info(" Trainable: %,", trainable_params)

    return model


def setup_training_args(num_training_examples: int) -> TrainingArguments:
    """
    Setup training arguments optimized for M1 Mac.

    Args:
        num_training_examples: Number of training examples (for warmup calculation)

    Returns:
        TrainingArguments instance
    """
    logger.info("\n%s", '=' * 80)
    logger.info("CONFIGURING TRAINING")
    logger.info("%s", '=' * 80)

    # Calculate warmup steps (10% of total training steps)
    steps_per_epoch = num_training_examples // config.TRAIN_BATCH_SIZE
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    logger.info("Training steps per epoch: %s", steps_per_epoch)
    logger.info("Total training steps: %s", total_steps)
    logger.info("Warmup steps: %s", warmup_steps)

    # Detect device
    device = config.get_device()
    logger.info("\nUsing device: %s", device.upper())

    # Enable MPS for M1 Mac
    use_mps_device = device == "mps"

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        # Evaluation and saving
        eval_strategy=config.EVALUATION_STRATEGY,
        save_strategy=config.SAVE_STRATEGY,
        # Optimization
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        num_train_epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        max_grad_norm=config.MAX_GRAD_NORM,
        # Device optimization (M1 Mac)
        use_mps_device=use_mps_device,
        # Data loading
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
        # Logging
        logging_dir=config.LOGS_DIR,
        logging_strategy=config.LOGGING_STRATEGY,
        logging_steps=config.LOGGING_STEPS,
        # Model selection
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
        metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
        greater_is_better=config.GREATER_IS_BETTER,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        # Reproducibility
        seed=config.SEED,
        # Report to TensorBoard
        report_to=["tensorboard"],
    )

    logger.info("\n Training arguments configured")

    return training_args


def train_model(
    model,
    tokenized_datasets: DatasetDict,
    tokenizer,
    training_args: TrainingArguments,
    compute_metrics,
    class_weights: Optional[torch.Tensor] = None,
) -> Trainer:
    """
    Train the model using Hugging Face Trainer.

    Args:
        model: Model to train
        tokenized_datasets: Tokenized datasets
        tokenizer: Tokenizer
        training_args: Training arguments
        compute_metrics: Metrics computation function
        class_weights: Optional class weights for weighted loss

    Returns:
        Trained Trainer instance
    """
    logger.info("\n%s", '=' * 80)
    logger.info("INITIALIZING TRAINER")
    logger.info("%s", '=' * 80)

    # Data collator for token classification
    data_collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer, padding=True
    )

    # Setup callbacks
    callbacks = []

    # Add early stopping callback if enabled
    if config.ENABLE_EARLY_STOPPING:
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            early_stopping_threshold=config.EARLY_STOPPING_THRESHOLD,
        )
        callbacks.append(early_stopping)
        logger.info(" Early stopping enabled (patience=%s)", config.EARLY_STOPPING_PATIENCE)

    # CRITICAL: Choose trainer based on whether class weights are enabled
    if config.USE_CLASS_WEIGHTS and class_weights is not None:
        logger.info(" Using WeightedLossTrainer with calculated class weights.")
        logger.info(" Class weights: %s", class_weights.tolist())

        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            class_weights=class_weights,  # Pass weights to custom trainer
        )
    else:
        logger.info(" Using standard Trainer (no class weights).")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    logger.info(" Trainer initialized")

    # Start training
    logger.info("\n%s", '=' * 80)
    logger.info("STARTING TRAINING")
    logger.info("%s", '=' * 80)
    logger.info("This will take approximately 25-35 minutes on M1 Mac...")
    logger.info("Monitor progress in TensorBoard: tensorboard --logdir %s", config.LOGS_DIR)
    logger.info("%s\n", '=' * 80)

    trainer.train()

    logger.info("\n%s", '=' * 80)
    logger.info(" TRAINING COMPLETED")
    logger.info("%s", '=' * 80)

    return trainer


# ============================================================================
# EVALUATION
# ============================================================================


def evaluate_final_model(
    trainer: Trainer, tokenized_datasets: DatasetDict, label_list: List[str]
):
    """
    Evaluate the trained model on the test set.

    Args:
        trainer: Trained trainer
        tokenized_datasets: Tokenized datasets with test split
        label_list: List of label names
    """
    logger.info("\n%s", '=' * 80)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("%s", '=' * 80)

    # Evaluate on test set
    test_results = trainer.evaluate(tokenized_datasets["test"])

    logger.info("\nTest Set Results:")
    logger.info("-" * 40)
    logger.info(" Precision: %.4f", test_results['eval_precision'])
    logger.info(" Recall: %.4f", test_results['eval_recall'])
    logger.info(" F1-Score: %.4f", test_results['eval_f1'])
    logger.info(" Accuracy: %.4f", test_results['eval_accuracy'])

    # Get detailed predictions for classification report
    predictions_output = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions_output.predictions, axis=2)
    labels = predictions_output.label_ids

    # Convert to label strings (removing -100)
    true_labels = []
    true_predictions = []

    for prediction, label in zip(predictions, labels, strict=False):
        filtered = [
            (p, lb) for p, lb in zip(prediction, label, strict=False) if lb != -100
        ]
        if filtered:
            pred_ids, label_ids = zip(*filtered, strict=False)
            true_labels.append([label_list[lb] for lb in label_ids])
            true_predictions.append([label_list[p] for p in pred_ids])

    # Print detailed classification report
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED CLASSIFICATION REPORT")
    logger.info("=" * 80)
    logger.info(classification_report(true_labels, true_predictions, digits=4))

    return test_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    """Main training pipeline with B-I-O correction and class weighting."""

    # Print configuration
    config.print_config()

    # Set random seed for reproducibility
    config.set_seed()
    logger.info("\n Random seed set to %s for reproducibility", config.SEED)

    # 1. Load and clean data (with B-I-O correction)
    dataset = load_and_clean_data(config.DATA_FILE)

    # 2. Split into train/val/test
    dataset_dict = split_dataset(dataset)

    # 3. Create label mappings (will now include I-ABK)
    label_list, label2id, id2label = create_label_mappings(dataset_dict["train"])

    # 4. Calculate class weights (if enabled)
    class_weights = calculate_class_weights(dataset_dict["train"], label2id)

    # 5. Load tokenizer
    logger.info("\n%s", '=' * 80)
    logger.info("LOADING TOKENIZER")
    logger.info("%s", '=' * 80)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT)
    logger.info(" Tokenizer loaded: %s", config.MODEL_CHECKPOINT)

    # 6. Tokenize datasets
    tokenized_datasets = tokenize_datasets(dataset_dict, tokenizer, label2id)

    # 7. Setup model
    model = setup_model(label_list, label2id, id2label)

    # 8. Setup training arguments
    training_args = setup_training_args(len(tokenized_datasets["train"]))

    # 9. Create metrics function
    compute_metrics = create_compute_metrics(label_list)

    # 10. Train model (with optional class weights)
    trainer = train_model(
        model,
        tokenized_datasets,
        tokenizer,
        training_args,
        compute_metrics,
        class_weights=class_weights,
    )

    # 11. Final evaluation
    test_results = evaluate_final_model(trainer, tokenized_datasets, label_list)

    # 12. Save final model
    logger.info("\n%s", '=' * 80)
    logger.info("SAVING FINAL MODEL")
    logger.info("%s", '=' * 80)
    logger.info("Saving to: %s", config.FINAL_MODEL_DIR)
    trainer.save_model(config.FINAL_MODEL_DIR)
    tokenizer.save_pretrained(config.FINAL_MODEL_DIR)
    logger.info(" Model and tokenizer saved successfully")

    # Save label mappings
    label_mapping_file = Path(config.FINAL_MODEL_DIR) / "label_mappings.json"
    with open(label_mapping_file, "w") as f:
        json.dump(
            {
                "label_list": label_list,
                "label2id": label2id,
                "id2label": {
                    int(k): v for k, v in id2label.items()
                },  # Convert keys to int for JSON
            },
            f,
            indent=2,
        )
    logger.info(" Label mappings saved to: %s", label_mapping_file)

    logger.info("\n%s", '=' * 80)
    logger.info(" TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("%s", '=' * 80)
    logger.info("\nFinal Test F1-Score: %.4f", test_results['eval_f1'])
    logger.info("\nYou can now use the trained model from: %s", config.FINAL_MODEL_DIR)
    logger.info("\nTo use the model for inference:")
    logger.info(" from transformers import pipeline")
    logger.info(" ner_pipeline = pipeline('ner', model='%s')", config.FINAL_MODEL_DIR)
    logger.info(" results = ner_pipeline('Text mit GmbH & Co. KG und ca. Abkürzungen')")
    logger.info("\n%s", '=' * 80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
