"""
Sign2Text - ML Trainer

Training pipeline for hand-sign classifiers.
Handles data loading, validation split, model training,
accuracy evaluation, confusion matrix generation, and model saving.

Also handles edge cases like too few samples, invalid data,
and provides useful warnings instead of crashing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from config.settings import (
    TEST_SPLIT_RATIO,
    RANDOM_STATE,
    CROSS_VALIDATION_FOLDS,
)
from src.dataset.loader import DatasetLoader
from src.model.classifier import SignClassifier, MODEL_METADATA_FILENAME


def load_dataset(
    root_dir: Union[str, Path] = "data/raw",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load all recorded samples from the dataset directory.

    Args:
        root_dir: Root directory containing label subdirectories
            with sample .npy files.

    Returns:
        Tuple of (X, y) where X is the feature matrix and y is the label vector.
        Returns empty arrays if no data is found.
    """
    loader = DatasetLoader(root_dir=root_dir)
    X, y = loader.load()
    return X, y


def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and clean the dataset.

    Removes invalid samples (wrong feature dimensions, empty features, etc.)
    and returns clean data.

    Args:
        X: Feature matrix.
        y: Label vector.

    Returns:
        Cleaned (X, y) tuple.
    """
    # Check that X and y have the same number of samples
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"X has {X.shape[0]}, y has {y.shape[0]}."
        )

    # Check feature dimensions
    expected_dim = SignClassifier().feature_dimension
    if X.shape[1] != expected_dim:
        print(f"Warning: Feature dimension mismatch. "
              f"Expected {expected_dim}, got {X.shape[1]}.")
        # Filter to only keep samples with correct dimension
        valid_indices = [i for i in range(X.shape[0]) if X.shape[1] == expected_dim]
        X = X[valid_indices]
        y = y[valid_indices]

    # Remove samples with zero/NaN features
    valid_rows = []
    for i in range(X.shape[0]):
        row = X[i]
        if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
            valid_rows.append(i)

    if len(valid_rows) < X.shape[0]:
        print(f"Warning: Removed {X.shape[0] - len(valid_rows)} "
              f"samples with invalid features.")
        X = X[valid_rows]
        y = y[valid_rows]

    return X, y


def check_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, any]:
    """Print and return a summary of the dataset.

    Args:
        X: Feature matrix.
        y: Label vector.

    Returns:
        Dictionary with dataset summary statistics.
    """
    if X.shape[0] == 0:
        print("Dataset is empty - no samples found.")
        return {
            "num_samples": 0,
            "num_classes": 0,
            "samples_per_class": {},
            "feature_dimension": 0,
        }

    # Count samples per class
    unique_labels, counts = np.unique(y, return_counts=True)
    samples_per_class = dict(zip([str(l) for l in unique_labels], counts.tolist()))

    summary = {
        "num_samples": int(X.shape[0]),
        "num_classes": int(len(unique_labels)),
        "samples_per_class": samples_per_class,
        "feature_dimension": int(X.shape[1]) if X.shape[0] > 0 else 0,
    }

    # Print summary
    print("Dataset Summary:")
    print(f"  Total samples: {summary['num_samples']}")
    print(f"  Number of classes: {summary['num_classes']}")
    print(f"  Feature dimension: {summary['feature_dimension']}")
    print("  Samples per class:")
    for label, count in samples_per_class.items():
        print(f"    {label}: {count}")

    # Check if we have enough samples per class
    min_samples = min(samples_per_class.values())
    if min_samples < 2:
        print(f"  WARNING: Minimum samples per class is {min_samples}. "
              f"You need at least 2 samples per class for a train/test split.")

    return summary


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str = CLASSIFIER_TYPE,
    model_path: Union[str, Path] = None,
) -> Tuple[SignClassifier, Dict[str, any]]:
    """Train a sign classifier on the provided data.

    Includes dataset validation, train/test split, evaluation,
    and model saving.

    Args:
        X: Feature matrix of shape (num_samples, FEATURE_VECTOR_SIZE).
        y: Label vector of shape (num_samples,).
        classifier_type: "random_forest" or "svm".
        model_path: Optional path to save the trained model.

    Returns:
        Tuple of (trained_classifier, metadata_dict).

    Raises:
        ValueError: If dataset is too small or invalid.
    """
    # Validate dataset
    X, y = validate_dataset(X, y)

    # Check dataset summary
    check_dataset_summary(X, y)

    # Check if we have enough samples for train/test split
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)

    if X.shape[0] < 2:
        raise ValueError("Need at least 2 samples for training.")

    if num_classes < 2:
        raise ValueError(
            f"Need at least 2 classes for training, got {num_classes}."
        )

    # Check minimum samples per class for train/test split
    unique, counts = np.unique(y, return_counts=True)
    min_samples_per_class = min(counts)
    if min_samples_per_class < 2:
        print(f"WARNING: Some classes have fewer than 2 samples "
              f"({min_samples_per_class}). Train/test split may not be meaningful.")
        # Still proceed but warn

    # Split into train/test
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError as e:
        # Stratify failed, try without stratify
        print(f"Warning: Stratified split failed: {e}. "
              f"Splitting without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SPLIT_RATIO, random_state=RANDOM_STATE
        )

    # Train the classifier
    classifier = SignClassifier(classifier_type=classifier_type)
    metadata = classifier.train(X_train, y_train)

    # Evaluate on test set
    y_pred = []
    for i in range(X_test.shape[0]):
        label, conf = classifier.predict(X_test[i])
        y_pred.append(label if label is not None else "")

    # Convert labels to strings for comparison
    y_test_str = [str(label) for label in y_test]
    y_pred_str = [pred if pred is not None else "" for pred in y_pred]

    # Compute accuracy
    if len(y_test_str) > 0:
        accuracy = accuracy_score(y_test_str, y_pred_str)
        print(f"\nTest Set Evaluation:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
        print(f"  Samples: {len(y_test_str)}")

        # Compute confusion matrix
        try:
            cm = confusion_matrix(y_test_str, y_pred_str, labels=sorted(set(y_test_str + y_pred_str)))
            print("  Confusion matrix:")
            print(cm)
        except Exception:
            print("  Could not compute confusion matrix.")

        # Per-class accuracy
        unique_labels_set = sorted(set(y_test_str + y_pred_str))
        print("  Per-class accuracy:")
        for label in unique_labels_set:
            true_pos = sum(1 for t, p in zip(y_test_str, y_pred_str) if t == label and p == label)
            false_pos = sum(1 for t, p in zip(y_test_str, y_pred_str) if p == label and t != label)
            false_neg = sum(1 for t, p in zip(y_test_str, y_pred_str) if t == label and p != label)
            total = sum(1 for t in y_test_str if t == label)
            if total > 0:
                class_acc = true_pos / total
                print(f"    {label}: {class_acc:.1%} ({true_pos}/{total})")

        # Update metadata with evaluation results
        metadata["test_accuracy"] = float(accuracy)
        metadata["test_samples"] = int(len(y_test_str))
    else:
        print("  No test samples available for evaluation.")
        metadata["test_accuracy"] = 0.0
        metadata["test_samples"] = 0

    # Save model if path provided
    if model_path is not None:
        try:
            classifier.save(model_path, metadata)
            print(f"\nModel saved to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model: {e}")

    return classifier, metadata


def evaluate_classifier(
    classifier: SignClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, any]:
    """Evaluate a trained classifier on provided data.

    Args:
        classifier: Trained SignClassifier instance.
        X: Feature matrix of shape (num_samples, FEATURE_VECTOR_SIZE).
        y: Label vector of shape (num_samples,).

    Returns:
        Dictionary of evaluation metrics.
    """
    if not classifier.is_trained:
        raise ValueError("Classifier has not been trained or loaded.")

    # Check feature dimension
    if X.shape[1] != classifier.feature_dimension:
        raise ValueError(
            f"Feature dimension mismatch. Expected {classifier.feature_dimension}, "
            f"got {X.shape[1]}."
        )

    # Predict on all samples
    y_pred = []
    confidences = []
    for i in range(X.shape[0]):
        label, conf = classifier.predict(X[i])
        y_pred.append(label if label is not None else "")
        confidences.append(conf if conf is not None else 0.0)

    # Convert to strings
    y_test_str = [str(label) for label in y]
    y_pred_str = [pred if pred is not None else "" for pred in y_pred]

    # Compute accuracy
    accuracy = accuracy_score(y_test_str, y_pred_str) if len(y_test_str) > 0 else 0.0

    # Compute confusion matrix
    try:
        unique_labels = sorted(set(y_test_str + y_pred_str))
        cm = confusion_matrix(y_test_str, y_pred_str, labels=unique_labels)
    except Exception:
        cm = np.array([])
        unique_labels = []

    # Per-class metrics
    per_class = {}
    if unique_labels:
        for label in unique_labels:
            true_pos = sum(1 for t, p in zip(y_test_str, y_pred_str) if t == label and p == label)
            false_pos = sum(1 for t, p in zip(y_test_str, y_pred_str) if p == label and t != label)
            false_neg = sum(1 for t, p in zip(y_test_str, y_pred_str) if t == label and p != label)
            total = sum(1 for t in y_test_str if t == label)
            precision = true_pos / (true_pos + false_pos + 1e-10) if (true_pos + false_pos) > 0 else 0.0
            recall = true_pos / (total + 1e-10) if total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-10) if (precision + recall) > 0 else 0.0

            per_class[label] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "support": int(total),
            }

    metrics = {
        "accuracy": float(accuracy),
        "num_samples": int(X.shape[0]),
        "num_classes": len(unique_labels) if unique_labels else 0,
        "confusion_matrix": cm.tolist() if cm.size > 0 else [],
        "per_class": per_class,
    }

    # Print summary
    print("Evaluation Results:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")
    print(f"  Samples: {X.shape[0]}")
    print(f"  Classes: {len(unique_labels)}")

    return metrics


def export_model_metadata(
    classifier: SignClassifier,
    output_path: Union[str, Path],
) -> None:
    """Export classifier metadata to a JSON file.

    Args:
        classifier: Trained or loaded SignClassifier instance.
        output_path: Path where metadata JSON should be saved.
    """
    metadata = classifier.get_model_info()
    # Add additional useful info
    metadata["export_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    metadata["python_version"] = __import__("sys").version

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(metadata, indent=2))

    print(f"Model metadata exported to {output_file}")