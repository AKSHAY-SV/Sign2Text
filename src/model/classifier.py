"""
Sign2Text - ML Classifier

Lightweight hand-sign classification using classical ML algorithms.
Currently supports Random Forest and SVM classifiers.

The classifier produces a label and confidence score from a 126-dimensional
feature vector extracted from hand landmarks.

Model management:
- Models are saved as joblib .pkl files
- Metadata (class labels, feature dimensions, preprocessing info, model version)
  is saved alongside the model in a separate .json file
- The prediction program loads the saved model rather than retraining every time
- If no model exists, a clear error message explains that training must be performed first
"""

from __future__ import annotations

import json
import joblib
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

from config.settings import (
    CLASSIFIER_TYPE,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_MIN_SAMPLES_SPLIT,
    RANDOM_FOREST_MIN_SAMPLES_LEAF,
    SVM_C,
    SVM_KERNEL,
    SVM_GAMMA,
)
from src.features.extractor import FEATURE_VECTOR_SIZE


MODEL_METADATA_FILENAME = "model_metadata.json"
"""Filename for model metadata JSON file (saved alongside the .pkl model file)."""

MODEL_VERSION = "1.0.0"
"""Version of the model format/sign classifier."""


class SignClassifier:
    """Classifies hand signs from feature vectors.

    The classifier maps a 126-dimensional feature vector (2 hands x 63 coords)
    to a gesture label string and a confidence score.

    Attributes:
        model_path: Path where the model is saved/loaded from.
        classifier_type: "random_forest" or "svm".
        _model: The underlying sklearn classifier.
        _label_classes: List of known label classes in order.
        _feature_dimension: Expected feature vector dimension.
        _preprocessing: Description of preprocessing applied.
        _model_version: Version string of the model.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = None,
        classifier_type: str = CLASSIFIER_TYPE,
    ) -> None:
        """Initialize the classifier.

        Args:
            model_path: Optional path to load a pretrained model from.
                If None, a new untrained classifier is created.
            classifier_type: "random_forest" or "svm".
        """
        self._model_path = Path(model_path) if model_path else None
        self._classifier_type = classifier_type
        self._model: Optional = None
        self._label_classes: Optional[list] = None
        self._feature_dimension = FEATURE_VECTOR_SIZE
        self._preprocessing = "wrist-relative, scale-normalized (wrist-to-MCP)"
        self._model_version = MODEL_VERSION

        if model_path is not None and Path(model_path).is_file():
            self.load(model_path)
        else:
            # Create a new untrained classifier
            self._build_classifier()

    def _build_classifier(self) -> None:
        """Build a new classifier instance based on the configured type."""
        if self._classifier_type == "random_forest":
            self._model = RandomForestClassifier(
                n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                max_depth=RANDOM_FOREST_MAX_DEPTH,
                min_samples_split=RANDOM_FOREST_MIN_SAMPLES_SPLIT,
                min_samples_leaf=RANDOM_FOREST_MIN_SAMPLES_LEAF,
                random_state=42,
            )
        elif self._classifier_type == "svm":
            self._model = svm.SVC(
                C=SVM_C,
                kernel=SVM_KERNEL,
                gamma=SVM_GAMMA,
                probability=True,
                random_state=42,
            )
        else:
            raise ValueError(
                f"Unknown classifier type: {self._classifier_type}. "
                f"Use 'random_forest' or 'svm'."
            )

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, any]:
        """Train the classifier on feature vectors and labels.

        Args:
            X: Feature matrix of shape (num_samples, FEATURE_VECTOR_SIZE).
            y: Label vector of shape (num_samples,).

        Returns:
            Dictionary of training metadata including accuracy, class info, etc.
        """
        if self._model is None:
            self._build_classifier()

        if X.shape[1] != FEATURE_VECTOR_SIZE:
            raise ValueError(
                f"Expected feature vectors of size {FEATURE_VECTOR_SIZE}, "
                f"got {X.shape[1]}."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"X has {X.shape[0]}, y has {y.shape[0]}."
            )

        if X.shape[0] == 0:
            raise ValueError("No training data provided.")

        if X.shape[0] < 2:
            raise ValueError("Need at least 2 samples for training.")

        # Fit the model - sklearn labels should be integers or strings
        self._model.fit(X, y)

        # Store the classes seen during training
        if hasattr(self._model, "classes_"):
            self._label_classes = list(self._model.classes_)

        # If no classes were stored, derive from unique labels
        if self._label_classes is None:
            self._label_classes = sorted(set(str(label) for label in y))

        # Compute training accuracy on the training data
        y_pred = self._model.predict(X)
        train_accuracy = accuracy_score(
            [str(label) for label in y],
            [str(pred) for pred in y_pred]
        )

        # Prepare metadata
        metadata = {
            "model_version": self._model_version,
            "classifier_type": self._classifier_type,
            "feature_dimension": self._feature_dimension,
            "num_classes": len(self._label_classes),
            "class_labels": self._label_classes,
            "preprocessing": self._preprocessing,
            "train_samples": int(X.shape[0]),
            "test_split_ratio": 0.2,  # Would be set by trainer
            "train_accuracy": float(train_accuracy),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        return metadata

    def predict(self, feature_vector: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """Predict the sign label and confidence for a single feature vector.

        Args:
            feature_vector: 1D array of shape (FEATURE_VECTOR_SIZE,).

        Returns:
            Tuple of (label, confidence). Returns (None, None) if
            the model has not been trained.
        """
        if self._model is None:
            return None, None

        if feature_vector.shape[0] != FEATURE_VECTOR_SIZE:
            raise ValueError(
                f"Expected feature vector of size {FEATURE_VECTOR_SIZE}, "
                f"got {feature_vector.shape[0]}."
            )

        # Get prediction
        label = self._model.predict([feature_vector])[0]

        # Get confidence
        confidence = None
        if hasattr(self._model, "predict_proba"):
            probs = self._model.predict_proba([feature_vector])[0]
            max_prob = float(np.max(probs))
            confidence = max_prob
        elif hasattr(self._model, "decision_function"):
            # For SVM, use decision function scaled to [0, 1] range
            decision = self._model.decision_function([feature_vector])[0]
            # Simple scaling - this is heuristic
            confidence = float(1.0 / (1.0 + abs(decision)))

        # Convert label to string
        label_str = str(label)

        return label_str, confidence

    def predict_proba(self, feature_vector: np.ndarray) -> Optional[np.ndarray]:
        """Get prediction probabilities for a feature vector.

        Args:
            feature_vector: 1D array of shape (FEATURE_VECTOR_SIZE,).

        Returns:
            Probability array, or None if not supported.
        """
        if self._model is None:
            return None

        if not hasattr(self._model, "predict_proba"):
            return None

        if feature_vector.shape[0] != FEATURE_VECTOR_SIZE:
            raise ValueError(
                f"Expected feature vector of size {FEATURE_VECTOR_SIZE}, "
                f"got {feature_vector.shape[0]}."
            )

        return self._model.predict_proba([feature_vector])[0]

    def save(self, model_path: Union[str, Path], metadata: Optional[Dict] = None) -> None:
        """Save the trained model and metadata to disk.

        The model is saved as a joblib .pkl file, and metadata is saved
        as a separate .json file with the same base name.

        Args:
            model_path: Path where the model should be saved.
            metadata: Optional metadata dictionary. If None, metadata
                generated during training will be used.
        """
        if self._model is None:
            raise ValueError("No trained model to save.")

        save_path = Path(model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the model
        joblib.dump(self._model, save_path)

        # Prepare and save metadata
        if metadata is None:
            # Generate default metadata
            if self._label_classes is None:
                metadata = {
                    "model_version": self._model_version,
                    "classifier_type": self._classifier_type,
                    "feature_dimension": self._feature_dimension,
                    "num_classes": 0,
                    "class_labels": [],
                    "preprocessing": self._preprocessing,
                    "train_samples": 0,
                    "test_split_ratio": 0.2,
                    "train_accuracy": 0.0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            else:
                metadata = {
                    "model_version": self._model_version,
                    "classifier_type": self._classifier_type,
                    "feature_dimension": self._feature_dimension,
                    "num_classes": len(self._label_classes),
                    "class_labels": self._label_classes,
                    "preprocessing": self._preprocessing,
                    "train_samples": 0,  # Would be filled by trainer
                    "test_split_ratio": 0.2,
                    "train_accuracy": 0.0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }

        # Save metadata alongside the model
        metadata_path = save_path.parent / MODEL_METADATA_FILENAME
        metadata_path.write_text(json.dumps(metadata, indent=2))

        print(f"Model saved to {save_path}")
        print(f"Metadata saved to {metadata_path}")

    def load(self, model_path: Union[str, Path]) -> Dict[str, any]:
        """Load a pretrained model and metadata from disk.

        The model is loaded from a joblib .pkl file, and metadata is loaded
        from the accompanying .json file.

        Args:
            model_path: Path to the saved model file (.pkl).

        Returns:
            Dictionary of loaded metadata.

        Raises:
            FileNotFoundError: If the model file or metadata file doesn't exist.
        """
        load_path = Path(model_path)
        if not load_path.is_file():
            raise FileNotFoundError(f"Model file not found: {load_path}")

        # Load the model
        self._model = joblib.load(load_path)

        # Try to load metadata
        metadata_path = load_path.parent / MODEL_METADATA_FILENAME
        metadata: Dict[str, any] = {}

        if metadata_path.is_file():
            try:
                loaded_metadata = json.loads(metadata_path.read_text())
                metadata.update(loaded_metadata)

                # Restore model attributes from metadata
                if "class_labels" in metadata:
                    self._label_classes = metadata["class_labels"]
                if "feature_dimension" in metadata:
                    self._feature_dimension = metadata["feature_dimension"]
                if "classifier_type" in metadata:
                    self._classifier_type = metadata["classifier_type"]
                if "model_version" in metadata:
                    self._model_version = metadata["model_version"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not parse model metadata: {e}")
                # Continue with minimal metadata
                metadata = {
                    "model_version": self._model_version,
                    "classifier_type": self._classifier_type,
                    "feature_dimension": self._feature_dimension,
                    "num_classes": 0,
                    "class_labels": [],
                    "preprocessing": self._preprocessing,
                    "train_samples": 0,
                    "test_split_ratio": 0.2,
                    "train_accuracy": 0.0,
                    "timestamp": "unknown",
                }
        else:
            # No metadata file - use defaults and warn
            print(
                f"Warning: No metadata file found at {metadata_path}. "
                f"Using default values. "
                f"Consider retraining and saving the model with metadata."
            )
            metadata = {
                "model_version": self._model_version,
                "classifier_type": self._classifier_type,
                "feature_dimension": self._feature_dimension,
                "num_classes": 0,
                "class_labels": [],
                "preprocessing": self._preprocessing,
                "train_samples": 0,
                "test_split_ratio": 0.2,
                "train_accuracy": 0.0,
                "timestamp": "unknown",
            }

        # Restore attributes from metadata
        self._classifier_type = metadata.get("classifier_type", self._classifier_type)
        self._feature_dimension = metadata.get(
            "feature_dimension", self._feature_dimension
        )
        self._model_version = metadata.get("model_version", self._model_version)

        # Restore label classes if available
        if "class_labels" in metadata and metadata["class_labels"] is not None:
            self._label_classes = metadata["class_labels"]

        print(f"Model loaded from {load_path}")
        print(f"Metadata loaded from {metadata_path}")

        return metadata

    @property
    def is_trained(self) -> bool:
        """Check if the classifier has been trained or loaded."""
        return self._model is not None

    @property
    def label_classes(self) -> Optional[list]:
        """Get the list of known label classes."""
        return self._label_classes

    @property
    def feature_dimension(self) -> int:
        """Get the expected feature vector dimension."""
        return self._feature_dimension

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model.

        Returns:
            Dictionary with model information including version, type,
            feature dimension, and class labels.
        """
        return {
            "model_version": self._model_version,
            "classifier_type": self._classifier_type,
            "feature_dimension": self._feature_dimension,
            "num_classes": len(self._label_classes) if self._label_classes else 0,
            "class_labels": self._label_classes,
            "is_trained": self.is_trained,
        }