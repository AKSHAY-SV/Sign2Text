"""
Sign2Text - CNN Image Classifier

Lightweight CNN classifier for ASL alphabet recognition.
Loads a trained .h5 model and provides predict() method for single-image inference.

Dependencies:
    - TensorFlow 2.x
    - NumPy

Model format:
    - Saved as sign_cnn.h5
    - Metadata saved as models/metadata.json

The classifier maps a preprocessed image to one of 35 classes:
    A-Z (26 letters), del, nothing, space
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from src.model.image_preprocessing import preprocess_rgb_tensor


# =============================================================================
# Configuration
# =============================================================================

MODEL_FILENAME = "sign_cnn.h5"
METADATA_FILENAME = "metadata.json"

DEFAULT_IMAGE_SIZE = 64
NUM_CLASSES = 29  # A-Z (26) + del + nothing + space

CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
               "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
               "del", "nothing", "space"]

# Mapping from class index to class name
INDEX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


# =============================================================================
# Model Management
# =============================================================================


class CnnClassifier:
    """
    CNN-based sign classifier for ASL alphabet recognition.

    Loads a pretrained model and provides prediction functionality.
    The classifier expects images resized to image_size x image_size,
    RGB channel order, pixel values normalized to [0, 1].

    Attributes:
        model: Loaded TensorFlow Keras model.
        image_size: Input image dimension (square).
        num_classes: Number of output classes.
        class_names: List of class name strings.
        label_to_idx: Dictionary mapping class name -> integer index.
        idx_to_label: Dictionary mapping integer index -> class name.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = None,
        metadata_path: Union[str, Path] = None,
    ) -> None:
        """Initialize the classifier.

        Args:
            model_path: Path to the .h5 model file. If None, attempts to
                load from default locations.
            metadata_path: Path to the metadata JSON file. If None, attempts
                to load from default locations.
        """
        self.model: Optional[keras.Model] = None
        explicit_metadata = metadata_path is not None
        self._model_input_size: Optional[int] = None
        self.image_size = DEFAULT_IMAGE_SIZE
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES.copy()
        self.label_to_idx = CLASS_TO_IDX.copy()
        self.idx_to_label = INDEX_TO_CLASS.copy()
        self.metadata: Dict[str, Any] = {}
        self._metadata_loaded = False
        self.model_path: Optional[Path] = None
        self.metadata_path: Optional[Path] = None

        # Try to load model and metadata
        if model_path is not None:
            self.load_model(str(model_path))
            if explicit_metadata:
                self.load_metadata(str(metadata_path))
            elif self.model_path is not None:
                sibling_metadata = self.model_path.parent / METADATA_FILENAME
                if sibling_metadata.is_file():
                    self.load_metadata(str(sibling_metadata))
            if not self._metadata_loaded:
                raise FileNotFoundError(
                    f"Matching metadata file not found beside model: {self.model_path}"
                )
        else:
            # Try default locations
            default_model = Path("models") / MODEL_FILENAME
            default_metadata = Path("models") / METADATA_FILENAME
            if default_model.is_file():
                self.load_model(str(default_model))
            if default_metadata.is_file():
                self.load_metadata(str(default_metadata))

    def load_model(self, model_path: str) -> None:
        """Load a Keras model from file.

        Args:
            model_path: Path to .h5 model file.

        Raises:
            FileNotFoundError: If the model file doesn't exist.
        """
        model_file = Path(model_path)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        print(f"Loading model from {model_file}...")
        self.model = keras.models.load_model(model_file, compile=False)
        self.model_path = model_file.resolve()

        # Update dimensions from the model if different
        if self.model.input_shape is not None:
            height, width, channels = self.model.input_shape[1:]
            if height != width or channels != 3:
                raise ValueError(f"Expected a square RGB model input, got {self.model.input_shape}.")
            self._model_input_size = int(height)
            self.image_size = self._model_input_size
        if self.model.output_shape[-1] != self.num_classes:
            raise ValueError(
                f"Model outputs {self.model.output_shape[-1]} classes, "
                f"but metadata defines {self.num_classes}."
            )

        print(f"  Input shape: {self.model.input_shape}")
        print(f"  Number of classes: {self.num_classes}")

    def load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """Load metadata from JSON file.

        Args:
            metadata_path: Path to metadata JSON file.
        """
        meta_file = Path(metadata_path)
        if not meta_file.is_file():
            print(f"Warning: Metadata file not found: {meta_file}")
            return {}

        print(f"Loading metadata from {meta_file}...")
        with open(meta_file, "r") as f:
            metadata = json.load(f)
        if not isinstance(metadata, dict):
            raise ValueError("Model metadata must be a JSON object.")
        self.metadata = metadata

        # Update attributes from metadata
        if "image_size" in metadata:
            metadata_image_size = int(metadata["image_size"])
            if self._model_input_size is not None and metadata_image_size != self._model_input_size:
                raise ValueError(
                    f"Metadata image_size {metadata_image_size} does not match model input "
                    f"{self._model_input_size}."
                )
            self.image_size = metadata_image_size
        metadata_num_classes = int(metadata.get("num_classes", len(self.class_names)))
        if self.model is not None and self.model.output_shape[-1] != metadata_num_classes:
            raise ValueError("Loaded model output dimension does not match metadata.")
        self.num_classes = metadata_num_classes
        if "class_names" in metadata:
            self.class_names = list(metadata["class_names"])
            if len(self.class_names) != self.num_classes:
                raise ValueError("Metadata class_names length does not match num_classes.")
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Metadata class_names contains duplicates.")
        expected_mapping = {name: i for i, name in enumerate(self.class_names)}
        if "class_to_idx" in metadata:
            mapping = metadata["class_to_idx"]
            if mapping != expected_mapping:
                raise ValueError("Metadata class_to_idx does not match class_names order.")
            self.label_to_idx = mapping
        if "idx_to_label" in metadata:
            index_mapping = {int(k): v for k, v in metadata["idx_to_label"].items()}
            if index_mapping != dict(enumerate(self.class_names)):
                raise ValueError("Metadata idx_to_label does not match class_names order.")
        self.idx_to_label = dict(enumerate(self.class_names))
        self.label_to_idx = expected_mapping
        self.metadata_path = meta_file.resolve()
        self._metadata_loaded = True
        if self.model is not None and self.model.output_shape[-1] != self.num_classes:
            raise ValueError("Loaded model output dimension does not match metadata.")

        print(f"  Image size: {self.image_size}")
        print(f"  Classes: {len(self.class_names)}")

        return metadata

    def is_trained(self) -> bool:
        """Check if the classifier has a loaded model."""
        return self.model is not None

    def predict(self, image: np.ndarray) -> Tuple[Optional[str], Optional[float]]:
        """
        Predict the sign class for a single preprocessed image.

        Args:
            image: NumPy array of shape (image_size, image_size, 3)
                   with pixel values in [0, 1] range (already normalized).

        Returns:
            Tuple of (predicted_label, confidence_score).
            Returns (None, None) if the model has not been loaded.
        """
        if self.model is None:
            return None, None

        if image.shape != (self.image_size, self.image_size, 3):
            raise ValueError(
                f"Expected image shape ({self.image_size}, {self.image_size}, 3), "
                f"got {image.shape}."
            )

        # Add batch dimension
        img_batch = np.expand_dims(image, axis=0).astype(np.float32)

        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]
        if predictions.ndim != 1 or predictions.size != self.num_classes:
            raise ValueError(f"Invalid model score vector shape: {predictions.shape}.")
        if not np.all(np.isfinite(predictions)):
            raise ValueError("Model returned non-finite scores.")
        if np.any(predictions < 0) or np.any(predictions > 1) or not np.isclose(predictions.sum(), 1.0, atol=1e-3):
            raise ValueError("Model output is not a normalized probability vector.")

        # Get top prediction
        pred_idx = int(np.argmax(predictions))
        pred_confidence = float(np.max(predictions))
        pred_label = self.idx_to_label.get(pred_idx, "unknown")

        return pred_label, pred_confidence

    def predict_proba(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get prediction probabilities for all classes.

        Args:
            image: Preprocessed image array.

        Returns:
            Probability array of shape (num_classes,), or None if model not loaded.
        """
        if self.model is None:
            return None

        if image.shape != (self.image_size, self.image_size, 3):
            raise ValueError(
                f"Expected image shape ({self.image_size}, {self.image_size}, 3), got {image.shape}."
            )

        img_batch = np.expand_dims(image, axis=0).astype(np.float32)
        scores = self.model.predict(img_batch, verbose=0)[0]
        if (scores.ndim != 1 or scores.size != self.num_classes
                or not np.all(np.isfinite(scores)) or np.any(scores < 0)
                or np.any(scores > 1) or not np.isclose(scores.sum(), 1.0, atol=1e-3)):
            raise ValueError("Model output is not a normalized probability vector.")
        return scores

    def get_model_info(self) -> Dict[str, any]:
        """Get information about the current model."""
        if self.model is None:
            return {"is_trained": False}

        return {
            "is_trained": True,
            "model_type": "CNN",
            "image_size": self.image_size,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "architecture": "CNN: Conv2D(32) -> Conv2D(64) -> Dense(128) -> Output",
        }


# =============================================================================
# Image Preprocessing (matching training pipeline)
# =============================================================================


def preprocess_image(
    img_array: np.ndarray,
    target_size: Tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
) -> np.ndarray:
    """
    Preprocess a raw image array for CNN prediction.

    Steps:
    1. Resize to target_size
    2. Convert to RGB if needed
    3. Normalize pixel values to [0, 1] range
    4. Ensure float32 dtype

    Args:
        img_array: Raw image array from OpenCV (BGR channel order).
        target_size: (height, width) to resize to.

    Returns:
        Preprocessed image array of shape (target_size[0], target_size[1], 3)
        with RGB channel order and pixel values in [0, 1] range.
    """
    import cv2

    # OpenCV reads as BGR; convert to RGB
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_array

    # Exactly the same resize and normalization used by train_cnn.make_dataset.
    return preprocess_rgb_tensor(tf.convert_to_tensor(img_rgb), target_size).numpy()


# =============================================================================
# Batch Prediction
# =============================================================================


def predict_batch(
    images: List[np.ndarray],
    classifier: CnnClassifier,
) -> List[Tuple[Optional[str], Optional[float]]]:
    """
    Predict classes for a batch of preprocessed images.

    Args:
        images: List of preprocessed image arrays (each shape: HxWx3, [0,1]).
        classifier: CnnClassifier instance.

    Returns:
        List of (label, confidence) tuples.
    """
    results = []
    for img in images:
        label, conf = classifier.predict(img)
        results.append((label, conf))
    return results
