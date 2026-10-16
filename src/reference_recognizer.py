"""Adapter for Umaima Manzoor's pretrained real-time ASL classifier.

The image preparation below follows the reference project's recognition.py:
pad the detected hand crop on a white 300x300 canvas without stretching,
resize to 128x128, convert BGR to RGB, and scale float32 pixels to [0, 1].
"""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "models/reference_rt_sign_classifier"
MODEL_PATH = REFERENCE_ROOT / "best_model.h5"
MAPPING_PATH = REFERENCE_ROOT / "class_mapping.pkl"
IMAGE_SIZE = 128
CANVAS_SIZE = 300
CLASS_NAMES = [chr(code) for code in range(ord("A"), ord("Z") + 1)] + [str(n) for n in range(10)]


def preprocess_hand(bgr_hand: np.ndarray) -> np.ndarray:
    """Return the reference model's batch-shaped normalized RGB input."""
    if (not isinstance(bgr_hand, np.ndarray) or bgr_hand.ndim != 3
            or bgr_hand.shape[2] != 3 or bgr_hand.size == 0):
        raise ValueError("Expected a non-empty BGR hand crop.")
    height, width = bgr_hand.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Hand crop has invalid dimensions.")
    white = np.full((CANVAS_SIZE, CANVAS_SIZE, 3), 255, dtype=np.uint8)
    if height > width:
        new_height = CANVAS_SIZE
        new_width = int(CANVAS_SIZE * (width / height))
        resized = cv2.resize(bgr_hand, (new_width, new_height))
        x_offset = (CANVAS_SIZE - new_width) // 2
        white[:, x_offset:x_offset + new_width] = resized
    else:
        new_width = CANVAS_SIZE
        new_height = int(CANVAS_SIZE * (height / width))
        resized = cv2.resize(bgr_hand, (new_width, new_height))
        y_offset = (CANVAS_SIZE - new_height) // 2
        white[y_offset:y_offset + new_height, :] = resized
    image = cv2.resize(white, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)


class ReferenceSignRecognizer:
    """Load and run the reference project's checkpoint and class mapping."""

    def __init__(self, model_path: Path = MODEL_PATH, mapping_path: Path = MAPPING_PATH):
        if not model_path.is_file() or not mapping_path.is_file():
            raise FileNotFoundError(f"Reference model and mapping are required: {model_path}, {mapping_path}")
        with mapping_path.open("rb") as mapping_file:
            mapping = pickle.load(mapping_file)
        self.classes = list(mapping.get("classes", []))
        if self.classes != CLASS_NAMES or mapping.get("idx_to_class") != dict(enumerate(CLASS_NAMES)):
            raise ValueError("Reference class mapping must be ordered A-Z, then 0-9.")
        self.model = tf.keras.models.load_model(model_path, compile=False)
        if list(self.model.input_shape[1:]) != [IMAGE_SIZE, IMAGE_SIZE, 3]:
            raise ValueError(f"Unexpected reference model input shape {self.model.input_shape}.")
        if self.model.output_shape[-1] != len(self.classes):
            raise ValueError("Reference model output count does not match class mapping.")
        self.model_path = Path(model_path).resolve()
        self.mapping_path = Path(mapping_path).resolve()

    def predict_crop(self, bgr_hand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the exact preprocessed batch and its model scores."""
        batch = preprocess_hand(bgr_hand)
        scores = np.asarray(self.model.predict(batch, verbose=0)[0], dtype=np.float32)
        if (scores.shape != (len(self.classes),) or not np.isfinite(scores).all()
                or np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=1e-3)):
            raise ValueError("Reference model returned an invalid probability vector.")
        return batch, scores
