"""Adapter for the external pretrained isolated-sign LSTM repository.

The source module is loaded under a private name and supplies the authoritative
feature extractor, normalizer, model, and output-index mapping. Its webcam loop is
never entered and its task-model downloader is never called.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np


DEFAULT_REPOSITORY = Path(__file__).resolve().parents[4] / "asl-recognition-lstm"
REQUIRED_ARTIFACTS = (
    Path("model/best.keras"),
    Path("model/actions.npy"),
    Path("model/mean.npy"),
    Path("model/std.npy"),
)


@dataclass(frozen=True)
class WordPrediction:
    label: str
    confidence: float
    index: int


@dataclass(frozen=True)
class WordPredictionResult:
    status: str
    predictions: tuple[WordPrediction, ...] = ()
    error: str | None = None

    @property
    def best(self) -> WordPrediction | None:
        return self.predictions[0] if self.predictions else None


def _check_artifacts(repository: Path) -> None:
    missing = [repository / relative for relative in REQUIRED_ARTIFACTS
               if not (repository / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "Required pretrained word-model artifact(s) are missing: "
            + ", ".join(str(path) for path in missing)
        )


def load_source_module(repository: Path) -> ModuleType:
    """Import the source app once, reusing its model and preprocessing functions."""
    source_path = repository / "app" / "main.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"Word recognizer source entry point not found: {source_path}")
    module_name = f"_sign2text_word_source_{abs(hash(str(source_path.resolve())))}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load word recognizer source: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    module._sign2text_source_module_name = module_name
    return module


class WordRecognizer:
    """Inference facade for sequences shaped like the pretrained model expects."""

    def __init__(
        self,
        repository: str | Path | None = None,
        *,
        source_module: ModuleType | Any | None = None,
    ):
        self.repository = Path(repository or DEFAULT_REPOSITORY).resolve()
        _check_artifacts(self.repository)
        try:
            self.source = source_module or load_source_module(self.repository)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize pretrained word recognizer: {exc}") from exc

        model = getattr(self.source, "model", None)
        actions = getattr(self.source, "actions", None)
        if model is None or actions is None:
            raise RuntimeError("Word source did not expose its loaded model and class labels")
        self.sequence_length = int(model.input_shape[1])
        self.feature_size = int(model.input_shape[2])
        action_array = np.asarray(actions)
        if action_array.ndim != 1:
            raise ValueError(f"Expected a one-dimensional action list, got {action_array.shape}")
        self.class_names = tuple(str(label) for label in action_array.tolist())
        if tuple(model.input_shape) != (None, 40, 444):
            raise ValueError(f"Expected trained input (None,40,444), got {model.input_shape}")
        if len(model.output_shape) != 2 or model.output_shape[-1] != len(self.class_names):
            raise ValueError(
                f"Output shape {model.output_shape} does not match "
                f"{len(self.class_names)} verified action labels"
            )
        if len(set(self.class_names)) != len(self.class_names):
            raise ValueError("Word class labels are not unique")
        self.model = model
        self.num_classes = len(self.class_names)

    def extract_frame(self, results: Any) -> np.ndarray:
        """Use the source webcam app's exact pose/hand/selected-face feature order."""
        features = np.asarray(self.source.extract_keypoints(results), dtype=np.float32)
        expected = (self.feature_size,)
        if features.shape != expected:
            raise ValueError(f"Expected one {self.feature_size}-feature frame, got {features.shape}")
        if not np.all(np.isfinite(features)):
            raise ValueError("Extracted frame contains non-finite values")
        return features

    def predict_sequence(self, sequence: Any, *, top_k: int = 3) -> WordPredictionResult:
        """Normalize and infer one complete source-compatible sequence."""
        try:
            values = np.asarray(sequence, dtype=np.float32)
            expected = (self.sequence_length, self.feature_size)
            if values.shape != expected:
                return WordPredictionResult(
                    status="insufficient frames" if values.ndim == 2 and values.shape[0] < self.sequence_length
                    else "invalid input",
                    error=f"Expected sequence shape {expected}, got {values.shape}",
                )
            if not np.all(np.isfinite(values)):
                return WordPredictionResult(status="invalid input", error="sequence contains non-finite values")
            normalized = np.asarray(self.source.normalize_sequence(values), dtype=np.float32)
            if normalized.shape != expected:
                raise ValueError(f"Source normalizer returned {normalized.shape}, expected {expected}")
            batch = np.ascontiguousarray(normalized[None, ...], dtype=np.float32)
            probabilities = np.asarray(self.model.predict(batch, verbose=0), dtype=np.float32)
            if probabilities.shape != (1, self.num_classes):
                raise ValueError(
                    f"Model returned {probabilities.shape}; expected (1,{self.num_classes})"
                )
            scores = probabilities[0]
            if not np.all(np.isfinite(scores)):
                raise ValueError("Model returned non-finite class scores")
            if (np.any(scores < 0) or np.any(scores > 1)
                    or not np.isclose(float(scores.sum()), 1.0, atol=1e-3)):
                raise ValueError("Model output is not a valid softmax probability vector")
            count = max(1, min(int(top_k), self.num_classes))
            indices = np.argsort(scores)[::-1][:count]
            predictions = tuple(
                WordPrediction(self.class_names[int(index)], float(scores[index]), int(index))
                for index in indices
            )
            return WordPredictionResult(status="ok", predictions=predictions)
        except Exception as exc:
            return WordPredictionResult(status="inference error", error=str(exc))

    def close(self) -> None:
        """Drop the model reference; this standalone mode owns its process/runtime."""
        self.model = None
        if getattr(self.source, "model", None) is not None:
            self.source.model = None
        module_name = getattr(self.source, "_sign2text_source_module_name", None)
        if module_name:
            sys.modules.pop(module_name, None)


def has_detected_hand(results: Any) -> bool:
    return bool(getattr(results, "left_hand_landmarks", None)
                or getattr(results, "right_hand_landmarks", None))

