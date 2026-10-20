"""Thin adapter around the separately cloned ASL alphabet model.

The ONNX weights and their source modules stay in the alphabet repository. This
adapter loads that repository's renderer, smoother, and J/Z motion rules so live
preprocessing remains identical to the model's own application.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np


DEFAULT_REPOSITORY = Path(__file__).resolve().parents[4] / "MediaPipe-ASL-sign-language-recognition"
DEFAULT_MODEL_NAME = "asl_cnn_model.onnx"
DEFAULT_CLASSES_NAME = "class_names.txt"
SMOOTHING_WINDOW = 8
MODEL_INPUT_SHAPE = (1, 96, 96, 3)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    index: int
    motion_resolved: bool = False


@dataclass(frozen=True)
class PredictionResult:
    status: str
    predictions: tuple[Prediction, ...] = ()
    moving: bool = False
    updated: bool = False
    error: str | None = None

    @property
    def best(self) -> Prediction | None:
        return self.predictions[0] if self.predictions else None


def _load_source_module(name: str, path: Path) -> ModuleType:
    """Load a clone-owned source module under a private name, avoiding sys.path edits."""
    module_name = f"_sign2text_alphabet_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load alphabet source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def load_support_modules(repository: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    required = {
        "preprocessing": repository / "preprocessing.py",
        "motion": repository / "motion.py",
        "smoothing": repository / "smoothing.py",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required alphabet source file(s): " + ", ".join(missing))
    return tuple(_load_source_module(name, path) for name, path in required.items())


def read_class_names(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Alphabet class list not found: {path}")
    names = tuple(part.strip() for part in path.read_text(encoding="utf-8").strip().split(","))
    expected = tuple("abcdefghijklmnopqrstuvwxyz")
    if names != expected:
        raise ValueError(
            "Alphabet class list must match the verified a-z output order; "
            f"found {names!r}"
        )
    return names


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    if values.ndim != 1 or values.size != 26 or not np.all(np.isfinite(values)):
        raise ValueError(f"Expected 26 finite logits, got shape {values.shape}")
    shifted = values - np.max(values)
    exponents = np.exp(shifted)
    total = float(exponents.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Could not normalize classifier logits")
    return exponents / total


class AlphabetRecognizer:
    """Run one-frame alphabet inference using the cloned model's exact preprocessing.

    MediaPipe `hand_landmarks` objects and original frame dimensions are accepted.
    The recognizer smooths pixel landmarks, runs the repository's J/Z motion rules,
    renders a 96x96 RGB hand skeleton and feeds float32 pixels in [0,255] to ONNX.
    """

    def __init__(
        self,
        repository: str | os.PathLike[str] | None = None,
        *,
        model_path: str | os.PathLike[str] | None = None,
        classes_path: str | os.PathLike[str] | None = None,
        session: Any | None = None,
        runtime: Any | None = None,
    ):
        self.repository = Path(repository or os.environ.get("ASL_ALPHABET_REPO", DEFAULT_REPOSITORY)).resolve()
        self.model_path = Path(model_path) if model_path else self.repository / DEFAULT_MODEL_NAME
        self.classes_path = Path(classes_path) if classes_path else self.repository / DEFAULT_CLASSES_NAME
        self.class_names = read_class_names(self.classes_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Pretrained alphabet checkpoint not found: {self.model_path}")

        self.preprocessing, motion_module, smoothing_module = load_support_modules(self.repository)
        self._tracker = motion_module.MotionTracker()
        self._smoother = smoothing_module.OneEuroFilter()
        self._probability_history: deque[np.ndarray] = deque(maxlen=SMOOTHING_WINDOW)
        self._last_static_label: str | None = None
        self._last_result = PredictionResult(status="insufficient frames")

        if session is None:
            if runtime is None:
                try:
                    runtime = importlib.import_module("onnxruntime")
                except ImportError as exc:
                    raise RuntimeError(
                        "ONNX Runtime is required for alphabet inference. The existing "
                        "project environments do not contain it; see docs/ALPHABET_MODE.md "
                        "for the isolated .venv-alphabet setup."
                    ) from exc
            session = runtime.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        self._session = session
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {len(inputs)}")
        self._input_name = inputs[0].name
        shape = tuple(inputs[0].shape)
        if len(shape) != 4 or tuple(shape[1:]) != (96, 96, 3):
            raise ValueError(f"Expected ONNX input [N,96,96,3], got {shape}")
        if not outputs:
            raise ValueError("Alphabet ONNX checkpoint has no outputs")
        output_names = [item.name for item in outputs]
        self._logits_name = "logits" if "logits" in output_names else output_names[0]

    def _run_logits(self, image: np.ndarray) -> np.ndarray:
        model_input = np.asarray(image, dtype=np.float32)
        if model_input.shape != MODEL_INPUT_SHAPE:
            raise ValueError(f"Expected image tensor {MODEL_INPUT_SHAPE}, got {model_input.shape}")
        if not np.all(np.isfinite(model_input)):
            raise ValueError("Model input contains non-finite values")
        raw = self._session.run([self._logits_name], {self._input_name: model_input})[0]
        logits = np.asarray(raw, dtype=np.float32)
        if logits.shape != (1, 26):
            raise ValueError(f"Expected ONNX logits shape (1,26), got {logits.shape}")
        return logits[0]

    def process_hand(self, hand_landmarks: Any, width: int, height: int, *, infer: bool = True) -> PredictionResult:
        """Update the motion tracker every frame and infer at the requested cadence."""
        if hand_landmarks is None:
            self.reset()
            return PredictionResult(status="no hand detected", updated=True)
        if width <= 0 or height <= 0:
            return PredictionResult(status="invalid input", error="frame dimensions must be positive")
        try:
            points_px = self.preprocessing.landmarks_to_pixels(hand_landmarks, width, height)
            points_px = np.asarray(points_px, dtype=np.float32)
            if points_px.shape != (21, 2) or not np.all(np.isfinite(points_px)):
                raise ValueError(f"Expected 21 finite hand points, got {points_px.shape}")
            smoothed = self._smoother(points_px)
            self._tracker.update(smoothed)
            moving = bool(self._tracker.is_moving())
            if not infer:
                return PredictionResult(
                    status=self._last_result.status,
                    predictions=self._last_result.predictions,
                    moving=moving,
                    updated=False,
                )
            normalized = self.preprocessing.normalize_landmarks(smoothed)
            skeleton = self.preprocessing.render_skeleton(
                normalized, size=self.preprocessing.MODEL_INPUT_SIZE
            )
            model_input = np.asarray(self.preprocessing.model_input(skeleton), dtype=np.float32)
            probabilities = softmax(self._run_logits(model_input))
            self._probability_history.append(probabilities)
            smoothed_probabilities = np.mean(self._probability_history, axis=0)
            index = int(np.argmax(smoothed_probabilities))
            label = self.class_names[index]
            resolved = self._tracker.classify(label)
            if resolved in self.class_names:
                label = resolved
                index = self.class_names.index(label)
            ranked_indices = np.argsort(smoothed_probabilities)[::-1][:3]
            ranked = tuple(
                Prediction(self.class_names[int(i)], float(smoothed_probabilities[i]), int(i))
                for i in ranked_indices
            )
            self._last_static_label = self.class_names[int(np.argmax(smoothed_probabilities))]
            if resolved in self.class_names and resolved != self._last_static_label:
                motion_prediction = Prediction(
                    resolved,
                    float(smoothed_probabilities[self.class_names.index(self._last_static_label)]),
                    self.class_names.index(resolved),
                    motion_resolved=True,
                )
                ranked = (motion_prediction,) + tuple(
                    item for item in ranked if item.label != resolved
                )
            self._last_result = PredictionResult(
                status="ok", predictions=ranked, moving=moving, updated=True
            )
            return self._last_result
        except Exception as exc:
            return PredictionResult(status="inference error", error=str(exc), updated=infer)

    def reset(self) -> None:
        self._probability_history.clear()
        self._last_static_label = None
        self._last_result = PredictionResult(status="no hand detected")
        self._tracker.reset()
        self._smoother.reset()

    def close(self) -> None:
        """Release the ONNX session reference; ONNX Runtime owns no explicit close API."""
        self._session = None

