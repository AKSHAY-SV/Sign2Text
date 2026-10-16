"""
Sign2Text - Prediction Module

Real-time sign prediction with stabilization/debouncing.

The predictor maintains state across frames to ensure that a sign is only
output once it has been consistently detected across multiple consecutive frames.
This prevents the output from flickering/jittering (e.g. HELLO HELLO HELLO
→ HELLO instead of HELLO HELLO HELLO).

Design
------
- Tracks the last N predicted signs with their confidence scores
- Only outputs a sign when it has been stable for STABLE_FRAMES consecutive frames
- Supports a confidence threshold below which predictions are ignored
- Provides a cooldown period to prevent rapid repeated outputs
- Uses temporal voting for more robust stabilization

Parameters are configurable via config/settings.py:
    CONFIDENCE_THRESHOLD - minimum confidence for valid prediction
    STABLE_FRAMES - consecutive frames needed for stability
    PREDICTION_COOLDOWN - ms cooldown between same-sign outputs
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import time
from pathlib import Path

from config.settings import (
    CONFIDENCE_THRESHOLD,
    STABLE_FRAMES,
    PREDICTION_COOLDOWN,
    PROBABILITY_HISTORY_SIZE,
    MIN_CONSISTENT_PREDICTIONS,
    CONFIRMATION_HOLD_SECONDS,
)
from src.features.extractor import FEATURE_VECTOR_SIZE, FeatureExtractor
from src.model.cnn_classifier import CnnClassifier, preprocess_image


class PredictionStabilizer:
    """Stabilizes sign predictions across frames using debouncing and temporal voting.

    Ensures that a sign is only output when it has been consistently
    detected across multiple consecutive frames, preventing flicker/jitter.

    The stabilizer uses a combination of:
    1. Confidence thresholding - only consider predictions above minimum confidence
    2. Consecutive frame voting - require same label for STABLE_FRAMES in a row
    3. Cooldown period - prevent rapid repeated outputs of the same sign
    4. Temporal smoothing - average recent predictions for stability

    Attributes:
        stable_frames: Number of consecutive frames with the same
            prediction required before outputting it.
        confidence_threshold: Minimum confidence score for a prediction
            to be considered valid.
        prediction_cooldown: Minimum milliseconds between outputs of
            the same sign to prevent flicker.
        _history: Recent predictions with timestamps and confidences.
        _last_output_time: Timestamp of last output sign.
        _last_output_label: Label of last output sign.
    """

    def __init__(
        self,
        stable_frames: int = STABLE_FRAMES,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        prediction_cooldown: float = PREDICTION_COOLDOWN,
        probability_history_size: int = PROBABILITY_HISTORY_SIZE,
        min_consistent_predictions: int = MIN_CONSISTENT_PREDICTIONS,
        confirmation_hold_seconds: float = 0.0,
    ) -> None:
        """Initialize the prediction stabilizer.

        Args:
            stable_frames: Number of consecutive frames with the same
                prediction required before outputting it.
            confidence_threshold: Minimum confidence score for a prediction
                to be considered valid.
            prediction_cooldown: Minimum milliseconds between outputs of
                the same sign to prevent flicker.
        """
        self._stable_frames = stable_frames
        self._probability_history_size = max(1, int(probability_history_size))
        self._min_consistent_predictions = max(1, int(min_consistent_predictions))
        self._confirmation_hold_seconds = max(0.0, float(confirmation_hold_seconds))
        self._confidence_threshold = confidence_threshold
        self._prediction_cooldown = prediction_cooldown
        self.confidence_threshold = confidence_threshold

        # Frame history: deque of (label, confidence, timestamp) tuples
        self._history: Deque[Tuple[Optional[str], Optional[float], float]] = deque(
            maxlen=stable_frames * 2
        )

        # Track output state
        self._last_output_time: float = 0.0
        self._last_output_label: Optional[str] = None
        self._consecutive_same: int = 0
        self._candidate_label: Optional[str] = None
        self._candidate_count: int = 0
        self._probability_history: Deque[np.ndarray] = deque(
            maxlen=self._probability_history_size
        )
        self._stable_label: Optional[str] = None
        self._stable_confidence: Optional[float] = None
        self._smoothed_probabilities: Optional[np.ndarray] = None
        self._pending_switch_count = 0
        self._candidate_started_at: Optional[float] = None
        self._last_committed_label: Optional[str] = None

    def update(self, label: Optional[str], confidence: Optional[float]) -> Optional[str]:
        """Update the stabilizer with a new prediction and return the
        stabilized output if ready.

        The method uses temporal voting: it looks at the last N predictions
        and determines the most stable label. A sign is only output when
        it has been consistently detected and enough cooldown time has passed.

        Args:
            label: The predicted sign label, or None if no prediction.
            confidence: The confidence score for the prediction, or None.

        Returns:
            The stabilized sign label to output, or None if not yet stable.
        """
        # Invalid/uncertain input acts as a release to neutral and clears
        # both the candidate run and the current display state.
        if (
            label is None
            or confidence is None
            or not np.isfinite(confidence)
            or confidence < self._confidence_threshold
        ):
            self._history.append((None, None, time.monotonic() * 1000.0))
            self._candidate_label = None
            self._candidate_count = 0
            self._candidate_started_at = None
            self._stable_label = None
            self._stable_confidence = None
            self._last_committed_label = None
            self._probability_history.clear()
            self._smoothed_probabilities = None
            return None

        # Add current valid prediction to history with timestamp.
        timestamp = time.monotonic() * 1000.0
        self._history.append((label, confidence, timestamp))
        if label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = label
            self._candidate_count = 1
            self._candidate_started_at = time.monotonic()

        # Check cooldown - don't output the same sign too quickly
        current_time = time.monotonic() * 1000.0
        cooldown_passed = (
            current_time - self._last_output_time
        ) > self._prediction_cooldown

        # Check if we should output this label
        should_output = False
        if self._candidate_count >= self._stable_frames:
            output_label = self._candidate_label
            if output_label != self._last_output_label:
                should_output = True
            elif cooldown_passed:
                # A repeated sign can be entered again only after neutral or
                # after a fresh stable run and cooldown.
                should_output = self._candidate_count == self._stable_frames

        if should_output and output_label is not None:
            # Update output state
            self._last_output_time = current_time
            self._last_output_label = output_label
            self._consecutive_same = 1

            # Clear history of stabilized output to avoid repeated outputs
            # unless the sign genuinely changes and comes back
            return output_label

        return None

    def update_probabilities(
        self, probabilities: np.ndarray, class_names: list[str]
    ) -> Optional[str]:
        """Smooth CNN probability vectors and accept only a short consensus.

        Returns a stable label, or ``None`` while evidence is uncertain. A
        competing class is shown as uncertain until it wins the configured
        number of smoothed observations, preventing one noisy frame from
        replacing an accepted class.
        """
        scores = np.asarray(probabilities, dtype=np.float32)
        if (scores.ndim != 1 or scores.size != len(class_names) or scores.size == 0
                or not np.all(np.isfinite(scores)) or np.any(scores < 0)
                or not np.isclose(float(scores.sum()), 1.0, atol=1e-3)):
            self.reset()
            raise ValueError("Expected a finite normalized probability vector matching class_names.")

        self._probability_history.append(scores.copy())
        self._smoothed_probabilities = np.mean(self._probability_history, axis=0)
        idx = int(np.argmax(self._smoothed_probabilities))
        label = class_names[idx]
        confidence = float(self._smoothed_probabilities[idx])
        self._history.append((label, confidence, time.monotonic() * 1000.0))

        if confidence < self._confidence_threshold:
            self._candidate_label = None
            self._candidate_count = 0
            self._candidate_started_at = None
            self._stable_label = None
            self._stable_confidence = None
            self._pending_switch_count = 0
            self._last_committed_label = None
            return None

        if label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = label
            self._candidate_count = 1
            self._candidate_started_at = time.monotonic()

        if label != self._stable_label:
            self._pending_switch_count += 1
        else:
            self._pending_switch_count = 0

        if self._candidate_count >= self._min_consistent_predictions:
            held_long_enough = (
                self._candidate_started_at is not None
                and time.monotonic() - self._candidate_started_at >= self._confirmation_hold_seconds
            )
            if held_long_enough:
                self._stable_label = label
                self._stable_confidence = confidence
                self._pending_switch_count = 0
        elif self._stable_label is not None and self._pending_switch_count >= self._probability_history_size:
            # Repeated disagreement without a new consensus is explicitly
            # uncertain instead of presenting the old stable label indefinitely.
            self._stable_label = None
            self._stable_confidence = None

        return self.current_stable_label

    def reset(self) -> None:
        """Reset all state. Call when starting a new session."""
        self._history.clear()
        self._last_output_time = 0.0
        self._last_output_label = None
        self._consecutive_same = 0
        self._candidate_label = None
        self._candidate_count = 0
        self._probability_history.clear()
        self._stable_label = None
        self._stable_confidence = None
        self._smoothed_probabilities = None
        self._pending_switch_count = 0
        self._candidate_started_at = None
        self._last_committed_label = None

    def commit(self, label: Optional[str] = None, explicit: bool = False) -> Optional[str]:
        """Return one confirmed label once; a new label or neutral re-arms it."""
        label = label or self.current_stable_label
        if (label is None or label != self._stable_label
                or (label == self._last_committed_label and not explicit)):
            return None
        self._last_committed_label = label
        return label

    def rearm(self) -> None:
        """Allow the same static sign to be captured again after deliberate release."""
        self._last_committed_label = None

    @property
    def current_stable_label(self) -> Optional[str]:
        """Get the current stabilized label (may be None if not enough frames)."""
        if self._probability_history:
            if (self._candidate_count < self._min_consistent_predictions
                    or self._candidate_label != self._stable_label):
                return None
            return self._stable_label
        if not self._history or self._candidate_label is None or self._candidate_count < self._stable_frames:
            return None
        return self._candidate_label

    @property
    def current_confidence(self) -> Optional[float]:
        if self._probability_history:
            return self._stable_confidence if self.current_stable_label is not None else None
        if self.current_stable_label is not None and self._history:
            return self._history[-1][1]
        return None

    @property
    def smoothed_probabilities(self) -> Optional[np.ndarray]:
        return None if self._smoothed_probabilities is None else self._smoothed_probabilities.copy()

    @property
    def history(self) -> list:
        """Get the prediction history (for debugging)."""
        return list(self._history)


class RealTimePredictor:
    """Handles real-time sign prediction from camera frames.

    The predictor orchestrates the full pipeline:
    - Frame acquisition
    - Hand detection
    - Feature extraction (classical ML) OR CNN classification
    - Prediction stabilization
    - Text buffer update

    Attributes:
        detector: HandDetector instance for landmark detection.
        extractor: FeatureExtractor instance for feature vectors.
        classifier: SignClassifier instance for classical ML prediction.
        cnn_classifier: CnnClassifier instance for CNN-based prediction.
        stabilizer: PredictionStabilizer instance for debouncing.
        current_prediction: The current stabilized sign label.
        current_confidence: The confidence of the current prediction.
        prediction_history: History of recent predictions.
    """

    def __init__(
        self,
        detector: Optional[HandDetector] = None,
        extractor: Optional[FeatureExtractor] = None,
        classifier: Optional[SignClassifier] = None,
        cnn_classifier: Optional[CnnClassifier] = None,
        stabilizer: Optional[PredictionStabilizer] = None,
    ) -> None:
        """Initialize the real-time predictor.

        Args:
            detector: HandDetector instance. Creates default if None.
            extractor: FeatureExtractor instance. Creates default if None.
            classifier: SignClassifier instance. Must be trained/loaded if None.
            cnn_classifier: CnnClassifier instance. Must be trained/loaded if None.
            stabilizer: PredictionStabilizer instance. Creates default if None.
        """
        self._detector = detector or HandDetector()
        self._extractor = extractor or FeatureExtractor()
        self._classifier = classifier
        self._cnn_classifier = cnn_classifier
        self._stabilizer = stabilizer or PredictionStabilizer()

        # State
        self._current_prediction: Optional[str] = None
        self._current_confidence: Optional[float] = None
        self._prediction_history: list = []
        self._cnn_image_size = cnn_classifier.image_size if cnn_classifier else 32
        self._last_cnn_crop: Optional[np.ndarray] = None
        self._last_cnn_input: Optional[np.ndarray] = None
        self._last_cnn_scores: Optional[np.ndarray] = None
        self._last_detection_timestamp_ms: Optional[int] = None

    def predict_from_frame(self, bgr_frame: np.ndarray) -> Tuple[
        Optional[str], Optional[float]
    ]:
        """Run full prediction pipeline on a single frame.

        Args:
            bgr_frame: BGR frame from camera.

        Returns:
            Tuple of (predicted_label, confidence).
        """
        # 1. Detect hands
        detection = self._detector.detect(bgr_frame)

        # 2. Extract features
        if detection is not None and detection.hands:
            feature_vector = self._extractor.extract(detection)
        else:
            feature_vector = np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)

        # 3. Classify
        label, confidence = self._classifier.predict(feature_vector) if self._classifier else (None, None)

        # 4. Stabilize prediction
        stabilized = self._stabilizer.update(label, confidence)

        # Update current state - use stabilized if available, otherwise raw
        self._current_prediction = stabilized if stabilized else label
        if label is None or confidence is None or confidence < self._stabilizer.confidence_threshold:
            self._current_prediction = None
        self._current_confidence = confidence

        # Track history for debugging
        self._prediction_history.append({
            "frame": len(self._prediction_history),
            "raw_label": label,
            "raw_confidence": confidence,
            "stabilized": stabilized,
        })

        return self._current_prediction, self._current_confidence

    def predict_from_features(
        self, feature_vector: np.ndarray
    ) -> Tuple[Optional[str], Optional[float]]:
        """Predict from a pre-extracted feature vector (bypasses detection).

        Args:
            feature_vector: 1D array of shape (FEATURE_VECTOR_SIZE,).

        Returns:
            Tuple of (predicted_label, confidence).
        """
        label, confidence = self._classifier.predict(feature_vector)
        stabilized = self._stabilizer.update(label, confidence)

        self._current_prediction = stabilized if stabilized else (label if label else None)
        if label is None or confidence is None or confidence < self._stabilizer.confidence_threshold:
            self._current_prediction = None
        self._current_confidence = confidence

        return self._current_prediction, self._current_confidence

    def predict_from_frame_cnn(
        self, bgr_frame: np.ndarray, detection=None
    ) -> Tuple[Optional[str], Optional[float]]:
        """Run CNN-based prediction on a camera frame using hand landmarks.

        Crops the detected hand region from the frame, preprocesses it for
        the CNN, and returns the predicted sign label and confidence.

        Args:
            bgr_frame: BGR frame from camera.

        Returns:
            Tuple of (predicted_label, confidence), or (None, None) if
            no hand detected or prediction fails.
        """
        # Never retain a previous crop or score when this frame is invalid.
        self._last_cnn_crop = None
        self._last_cnn_input = None
        self._last_cnn_scores = None

        # 1. Detect hands
        if detection is None:
            detection = self._detector.detect(bgr_frame)

        if (detection is None or not detection.hands
                or detection.frame_width != bgr_frame.shape[1]
                or detection.frame_height != bgr_frame.shape[0]
                or detection.timestamp_ms < 0
                or (self._last_detection_timestamp_ms is not None
                    and detection.timestamp_ms <= self._last_detection_timestamp_ms)):
            self.clear_current_prediction()
            return None, None
        self._last_detection_timestamp_ms = detection.timestamp_ms

        # 2. Crop the first detected hand region from the frame
        hand = detection.hands[0]
        hand_img = self._crop_hand_from_frame(bgr_frame, hand.landmarks)

        if hand_img is None:
            self.clear_current_prediction()
            return None, None

        # 3. Preprocess for CNN
        image_size = self._cnn_classifier.image_size
        preprocessed = preprocess_image(hand_img, target_size=(image_size, image_size))
        self._last_cnn_crop = hand_img.copy()
        self._last_cnn_input = preprocessed
        self._last_cnn_scores = self._cnn_classifier.predict_proba(preprocessed)

        # 4. Predict with CNN
        if self._last_cnn_scores is not None:
            scores = self._last_cnn_scores
            idx = int(np.argmax(scores))
            label = self._cnn_classifier.idx_to_label.get(idx, "unknown")
            confidence = float(scores[idx])
        else:
            label, confidence = self._cnn_classifier.predict(preprocessed)

        if label is None:
            self.clear_current_prediction()
            return None, None

        # 5. Smooth the complete CNN probability vector before accepting a label.
        if self._last_cnn_scores is not None:
            self._current_prediction = self._stabilizer.update_probabilities(
                self._last_cnn_scores, self._cnn_classifier.class_names
            )
            self._current_confidence = self._stabilizer.current_confidence
        else:
            self._current_prediction = self._stabilizer.update(label, confidence)
            self._current_confidence = confidence if self._current_prediction else None

        return self._current_prediction, self._current_confidence

    def clear_current_prediction(self) -> None:
        """Invalidate displayed prediction and all temporal state immediately."""
        self._stabilizer.reset()
        self._current_prediction = None
        self._current_confidence = None
        self._last_cnn_crop = None
        self._last_cnn_input = None
        self._last_cnn_scores = None

    @staticmethod
    def _crop_hand_from_frame(
        bgr_frame: np.ndarray, landmarks: List[object],
    ) -> Optional[np.ndarray]:
        """Crop the hand region from a BGR frame using MediaPipe landmarks.

        Args:
            bgr_frame: BGR frame from camera.
            landmarks: List of 21 MediaPipe landmark objects with .x, .y attributes.

        Returns:
            Cropped hand image as numpy array, or None if cropping fails.
        """
        if (not isinstance(bgr_frame, np.ndarray) or bgr_frame.ndim != 3
                or bgr_frame.shape[2] != 3 or bgr_frame.size == 0
                or not landmarks or len(landmarks) != 21):
            return None
        h, w = bgr_frame.shape[:2]

        # Get bounding box of all landmarks
        try:
            xs = [float(lm.x) * w for lm in landmarks]
            ys = [float(lm.y) * h for lm in landmarks]
        except (AttributeError, TypeError, ValueError):
            return None
        if (not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys))
                or any(x < 0 or x > w for x in xs)
                or any(y < 0 or y > h for y in ys)):
            return None

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Expand to a square crop with generous context, matching the centered
        # hand composition used by the ASL image dataset more closely.
        box_w = max_x - min_x
        box_h = max_y - min_y
        side = max(box_w, box_h) * 1.6
        if side <= 0:
            return None
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        min_x, max_x = center_x - side / 2, center_x + side / 2
        min_y, max_y = center_y - side / 2, center_y + side / 2

        # Keep the intended square size even when it crosses the frame edge.
        # Clipping each edge independently used to make an off-center crop
        # rectangular, which then stretched the hand during resize.
        side_px = max(1, int(np.ceil(side)))
        left = int(np.floor(center_x - side_px / 2))
        top = int(np.floor(center_y - side_px / 2))
        right = left + side_px
        bottom = top + side_px
        x0, x1 = max(0, left), min(w, right)
        y0, y1 = max(0, top), min(h, bottom)
        if x1 <= x0 or y1 <= y0:
            return None

        # Crop the hand region
        hand_roi = bgr_frame[y0:y1, x0:x1]

        if hand_roi.size == 0:
            return None

        pad_top, pad_left = max(0, -top), max(0, -left)
        pad_bottom, pad_right = max(0, bottom - h), max(0, right - w)
        if pad_top or pad_bottom or pad_left or pad_right:
            hand_roi = np.pad(
                hand_roi,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode="edge",
            )
        if hand_roi.shape[0] != hand_roi.shape[1]:
            return None

        return hand_roi

    def set_cnn_classifier(self, cnn_classifier: CnnClassifier, image_size: int) -> None:
        """Set the CNN classifier and expected input image size.

        Args:
            cnn_classifier: Instance of CnnClassifier.
            image_size: Expected input image size (square) for the CNN.
        """
        self._cnn_classifier = cnn_classifier
        self._cnn_image_size = image_size

    def reset_prediction(self) -> None:
        """Reset the prediction stabilizer state."""
        self._stabilizer.reset()
        self._current_prediction = None
        self._current_confidence = None
        self._prediction_history = []
        self._last_detection_timestamp_ms = None
        self._cnn_classifier = None
        self._cnn_image_size = 32

    @property
    def current_prediction(self) -> Optional[str]:
        """Get the current stabilized prediction."""
        return self._current_prediction

    @property
    def current_confidence(self) -> Optional[float]:
        """Get the confidence of the current prediction."""
        return self._current_confidence

    @property
    def prediction_history(self) -> list:
        """Get the prediction history."""
        return self._prediction_history
