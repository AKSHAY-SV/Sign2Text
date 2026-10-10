"""
Sign2Text - Feature Extraction Module

Converts a `HandDetection` (as produced by `app.detection.detector.HandDetector`)
into a fixed-length, wrist-relative, scale-normalized feature vector suitable
for downstream ML classification.

This module depends only on the stable `HandDetection` / `HandResult` /
`Landmark3D` dataclasses exposed by detector.py, never on MediaPipe's
native result types. This keeps feature extraction fully decoupled from
the MediaPipe Tasks API surface.
"""

from __future__ import annotations

from typing import List

import numpy as np

from src.detection.detector import HandDetection, HandResult

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

NUM_LANDMARKS: int = 21
COORDS_PER_LANDMARK: int = 3
FEATURES_PER_HAND: int = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
MAX_HANDS: int = 2
FEATURE_VECTOR_SIZE: int = FEATURES_PER_HAND * MAX_HANDS  # 126

WRIST_INDEX: int = 0
MIDDLE_FINGER_MCP_INDEX: int = 9

LEFT_SLOT: int = 0
RIGHT_SLOT: int = 1

# Minimum scale denominator to avoid division-by-zero on degenerate
# (e.g. fully collapsed) landmark sets.
_EPSILON: float = 1e-6


class FeatureExtractor:
    """Extracts fixed-length, normalized feature vectors from a `HandDetection`.

    The extractor converts a `HandDetection` containing 0, 1, or 2 detected
    hands into a 126-dimensional feature vector:

        [hand_left_63_features, hand_right_63_features]

    Each hand's 63 features are wrist-relative (translation-normalized) and
    scaled by the wrist-to-middle-finger-MCP distance (scale-normalized).
    Missing hands are zero-padded. Hand ordering within the output vector is
    deterministic by handedness label ("Left" -> slot 0, "Right" -> slot 1),
    not by detection order, so the feature layout is stable across frames.
    A hand reported with an "Unknown" label (see detector.py, when
    MediaPipe returns no handedness category) falls back to the "Right"
    slot to guarantee a valid, bounded index; if both a genuine "Right"
    hand and an "Unknown" hand appear in the same frame, the later one in
    `detection.hands` overwrites the earlier one in that slot.
    """

    def extract(self, detection: HandDetection) -> np.ndarray:
        """Extracts a fixed-length feature vector from a `HandDetection`.

        Args:
            detection: The `HandDetection` produced by `HandDetector.detect`
                for a single frame. May contain zero, one, or two hands.

        Returns:
            A numpy array of shape (126,) and dtype float32. Each hand
            contributes 63 values (21 landmarks x 3 coords), ordered as
            [Left(63), Right(63)]. Absent hands are zero-filled.

        Raises:
            ValueError: If a `HandResult` does not contain exactly
                `NUM_LANDMARKS` landmarks.
        """
        feature_vector = np.zeros(FEATURE_VECTOR_SIZE, dtype=np.float32)

        if detection is None or not detection.hands:
            return feature_vector

        for hand in detection.hands:
            slot = self._resolve_slot(hand.label)
            hand_features = self._extract_single_hand(hand)

            start = slot * FEATURES_PER_HAND
            end = start + FEATURES_PER_HAND
            feature_vector[start:end] = hand_features

        return feature_vector

    @staticmethod
    def _resolve_slot(label: str) -> int:
        """Maps a handedness label to a stable output slot index.

        Args:
            label: The `HandResult.label` value, expected to be "Left",
                "Right", or "Unknown" (case-insensitive).

        Returns:
            0 for "Left", 1 for "Right" or any other/unrecognized label.
        """
        if label.strip().lower() == "left":
            return LEFT_SLOT
        return RIGHT_SLOT

    @staticmethod
    def _extract_single_hand(hand: HandResult) -> np.ndarray:
        """Normalizes one `HandResult`'s landmarks into a 63-dim vector.

        Args:
            hand: A single detected hand with exactly `NUM_LANDMARKS`
                image-space landmarks.

        Returns:
            A numpy array of shape (63,) and dtype float32, wrist-relative
            and scale-normalized.

        Raises:
            ValueError: If `hand.landmarks` does not contain exactly
                `NUM_LANDMARKS` entries.
        """
        landmarks = hand.landmarks
        if len(landmarks) != NUM_LANDMARKS:
            raise ValueError(
                f"Expected {NUM_LANDMARKS} landmarks, got {len(landmarks)}."
            )

        coords = np.array(
            [[point.x, point.y, point.z] for point in landmarks],
            dtype=np.float32,
        )  # shape: (21, 3)

        wrist = coords[WRIST_INDEX]
        translated = coords - wrist

        scale_reference = translated[MIDDLE_FINGER_MCP_INDEX]
        scale = float(np.linalg.norm(scale_reference))
        scale = max(scale, _EPSILON)

        normalized = translated / scale

        return normalized.flatten()