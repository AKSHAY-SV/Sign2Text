"""
Sign2Text - Hand Detection Module

Production-grade hand landmark detector built on the official MediaPipe
Tasks API (mediapipe.tasks.python.vision.HandLandmarker).

This module is intentionally decoupled from drawing/visualization and
from any downstream consumer. It exposes a small, plain-dataclass
result type (HandDetection) instead of leaking MediaPipe's native
result object, so that future modules (feature extraction, gesture
classification, sentence generation, dataset recording, UART
communication) can depend on a stable, serializable contract rather
than on MediaPipe internals.

Only the officially documented MediaPipe Tasks Python surface is used:
    mp.tasks.BaseOptions
    mp.tasks.vision.HandLandmarker
    mp.tasks.vision.HandLandmarkerOptions
    mp.tasks.vision.HandLandmarkerResult
    mp.tasks.vision.RunningMode
    HandLandmarker.detect_for_video(image, timestamp_ms)
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from typing import List, Union, Optional

import cv2
import mediapipe as mp
import numpy as np

from config.settings import (
    MAX_NUM_HANDS,
    MIN_HAND_DETECTION_CONFIDENCE,
    MIN_HAND_PRESENCE_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MODEL_PATH,
)

# Bind the public Tasks API surface explicitly, exactly as documented by
# Google (developers.google.com/mediapipe/solutions/vision/hand_landmarker).
_BaseOptions = mp.tasks.BaseOptions
_HandLandmarker = mp.tasks.vision.HandLandmarker
_HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
_HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
_RunningMode = mp.tasks.vision.RunningMode

#: Default hand landmarker model path. If the file does not exist at
#: startup, the detector will be initialised with ``running_mode=_RunningMode.IMAGE``
#: but ``_landmarker`` will be set to ``None``, allowing the pipeline to continue
#: in a degraded mode. Set this path to a valid ``.task`` file to enable full
#: hand landmark detection.
DEFAULT_HAND_LANDMARKER_PATH = Path("models") / "hand_landmarker.task"


@dataclasses.dataclass(frozen=True)
class Landmark3D:
    """A single 3D landmark point, decoupled from MediaPipe's proto type."""

    x: float
    y: float
    z: float


@dataclasses.dataclass(frozen=True)
class HandResult:
    """Detection output for a single hand."""

    label: str                     # "Left" or "Right"
    confidence: float              # handedness classification score
    landmarks: List[Landmark3D]    # 21 normalized image-space landmarks
    world_landmarks: List[Landmark3D]  # 21 metric world-space landmarks


@dataclasses.dataclass(frozen=True)
class HandDetection:
    """Complete detector output for a single processed frame."""

    hands: List[HandResult]
    frame_width: int
    frame_height: int
    timestamp_ms: int

    @property
    def num_hands(self) -> int:
        return len(self.hands)


class HandDetector:
    """Runs MediaPipe HandLandmarker in VIDEO mode on a sequence of frames.

    VIDEO mode is used (rather than IMAGE mode) because incoming webcam
    frames form a temporally ordered stream: HandLandmarker's internal
    tracker uses the previous frame's result to avoid re-running the
    (expensive) palm detector on every single frame, which is what makes
    two-hand tracking stable and real-time capable.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_HAND_LANDMARKER_PATH,
        num_hands: int = MAX_NUM_HANDS,
        min_hand_detection_confidence: float = MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence: float = MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence: float = MIN_TRACKING_CONFIDENCE,
    ) -> None:
        """Loads the .task model and creates the HandLandmarker instance.

        Args:
            model_path: Path to hand_landmarker.task. If the file does not
                exist, the detector is initialised with ``_landmarker = None``
                and a warning is printed. The pipeline will run in a degraded
                mode without hand landmark detection.
            num_hands: Maximum number of hands to detect simultaneously.
            min_hand_detection_confidence: Minimum confidence for the
                initial palm detection to be accepted.
            min_hand_presence_confidence: Minimum confidence for hand
                presence during landmark detection.
            min_tracking_confidence: Minimum confidence for the tracker
                to keep following a hand across frames.

        Raises:
            FileNotFoundError: If the model file does not exist and
                falling back is not possible.
        """
        model_path = Path(model_path)
        self._landmarker: Optional[_HandLandmarker] = None
        self._running_mode: any = None
        self._start_time_ms: float = 0.0
        self._last_timestamp_ms: int = -1

        if not model_path.is_file():
            print(
                f"Warning: HandLandmarker model not found at: {model_path}"
            )
            print(
                "  See https://mediapipe.dev/models for model download instructions."
            )
            print(
                "  Disabling hand detection. The CNN pipeline will use "
                "alternative preprocessing (center crop / full frame)."
            )
            # Initialise with None landmarker - detect() will return
            # a detection with zero hands.
            return

        if not model_path.parent.is_dir():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = _HandLandmarker.create_from_options(options)

        # Monotonic clock anchored at construction time so timestamps are
        # small, strictly increasing integers, as required by
        # detect_for_video().
        self._start_time_ms: float = time.monotonic() * 1000.0
        self._last_timestamp_ms: int = -1

    def _next_timestamp_ms(self) -> int:
        """Produces a strictly increasing millisecond timestamp."""
        timestamp_ms = int(time.monotonic() * 1000.0 - self._start_time_ms)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def detect(self, bgr_frame: np.ndarray) -> HandDetection:
        """Runs detection on a single BGR frame (as read by OpenCV).

        Args:
            bgr_frame: Frame in OpenCV's native BGR channel order.

        Returns:
            A HandDetection with zero or more detected hands. If the
            landmarker model was not loaded, returns a detection with
            zero hands.
        """
        # If the model failed to load, return a detection with zero hands.
        if self._landmarker is None:
            height, width = bgr_frame.shape[:2]
            return HandDetection(
                hands=[],
                frame_width=width,
                frame_height=height,
                timestamp_ms=int(time.monotonic() * 1000.0),
            )

        # MediaPipe Tasks expects RGB. This single conversion per frame
        # is unavoidable, so it is done exactly once and nowhere else.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = self._next_timestamp_ms()
        raw_result: _HandLandmarkerResult = self._landmarker.detect_for_video(
            mp_image, timestamp_ms
        )

        height, width = bgr_frame.shape[:2]
        return self._to_hand_detection(raw_result, width, height, timestamp_ms)

    @staticmethod
    def _to_hand_detection(
        raw_result: _HandLandmarkerResult,
        width: int,
        height: int,
        timestamp_ms: int,
    ) -> HandDetection:
        """Converts MediaPipe's native result into the stable HandDetection."""
        hands: List[HandResult] = []

        for hand_index, hand_landmarks in enumerate(raw_result.hand_landmarks):
            world_landmarks = (
                raw_result.hand_world_landmarks[hand_index]
                if hand_index < len(raw_result.hand_world_landmarks)
                else []
            )
            handedness_categories = (
                raw_result.handedness[hand_index]
                if hand_index < len(raw_result.handedness)
                else []
            )
            top_category = (
                handedness_categories[0] if handedness_categories else None
            )
            label = top_category.category_name if top_category else "Unknown"
            confidence = float(top_category.score) if top_category else 0.0

            hands.append(
                HandResult(
                    label=label,
                    confidence=confidence,
                    landmarks=[
                        Landmark3D(lm.x, lm.y, lm.z) for lm in hand_landmarks
                    ],
                    world_landmarks=[
                        Landmark3D(lm.x, lm.y, lm.z) for lm in world_landmarks
                    ],
                )
            )

        return HandDetection(
            hands=hands,
            frame_width=width,
            frame_height=height,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        """Releases the underlying MediaPipe task graph."""
        if self._landmarker is not None:
            self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()