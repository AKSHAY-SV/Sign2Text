"""
Sign2Text - Hand Visualizer Module

Renders HandDetection results onto BGR video frames using MediaPipe's
Tasks-API-native drawing utilities and the official hand connection
topology (`mediapipe.tasks.vision.HandLandmarksConnections`).

This module uses the Tasks API drawing utilities, which are the modern
MediaPipe way to draw landmarks (the legacy `mediapipe.solutions` module
is not available in current MediaPipe wheels for newer Python versions).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import (
    NormalizedLandmark,
)
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections

from src.detection.detector import HandDetection, HandResult

# Color constants (BGR format for OpenCV)
_LEFT_HAND_COLOR: Tuple[int, int, int] = (255, 144, 30)   # Orange-ish
_RIGHT_HAND_COLOR: Tuple[int, int, int] = (60, 200, 60)   # Green-ish
_TEXT_COLOR: Tuple[int, int, int] = (255, 255, 255)       # White
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FPS_SMOOTHING = 0.9


class HandVisualizer:
    """Draws the full hand skeleton, handedness, confidence, and FPS."""

    def __init__(self) -> None:
        # Official connection graph for the 21 hand landmarks, shipped by
        # the Tasks API itself -- not hardcoded line pairs.
        self._connections = HandLandmarksConnections.HAND_CONNECTIONS

        self._landmark_style = {
            "left": drawing_utils.DrawingSpec(
                color=_LEFT_HAND_COLOR, thickness=2, circle_radius=3
            ),
            "right": drawing_utils.DrawingSpec(
                color=_RIGHT_HAND_COLOR, thickness=2, circle_radius=3
            ),
        }
        self._connection_style = {
            "left": drawing_utils.DrawingSpec(
                color=_LEFT_HAND_COLOR, thickness=2
            ),
            "right": drawing_utils.DrawingSpec(
                color=_RIGHT_HAND_COLOR, thickness=2
            ),
        }

        self._prev_frame_time: Optional[float] = None
        self._fps: float = 0.0

    def draw(self, frame: np.ndarray, detection: HandDetection, show_landmarks: bool = True) -> np.ndarray:
        """Draws all detected hands plus an FPS counter onto `frame`.

        Mutates and returns `frame` in place to avoid an extra frame
        allocation per call.

        Args:
            frame: BGR frame to annotate (same frame the detector ran on).
            detection: Detector output for that frame.
            show_landmarks: Whether to draw hand landmarks and skeleton.

        Returns:
            The same `frame` object, annotated.
        """
        self._update_fps()

        if show_landmarks:
            for hand in detection.hands:
                self._draw_hand(frame, hand)

        cv2.putText(
            frame,
            f"FPS: {self._fps:.1f}",
            (10, 30),
            _FONT,
            0.7,
            _TEXT_COLOR,
            2,
            cv2.LINE_AA,
        )
        return frame

    def _draw_hand(self, frame: np.ndarray, hand: HandResult) -> None:
        """Draws a single hand's skeleton and text label."""
        landmark_list = [
            NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand.landmarks
        ]

        side = "left" if hand.label.lower() == "left" else "right"

        drawing_utils.draw_landmarks(
            frame,
            landmark_list,
            self._connections,
            self._landmark_style[side],
            self._connection_style[side],
        )

        self._draw_label(frame, hand, side)

    def _draw_label(self, frame: np.ndarray, hand: HandResult, side: str) -> None:
        """Draws the handedness + confidence text above the hand."""
        if not hand.landmarks:
            return

        height, width = frame.shape[:2]
        xs = [lm.x for lm in hand.landmarks]
        ys = [lm.y for lm in hand.landmarks]
        text_x = int(min(xs) * width)
        text_y = max(int(min(ys) * height) - 10, 15)

        color = _LEFT_HAND_COLOR if side == "left" else _RIGHT_HAND_COLOR
        display_label = hand.label

        # Webcam preview is mirrored, so swap displayed labels.
        if display_label.lower() == "left":
            display_label = "Right"
        elif display_label.lower() == "right":
            display_label = "Left"

        label_text = f"{display_label} ({hand.confidence:.2f})"
        cv2.putText(
            frame, label_text, (text_x, text_y), _FONT, 0.6, color, 2, cv2.LINE_AA
        )

    def _update_fps(self) -> None:
        """Updates an exponentially smoothed FPS estimate."""
        now = time.monotonic()
        if self._prev_frame_time is not None:
            delta = now - self._prev_frame_time
            if delta > 0:
                instantaneous_fps = 1.0 / delta
                self._fps = (
                    instantaneous_fps
                    if self._fps == 0.0
                    else (_FPS_SMOOTHING * self._fps + (1 - _FPS_SMOOTHING) * instantaneous_fps)
                )
        self._prev_frame_time = now