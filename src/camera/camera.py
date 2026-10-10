"""
Sign2Text - Camera Module

Abstract base class and webcam implementation for video capture.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera(ABC):
    """Abstract base class for camera devices."""

    @abstractmethod
    def open(self) -> None:
        """Initialize and open the camera."""
        pass

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera.

        Returns:
            BGR frame as numpy array, or None if read failed.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is open."""
        pass

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


class Webcam(Camera):
    """OpenCV webcam implementation with warm-up and retry logic."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 5,
        max_read_retries: int = 3,
    ) -> None:
        """Initialize webcam with desired settings.

        Args:
            camera_index: Camera device index (0 for default).
            width: Desired frame width.
            height: Desired frame height.
            fps: Desired frames per second.
            warmup_frames: Number of frames to discard during warm-up.
            max_read_retries: Maximum retries for failed frame reads.
        """
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._fps = fps
        self._warmup_frames = warmup_frames
        self._max_read_retries = max_read_retries
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the webcam, configure settings, and warm up."""
        self._capture = cv2.VideoCapture(self._camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera at index {self._camera_index}")

        # Attempt to set resolution and FPS
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture.set(cv2.CAP_PROP_FPS, self._fps)

        # Verify actual settings
        actual_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._capture.get(cv2.CAP_PROP_FPS)
        print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS")

        # Warm up: read and discard initial frames to let camera stabilize
        self._warmup()

    def _warmup(self) -> None:
        """Read and discard initial frames to let camera auto-exposure/white-balance stabilize."""
        for _ in range(self._warmup_frames):
            success, _ = self._capture.read()
            if not success:
                # Non-fatal: camera might need more time, continue anyway
                pass

    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the webcam with retry logic.

        Returns:
            BGR frame, or None if read failed after retries.
        """
        if self._capture is None:
            return None

        for attempt in range(self._max_read_retries):
            success, frame = self._capture.read()
            if success and frame is not None:
                return frame
            # Brief pause before retry
            if attempt < self._max_read_retries - 1:
                import time
                time.sleep(0.01)

        # All retries exhausted
        return None

    def release(self) -> None:
        """Release the webcam."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def is_opened(self) -> bool:
        """Check if webcam is open."""
        return self._capture is not None and self._capture.isOpened()