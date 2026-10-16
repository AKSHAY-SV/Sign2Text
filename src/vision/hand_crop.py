"""Clean RGB hand crops for the candidate image classifier."""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np


class HandCrop(NamedTuple):
    """A square crop plus its source-frame geometry."""

    image: np.ndarray
    bounds: tuple[int, int, int, int]
    landmark_bounds: tuple[int, int, int, int]
    source_shape: tuple[int, int]


def crop_hand_square(
    bgr_frame: np.ndarray,
    landmarks: Sequence[object],
    padding: float = 1.4,
) -> Optional[np.ndarray]:
    """Return a padded square BGR crop or ``None`` for invalid landmarks.

    Coordinates are normalized MediaPipe coordinates in the same frame passed
    in. Padding is applied around the full landmark extent, then out-of-frame
    portions are edge-padded so hands near borders stay square and complete.
    """
    result = crop_hand_with_bounds(bgr_frame, landmarks, padding=padding)
    return None if result is None else result.image


def crop_hand_with_bounds(
    bgr_frame: np.ndarray,
    landmarks: Sequence[object],
    padding: float = 1.4,
) -> Optional[HandCrop]:
    """Crop the padded square around all 21 landmarks and return source bounds.

    ``bounds`` are the requested square ROI as ``(left, top, right, bottom)``;
    they can extend outside the source frame because those pixels are edge-padded.
    ``landmark_bounds`` are the integer enclosure of the detected hand points.
    """
    if (not isinstance(bgr_frame, np.ndarray) or bgr_frame.ndim != 3
            or bgr_frame.shape[2] != 3 or bgr_frame.size == 0
            or not isinstance(padding, (int, float)) or padding < 1
            or landmarks is None or len(landmarks) != 21):
        return None
    h, w = bgr_frame.shape[:2]
    try:
        xs = np.asarray([float(point.x) * w for point in landmarks], dtype=np.float64)
        ys = np.asarray([float(point.y) * h for point in landmarks], dtype=np.float64)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None
    if (not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys))
            or np.any(xs < 0) or np.any(xs > w)
            or np.any(ys < 0) or np.any(ys > h)):
        return None
    width, height = float(xs.max() - xs.min()), float(ys.max() - ys.min())
    side = int(np.ceil(max(width, height) * float(padding)))
    if side < 8 or max(width, height) / side < 0.65:
        return None
    cx, cy = (float(xs.min() + xs.max()) / 2, float(ys.min() + ys.max()) / 2)
    left, top = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
    right, bottom = left + side, top + side
    x0, x1 = max(0, left), min(w, right)
    y0, y1 = max(0, top), min(h, bottom)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = bgr_frame[y0:y1, x0:x1]
    pads = (max(0, -top), max(0, bottom - h),
            max(0, -left), max(0, right - w))
    if any(pads):
        crop = np.pad(crop, ((pads[0], pads[1]), (pads[2], pads[3]), (0, 0)), mode="edge")
    if crop.shape[0] != crop.shape[1] or crop.size == 0:
        return None
    landmark_bounds = (
        int(np.floor(xs.min())), int(np.floor(ys.min())),
        int(np.ceil(xs.max())) + 1, int(np.ceil(ys.max())) + 1,
    )
    return HandCrop(
        np.ascontiguousarray(crop),
        (left, top, right, bottom),
        landmark_bounds,
        (h, w),
    )
