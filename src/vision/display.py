"""Helpers for preparing image canvases used by OpenCV display/drawing."""

from __future__ import annotations

import numpy as np


def to_display_uint8(image: np.ndarray) -> np.ndarray:
    """Return an 8-bit display copy without modifying model input arrays.

    Float images in [0, 1] are scaled to [0, 255]. Other numeric arrays are
    clipped to the representable 8-bit range before conversion.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Display image must be a 2D or 3D NumPy array.")
    if image.dtype == np.uint8:
        return image.copy()
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError(f"Unsupported display image dtype: {image.dtype}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Display image contains non-finite values.")

    values = image.astype(np.float32, copy=False)
    if values.size and values.min() >= 0.0 and values.max() <= 1.0:
        values = values * 255.0
    return np.rint(np.clip(values, 0.0, 255.0)).astype(np.uint8)
