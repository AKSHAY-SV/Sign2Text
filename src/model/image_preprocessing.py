"""Shared RGB resize and normalization used by CNN training and inference."""

from __future__ import annotations

import tensorflow as tf


RESIZE_METHOD = "bilinear antialiased square resize"
NORMALIZATION = "float32 pixel values divided by 255, range [0,1]"


def preprocess_rgb_tensor(image: tf.Tensor, image_size: int | tuple[int, int]) -> tf.Tensor:
    """Convert decoded RGB uint8 pixels to resized float32 RGB in [0, 1]."""
    image = tf.cast(image, tf.float32)
    target = [image_size, image_size] if isinstance(image_size, int) else list(image_size)
    if len(target) != 2 or min(target) <= 0:
        raise ValueError("image_size must be a positive int or (height, width) pair")
    image = tf.image.resize(image, target, method="bilinear", antialias=True)
    # TensorFlow's resize kernel can overshoot the input extrema by a tiny
    # floating-point amount (for example 255.00003). Keep the documented
    # training/inference contract exactly within [0, 1].
    return tf.clip_by_value(image / 255.0, 0.0, 1.0)
