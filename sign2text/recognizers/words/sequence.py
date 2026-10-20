"""Fixed-length input and stable isolated-word display state."""

from collections import deque
import math

import numpy as np


class WordSequenceBuffer:
    def __init__(self, sequence_length: int = 40, feature_size: int = 444):
        if sequence_length < 1 or feature_size < 1:
            raise ValueError("sequence_length and feature_size must be positive")
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self._frames = deque(maxlen=sequence_length)

    def append(self, frame):
        values = np.asarray(frame, dtype=np.float32)
        if values.shape != (self.feature_size,):
            raise ValueError(f"Expected frame shape ({self.feature_size},), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("frame contains non-finite values")
        self._frames.append(values.copy())

    @property
    def ready(self):
        return len(self._frames) == self.sequence_length

    @property
    def count(self):
        return len(self._frames)

    def array(self):
        if not self.ready:
            return None
        return np.stack(self._frames, axis=0).astype(np.float32, copy=False)

    def clear(self):
        self._frames.clear()


class StableWordText:
    """Commit one isolated word after repeated confident predictions.

    The display contains one recognized sign at a time; predictions are not joined
    into sentence translation. A held sign cannot append repeatedly.
    """

    def __init__(self, class_names, stable_updates: int = 3, confidence_threshold: float = 0.60):
        if stable_updates < 1:
            raise ValueError("stable_updates must be at least 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.stable_updates = stable_updates
        self.confidence_threshold = confidence_threshold
        self.class_names = frozenset(str(label) for label in class_names)
        self.text = ""
        self._candidate = None
        self._streak = 0
        self._committed = None

    def update(self, label, confidence):
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = float("nan")
        if (not isinstance(label, str) or label not in self.class_names or
                not math.isfinite(confidence) or confidence < self.confidence_threshold):
            self._candidate = None
            self._streak = 0
            return False
        if label != self._candidate:
            self._candidate, self._streak = label, 1
        else:
            self._streak += 1
        if self._streak >= self.stable_updates and label != self._committed:
            self.text = label.upper()
            self._committed = label
            return True
        return False

    @property
    def progress(self):
        return min(self._streak / self.stable_updates, 1.0)

    def clear(self):
        self.text = ""
        self._candidate = None
        self._streak = 0
        self._committed = None


class WordReleaseGate:
    """Allow each label at most once until the hands are released."""

    def __init__(self):
        self._accepted = set()

    def accept(self, label: str) -> bool:
        if not isinstance(label, str) or not label or label in self._accepted:
            return False
        self._accepted.add(label)
        return True

    def reset(self) -> None:
        self._accepted.clear()

