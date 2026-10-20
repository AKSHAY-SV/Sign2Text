"""Stable fingerspelling accumulation, separate from model inference."""


class StableTextBuffer:
    def __init__(self, commit_frames: int = 10, confidence_threshold: float = 0.75):
        if commit_frames < 1:
            raise ValueError("commit_frames must be at least 1")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.commit_frames = commit_frames
        self.confidence_threshold = confidence_threshold
        self.text = ""
        self._candidate = None
        self._streak = 0
        self._last_committed = None

    def update(self, label, confidence, *, hand_present=True, moving=False):
        if not hand_present:
            self._candidate = None
            self._streak = 0
            self._last_committed = None
            return False
        if moving or not isinstance(label, str) or len(label) != 1 or label not in "abcdefghijklmnopqrstuvwxyz":
            self._streak = 0
            return False
        if label != self._candidate:
            self._candidate, self._streak = label, 1
            if label != self._last_committed:
                self._last_committed = None
        else:
            self._streak += 1
        if (self._streak >= self.commit_frames
                and confidence >= self.confidence_threshold
                and label != self._last_committed):
            self.text += label.upper()
            self._last_committed = label
            return True
        return False

    @property
    def progress(self):
        return min(self._streak / self.commit_frames, 1.0)

    def backspace(self):
        self.text = self.text[:-1]
        self._last_committed = None

    def clear(self):
        self.text = ""
        self._last_committed = None
        self._candidate = None
        self._streak = 0

    def space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
        self._last_committed = None

    def repeat(self):
        if self.text and self.text[-1] != " ":
            self.text += self.text[-1]
