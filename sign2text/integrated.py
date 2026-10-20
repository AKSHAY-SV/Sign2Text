"""Small, dependency-light helpers shared by the integrated Sign2Text UI."""

from __future__ import annotations

from dataclasses import dataclass
import math
from queue import Empty, Full, Queue
import struct
from typing import BinaryIO

import numpy as np


_HEADER = struct.Struct("!IIIII")  # sequence, width, height, byte count, accept flag
MAX_FRAME_BYTES = 4096 * 2160 * 3


def encode_frame(sequence: int, frame: np.ndarray, *, accept_input: bool = True) -> tuple[bytes, bytes]:
    """Return protocol header and contiguous raw uint8 BGR frame bytes."""
    values = np.asarray(frame)
    if values.dtype != np.uint8 or values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("expected a uint8 HxWx3 BGR frame")
    height, width = values.shape[:2]
    if width <= 0 or height <= 0 or values.nbytes > MAX_FRAME_BYTES:
        raise ValueError("invalid or oversized frame")
    payload = np.ascontiguousarray(values).tobytes()
    return _HEADER.pack(int(sequence), width, height, len(payload), int(bool(accept_input))), payload


def _read_exact(stream: BinaryIO, size: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if not chunks:
                return None
            raise EOFError("truncated frame protocol message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class FrameMessage:
    sequence: int
    frame: np.ndarray
    accept_input: bool


def read_frame(stream: BinaryIO) -> FrameMessage | None:
    """Read one validated protocol frame; return None on clean EOF."""
    header = _read_exact(stream, _HEADER.size)
    if header is None:
        return None
    sequence, width, height, byte_count, accept_flag = _HEADER.unpack(header)
    expected = width * height * 3
    if (accept_flag not in (0, 1) or width <= 0 or height <= 0 or expected > MAX_FRAME_BYTES
            or byte_count != expected):
        raise ValueError("invalid frame protocol dimensions")
    payload = _read_exact(stream, byte_count)
    if payload is None:
        raise EOFError("missing frame payload")
    frame = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3)
    return FrameMessage(sequence, frame, bool(accept_flag))


def enqueue_latest(target: Queue, item: object) -> bool:
    """Put an item in a bounded queue, discarding its stale pending item if full."""
    try:
        target.put_nowait(item)
        return False
    except Full:
        try:
            target.get_nowait()
        except Empty:
            pass
        target.put_nowait(item)
        return True


class TextAccumulator:
    """Shared visible text; word tokens and fingerspelled letters stay distinct."""

    def __init__(self) -> None:
        self._text = ""
        self._last_kind: str | None = None

    @property
    def text(self) -> str:
        return self._text

    def append_word(self, word: str) -> None:
        token = word.strip().upper()
        if token:
            self._text += (" " if self._text and not self._text.endswith(" ") else "") + token
            self._last_kind = "word"

    def append_letter(self, letter: str) -> None:
        token = letter.strip().upper()
        if len(token) != 1 or not token.isalpha():
            raise ValueError("expected one alphabetic letter")
        if self._last_kind == "word":
            self.space()
        self._text += token
        self._last_kind = "letter"

    def space(self) -> None:
        if self._text and not self._text.endswith(" "):
            self._text += " "

    def backspace(self) -> None:
        self._text = self._text[:-1]

    def clear(self) -> None:
        self._text = ""
        self._last_kind = None


def apply_worker_commit(
    text: TextAccumulator, record: object, *, selected_mode: str,
    minimum_sequence: int = 0,
) -> bool:
    """Append an accepted token only if its mode is still selected by the UI.

    ``input_enabled`` reflects the mode when a frame entered a worker. Checking
    ``selected_mode`` as well prevents a delayed IPC result from the previous
    mode being committed after the user has switched modes.
    """
    if not isinstance(record, dict) or record.get("input_enabled") is not True:
        return False
    if selected_mode not in ("word", "alphabet") or record.get("mode") != selected_mode:
        return False
    sequence = record.get("sequence")
    if (not isinstance(sequence, int) or isinstance(sequence, bool)
            or sequence < minimum_sequence):
        return False
    committed = record.get("committed")
    try:
        if record.get("mode") == "word" and isinstance(committed, str):
            text.append_word(committed)
            return bool(committed.strip())
        if record.get("mode") == "alphabet" and isinstance(committed, str):
            text.append_letter(committed)
            return True
    except ValueError:
        return False
    return False


def apply_text_key(text: TextAccumulator, raw_key: int) -> bool:
    """Apply text-edit keys from OpenCV; a negative key means no key was pressed."""
    if not isinstance(raw_key, int) or raw_key < 0:
        return False
    key = raw_key & 0xFF
    if key in (ord("c"), ord("C")):
        text.clear()
        return True
    if key in (8, 127):
        text.backspace()
        return True
    if key == ord(" "):
        text.space()
        return True
    return False


def presentation_fields(
    mode: str,
    latest_by_mode: dict,
    worker_status: dict,
    text: TextAccumulator,
    *,
    minimum_sequence: int = 0,
) -> tuple[tuple[str, str], ...]:
    """Return the concise user-facing fields; keep worker diagnostics internal."""
    if mode not in ("word", "alphabet"):
        raise ValueError("mode must be 'word' or 'alphabet'")

    def prediction(record: object) -> str:
        if not isinstance(record, dict):
            return "waiting"
        value = record.get("prediction")
        if not isinstance(value, dict):
            return "waiting"
        label = value.get("label")
        confidence = value.get("confidence")
        if (not isinstance(label, str) or not isinstance(confidence, (int, float))
                or not math.isfinite(confidence) or not 0 <= confidence <= 1):
            return "waiting"
        return f"{label.upper()} {confidence:.1%}"

    active = latest_by_mode.get(mode)
    if (isinstance(active, dict)
            and (not isinstance(active.get("sequence"), int)
                 or active["sequence"] < minimum_sequence)):
        active = None
    state = str(worker_status.get(mode, "loading"))
    if state.startswith("error:"):
        status = "Word worker error" if mode == "word" else "Letter worker error"
    elif state != "ready":
        status = "Starting word recognizer" if mode == "word" else "Starting letter recognizer"
    elif isinstance(active, dict):
        raw = active.get("status", "waiting")
        status = {
            "no hand detected": "Show a hand",
            "collecting": "Preparing word recognition",
            "low confidence": "Low confidence",
            "stabilizing": "Stabilizing word" if mode == "word" else "Stabilizing letter",
            "motion": "Hold still or complete the motion",
            "accepted": "Accepted",
            "release to repeat": "Release hand to repeat",
        }.get(raw, "Preparing word recognition" if isinstance(raw, str) and raw.startswith("collecting ")
              else "Ready")
    else:
        status = "Waiting for selected mode" if state == "ready" else "Waiting"

    return (
        ("Input mode", "WORD" if mode == "word" else "SPELL"),
        ("Word", prediction(latest_by_mode.get("word"))),
        ("Letter", prediction(latest_by_mode.get("alphabet"))),
        ("Status", status),
        ("Text", text.text or "(empty)"),
    )


def valid_worker_result(
    record: object, expected_mode: str, known_labels: set[str] | frozenset[str] | None = None
) -> bool:
    """Validate the worker result envelope before the UI formats or accepts it."""
    if not isinstance(record, dict) or record.get("mode") != expected_mode:
        return False
    if (not isinstance(record.get("sequence"), int)
            or not isinstance(record.get("status"), str)
            or type(record.get("input_enabled")) is not bool):
        return False
    progress = record.get("progress")
    if (not isinstance(progress, (int, float)) or not math.isfinite(progress)
            or not 0 <= progress <= 1):
        return False
    prediction = record.get("prediction")
    if prediction is not None and not _valid_ranked_prediction(prediction, known_labels):
        return False
    top = record.get("top")
    if (not isinstance(top, list)
            or not all(_valid_ranked_prediction(item, known_labels) for item in top)):
        return False
    committed = record.get("committed")
    return (committed is None or
            (isinstance(committed, str)
             and (known_labels is None or committed in known_labels)))


def _valid_ranked_prediction(
    value: object, known_labels: set[str] | frozenset[str] | None = None
) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("label"), str):
        return False
    if known_labels is not None and value["label"] not in known_labels:
        return False
    confidence = value.get("confidence")
    return (isinstance(confidence, (int, float)) and math.isfinite(confidence)
            and 0 <= confidence <= 1)


def switch_input_mode(text: TextAccumulator, previous: str, selected: str) -> str:
    """Switch manual input mode, separating a completed word before spelling."""
    if selected not in ("word", "alphabet"):
        raise ValueError("mode must be 'word' or 'alphabet'")
    if previous == "word" and selected == "alphabet":
        text.space()
    return selected
