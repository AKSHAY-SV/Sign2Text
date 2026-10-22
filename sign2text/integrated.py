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
HAND_IDENTITY_CONFIDENCE = 0.85
HAND_OVERLAP_RATIO = 0.20
MAX_CONTROLLER_OBSERVATION_GAP_SECONDS = 0.25
CONTROLLER_COOLDOWN_SECONDS = 3.0


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


@dataclass(frozen=True)
class PendingCandidate:
    mode: str
    label: str
    confidence: float
    sequence: int


class ConfirmationController:
    """One-shot open-palm confirmation with cooldown and neutral re-arm."""

    def __init__(self, hold_seconds: float = 0.3,
                 cooldown_seconds: float = CONTROLLER_COOLDOWN_SECONDS):
        if hold_seconds <= 0:
            raise ValueError("hold_seconds must be positive")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds cannot be negative")
        self.hold_seconds = hold_seconds
        self.cooldown_seconds = cooldown_seconds
        self._gesture: str | None = None
        self._started_at: float | None = None
        self._readiness_started_at: float | None = None
        self._phase = "armed"
        self._cooldown_until: float | None = None
        self._candidate_key: tuple[str, str] | None = None
        self._last_observation_sequence: int | None = None
        self._last_observation_time: float | None = None

    @property
    def state(self) -> str:
        if self._phase == "cooldown":
            return "cooldown"
        if self._phase == "wait_neutral":
            return "waiting for release"
        return "holding open palm" if self._started_at is not None else "ready"

    def cooldown_remaining(self, now: float) -> float:
        if self._cooldown_until is None:
            return 0.0
        return max(0.0, self._cooldown_until - now)

    def in_cooldown(self, now: float) -> bool:
        return self.cooldown_remaining(now) > 0

    def begin_cooldown(self, now: float) -> None:
        self._cooldown_until = now + self.cooldown_seconds
        self._phase = "cooldown"
        self._gesture = None
        self._started_at = None
        self._readiness_started_at = None
        self._last_observation_sequence = None
        self._last_observation_time = None

    def update(self, gesture: str | None, now: float, *, handedness_reliable: bool,
               candidate: PendingCandidate | None,
               observation_sequence: int | None = None) -> str | None:
        if not handedness_reliable:
            self._gesture = None
            self._started_at = None
            self._readiness_started_at = None
            return None
        if observation_sequence is not None:
            if (self._last_observation_sequence is not None
                    and observation_sequence <= self._last_observation_sequence):
                return None
            if (self._last_observation_time is not None
                    and now - self._last_observation_time > MAX_CONTROLLER_OBSERVATION_GAP_SECONDS):
                self._gesture = None
                self._started_at = None
                self._readiness_started_at = None
            self._last_observation_sequence = observation_sequence
            self._last_observation_time = now
        if self._phase == "cooldown":
            if self.in_cooldown(now):
                return None
            self._phase = "wait_neutral"
        candidate_key = (candidate.mode, candidate.label) if candidate else None
        if candidate_key != self._candidate_key:
            self._candidate_key = candidate_key
            self._gesture = None
            self._started_at = None
        if gesture == "neutral":
            self._gesture = None
            self._started_at = None
            self._readiness_started_at = None
            if self._phase == "wait_neutral":
                self._phase = "armed"
            return None
        if self._phase == "wait_neutral":
            return None
        if gesture != "open_palm" or candidate is None:
            self._gesture = None
            self._started_at = None
            return None
        if gesture != self._gesture:
            self._gesture = gesture
            self._started_at = now
            return None
        if self._started_at is not None and now - self._started_at >= self.hold_seconds:
            return "confirm"
        return None

    def reset(self) -> None:
        self._gesture = None
        self._started_at = None
        self._phase = "armed"
        self._cooldown_until = None
        self._candidate_key = None
        self._last_observation_sequence = None
        self._last_observation_time = None
        self._readiness_started_at = None


def word_candidate_allowed(record: object) -> bool:
    """Require the worker's verified anatomical-right Holistic hand result."""
    return (isinstance(record, dict)
            and record.get("mode") == "word"
            and record.get("input_enabled") is True
            and record.get("signing_hand") == "Right"
            and record.get("signing_hand_verified") is True)


def apply_candidate_action(text: TextAccumulator, candidate: PendingCandidate | None,
                           action: str) -> bool:
    """Commit the candidate once or discard it without changing committed text."""
    if candidate is None or action not in ("confirm", "reject"):
        return False
    if action == "reject":
        return True
    try:
        if candidate.mode == "word":
            text.append_word(candidate.label)
        elif candidate.mode == "alphabet":
            text.append_letter(candidate.label)
        else:
            return False
    except ValueError:
        return False
    return True


def apply_candidate_key(text: TextAccumulator, candidate: PendingCandidate | None,
                        raw_key: int) -> str | None:
    """Keyboard fallback: +/= confirms, -/R rejects the pending candidate."""
    if not isinstance(raw_key, int) or raw_key < 0:
        return None
    key = raw_key & 0xFF
    action = "confirm" if key in (ord("+"), ord("=")) else (
        "reject" if key in (ord("-"), ord("r"), ord("R")) else None)
    if action is not None and apply_candidate_action(text, candidate, action):
        return action
    return None


def candidate_from_result(record: object, *, selected_mode: str,
                          minimum_sequence: int = 0) -> PendingCandidate | None:
    if (not isinstance(record, dict) or record.get("mode") != selected_mode
            or record.get("input_enabled") is not True):
        return None
    sequence = record.get("sequence")
    prediction = record.get("prediction")
    if (not isinstance(sequence, int) or isinstance(sequence, bool)
            or sequence < minimum_sequence or not isinstance(prediction, dict)):
        return None
    label, confidence = prediction.get("label"), prediction.get("confidence")
    if (not isinstance(label, str) or not label.strip()
            or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        return None
    if selected_mode == "word" and not word_candidate_allowed(record):
        return None
    if selected_mode == "alphabet" and (
            record.get("signing_hand") != "Right"
            or not isinstance(record.get("signing_hand_confidence"), (int, float))
            or not math.isfinite(record["signing_hand_confidence"])
            or record["signing_hand_confidence"] < 0.85):
        return None
    return PendingCandidate(selected_mode, label, float(confidence), sequence)


def update_pending_candidate(current: PendingCandidate | None,
                             incoming: PendingCandidate | None
                             ) -> tuple[PendingCandidate | None, bool]:
    """Keep pending input through neutral frames; ignore delayed older results."""
    if incoming is None:
        return current, False
    if current is not None and incoming.sequence < current.sequence:
        return current, False
    changed = (current is None or current.mode != incoming.mode
               or current.label != incoming.label)
    return incoming, changed


def candidate_outside_cooldown(candidate: PendingCandidate | None, *,
                               controller: ConfirmationController, now: float,
                               blocked_through_sequence: int) -> PendingCandidate | None:
    """Drop candidates produced during cooldown, including delayed IPC results."""
    if (candidate is None or controller.in_cooldown(now)
            or candidate.sequence <= blocked_through_sequence):
        return None
    return candidate


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
    if selected_mode == "word" and not word_candidate_allowed(record):
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
    candidate: PendingCandidate | None = None,
    candidate_status: str = "Pending",
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
        ("Candidate", (f"{candidate.mode.upper()} {candidate.label.upper()} "
                       f"{candidate.confidence:.1%} — {candidate_status}"
                       if candidate else "(none)")),
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
    if not (committed is None or
            (isinstance(committed, str)
             and (known_labels is None or committed in known_labels))):
        return False
    if "controller_handedness" in record and type(record["controller_handedness"]) is not bool:
        return False
    if "controller_gesture" in record and record["controller_gesture"] not in (
            None, "confirm", "open_palm", "neutral"):
        return False
    if "controller_sequence" in record and (
            not isinstance(record["controller_sequence"], int)
            or isinstance(record["controller_sequence"], bool)
            or record["controller_sequence"] != record["sequence"]):
        return False
    if "signing_hand" in record and record["signing_hand"] not in (None, "Left", "Right"):
        return False
    if "signing_hand_verified" in record and type(record["signing_hand_verified"]) is not bool:
        return False
    if "signing_hand_confidence" in record and (
            not isinstance(record["signing_hand_confidence"], (int, float))
            or not math.isfinite(record["signing_hand_confidence"])
            or not 0 <= record["signing_hand_confidence"] <= 1):
        return False
    return True


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
