#!/usr/bin/env python3
"""One-camera UI for selecting the isolated word or alphabet recognizer."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from queue import Queue, Empty

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sign2text.integrated import (
    HAND_IDENTITY_CONFIDENCE, HAND_OVERLAP_RATIO,
    ConfirmationController, TextAccumulator, apply_candidate_action, apply_candidate_key,
    apply_text_key, candidate_outside_cooldown,
    candidate_from_result, encode_frame, enqueue_latest, presentation_fields,
    update_pending_candidate,
    switch_input_mode, valid_worker_result,
)
from sign2text.recognizers.alphabet.adapter import DEFAULT_REPOSITORY as ALPHABET_REPOSITORY
from sign2text.recognizers.words.adapter import DEFAULT_REPOSITORY as WORD_REPOSITORY


WINDOW = "Sign2Text — WORD / SPELL"
ENVIRONMENTS = {"word": PROJECT_ROOT / ".venv-repair",
                "alphabet": PROJECT_ROOT / ".venv-alphabet"}
MIN_DISPLAY_SIZE = (320, 240)
MIN_WINDOW_QUERY_SIZE = (160, 120)
CONTROLLER_HOLD_SECONDS = 0.3
CONTROLLER_MAX_HANDS = 2
CONTROLLER_ENABLED_DEFAULT = True
CONTROLLER_REGION = (0.00, 0.00, 0.34, 1.00)  # normalized x0, y0, x1, y1
SIGNING_REGION = (0.34, 0.00, 1.00, 1.00)


def classify_controller_hands(result, *, controller_region=CONTROLLER_REGION,
                              signing_region=SIGNING_REGION) -> dict:
    """Identify spatially separated hands and classify the left-hand open palm.

    The image passed to MediaPipe is the mirrored preview frame, matching its
    documented handedness assumption.  Therefore its label maps directly to
    the user's anatomical hand.
    """
    landmarks = getattr(result, "multi_hand_landmarks", None) or []
    handedness = getattr(result, "multi_handedness", None) or []
    state = {"gesture": None, "reliable": False, "signing_hand": None,
             "signing_confidence": 0.0, "controller": None,
             "controller_confidence": 0.0, "reason": "missing or ambiguous handedness",
             "hands": []}
    if (not landmarks or len(landmarks) != len(handedness)
            or len(landmarks) > CONTROLLER_MAX_HANDS):
        return state
    left_indices, right_indices = [], []
    for index, classifications in enumerate(handedness):
        if not classifications or len(classifications) != 1:
            return state
        item = classifications[0]
        score = getattr(item, "score", None)
        label = getattr(item, "label", None)
        if not isinstance(score, (int, float)) or score < HAND_IDENTITY_CONFIDENCE:
            return state
        if label not in ("Left", "Right"):
            return state
        center = (sum(point.x for point in landmarks[index].landmark) / 21,
                  sum(point.y for point in landmarks[index].landmark) / 21)
        state["hands"].append((label, score, center))
        if label == "Left":
            left_indices.append(index)
        else:
            right_indices.append(index)
    if len(left_indices) != 1:
        return state
    left_index = left_indices[0]
    left = landmarks[left_index]
    center = state["hands"][left_index][2]
    if not _inside_region(center, controller_region):
        state["reason"] = "left hand outside controller region"
        return state
    if len(right_indices) == 1:
        right_index = right_indices[0]
        right_center = state["hands"][right_index][2]
        if (not _inside_region(right_center, signing_region)
                or _hands_overlap(left, landmarks[right_index])):
            state["reason"] = "hands overlap or right hand outside signing region"
            return state
        state["signing_hand"] = "Right"
        state["signing_confidence"] = state["hands"][right_index][1]
    state["controller"] = "Left"
    state["controller_confidence"] = state["hands"][left_index][1]
    state["gesture"] = _classify_left_hand_gesture(left)
    state["reliable"] = True
    state["reason"] = "ok"
    return state


def controller_action(gesture, *, pending, reliable, now, controller,
                      observation_sequence=None):
    """Apply a verified left-hand open-palm event to the current candidate."""
    return controller.update(
        gesture, now, handedness_reliable=reliable, candidate=pending,
        observation_sequence=observation_sequence,
    )


def _inside_region(center, region) -> bool:
    x0, y0, x1, y1 = region
    return x0 <= center[0] <= x1 and y0 <= center[1] <= y1


def _hands_overlap(first, second) -> bool:
    def bounds(hand):
        xs = [point.x for point in hand.landmark]
        ys = [point.y for point in hand.landmark]
        return min(xs), min(ys), max(xs), max(ys)
    a, b = bounds(first), bounds(second)
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(1e-8, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-8, (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / min(area_a, area_b) > HAND_OVERLAP_RATIO


def _classify_left_hand_gesture(hand) -> str | None:
    points = hand.landmark
    if len(points) < 21:
        return None
    fingers_open = [points[tip].y < points[pip].y for tip, pip in
                    ((8, 6), (12, 10), (16, 14), (20, 18))]
    if all(fingers_open):
        return "open_palm"
    return None


def fit_camera_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale a camera frame into a window-sized canvas without distortion."""
    if frame.ndim != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("frame must have positive height and width")
    if width < 1 or height < 1:
        raise ValueError("display dimensions must be positive")

    source_height, source_width = frame.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, min(width, round(source_width * scale)))
    resized_height = max(1, min(height, round(source_height * scale)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA
                         if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, frame.shape[2]), dtype=frame.dtype)
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


def _window_size(window: str, fallback: tuple[int, int]) -> tuple[int, int]:
    """Read a plausible OpenCV client area, otherwise use the camera size.

    OpenCV's Linux backends may transiently report the last image rectangle or
    a tiny placeholder while a window is first mapped or changing fullscreen.
    """
    try:
        _, _, width, height = cv2.getWindowImageRect(window)
        if width >= MIN_WINDOW_QUERY_SIZE[0] and height >= MIN_WINDOW_QUERY_SIZE[1]:
            return width, height
    except Exception:
        pass
    return fallback


def _render_dimensions(frame: np.ndarray, window: str, fallback: tuple[int, int]) -> tuple[int, int]:
    """Resolve safe canvas dimensions, preferring a real window client area."""
    width, height = _window_size(window, fallback)
    if width < 1 or height < 1:
        return frame.shape[1], frame.shape[0]
    return width, height


def _draw_regions(display: np.ndarray, source_shape, controller_region, signing_region) -> None:
    """Draw normalized source-frame zones on the display-only copy."""
    source_h, source_w = source_shape[:2]
    out_h, out_w = display.shape[:2]
    scale = min(out_w / source_w, out_h / source_h)
    view_w, view_h = round(source_w * scale), round(source_h * scale)
    offset_x, offset_y = (out_w - view_w) // 2, (out_h - view_h) // 2
    for region, color, label in ((controller_region, (255, 180, 0), "LEFT CONTROLLER"),
                                 (signing_region, (0, 210, 90), "RIGHT SIGNING")):
        x0, y0, x1, y1 = region
        p0 = (offset_x + round(x0 * view_w), offset_y + round(y0 * view_h))
        p1 = (offset_x + round(x1 * view_w), offset_y + round(y1 * view_h))
        cv2.rectangle(display, p0, p1, color, 2, cv2.LINE_AA)
        cv2.putText(display, label, (p0[0] + 8, p0[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


class InferenceWorker:
    """Own one subprocess interpreter; never opens or reads a camera."""

    def __init__(self, mode: str, repository: Path, *, controller_landmarks: bool = False,
                 controller_region=CONTROLLER_REGION, signing_region=SIGNING_REGION):
        env_python = ENVIRONMENTS[mode] / "bin" / "python"
        if not env_python.is_file():
            raise FileNotFoundError(f"Required {mode} interpreter is missing: {env_python}")
        self.mode = mode
        self.controller_landmarks_enabled = controller_landmarks
        command = [str(env_python), "-u", str(PROJECT_ROOT / "tools" / "sign2text_inference_worker.py"),
                   "--mode", mode, "--repository", str(repository),
                   "--signing-region", *map(str, signing_region)]
        if controller_landmarks:
            command.extend(["--controller-landmarks", "--controller-region",
                            *map(str, controller_region)])
        self.process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, bufsize=0,
        )
        self._frames: queue.Queue = queue.Queue(maxsize=1)
        self._results: deque[dict] = deque(maxlen=32)
        self._lock = threading.Lock()
        self._fatal_lock = threading.Lock()
        self._ready = threading.Event()
        self._fatal: str | None = None
        self._closing = False
        self.known_labels: frozenset[str] = frozenset()
        self.dropped_frames = 0
        self._reader = threading.Thread(target=self._read_results, name=f"{mode}-result-reader", daemon=True)
        self._writer = threading.Thread(target=self._write_frames, name=f"{mode}-frame-writer", daemon=True)
        self._reader.start()
        self._writer.start()

    @property
    def ready(self) -> bool:
        return self._ready.is_set()

    @property
    def fatal(self) -> str | None:
        return self._fatal

    def _read_results(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._report_fatal("Worker emitted invalid protocol output")
                    continue
                if not isinstance(record, dict):
                    self._report_fatal("Worker protocol record was not a JSON object")
                    continue
                if record.get("type") == "ready":
                    classes = record.get("classes")
                    if (record.get("mode") == self.mode and isinstance(classes, list)
                            and classes and all(isinstance(label, str) and label for label in classes)
                            and len(set(classes)) == len(classes)):
                        self.known_labels = frozenset(classes)
                        self._ready.set()
                    else:
                        self._report_fatal("Worker readiness record has invalid mode or class labels")
                elif record.get("type") == "fatal":
                    if record.get("mode") == self.mode:
                        self._report_fatal(str(record.get("error", "worker failed")))
                    else:
                        self._report_fatal("Worker failure mode did not match the request")
                elif record.get("type") == "result":
                    if not valid_worker_result(record, self.mode, self.known_labels):
                        self._report_fatal("Worker emitted an invalid result record")
                        continue
                    with self._lock:
                        if len(self._results) == self._results.maxlen:
                            self._results.clear()
                            self._report_fatal("Worker result backlog exceeded its bound; results discarded")
                        self._results.append(record)
        except (OSError, ValueError) as exc:
            self._report_fatal(f"Worker output error: {exc}")
        finally:
            if self._fatal is None and not self._closing:
                stage = "during initialization" if not self._ready.is_set() else "unexpectedly"
                self._report_fatal(f"{self.mode} worker exited {stage}")

    def _report_fatal(self, message: str) -> None:
        with self._fatal_lock:
            if self._fatal is None and not self._closing:
                self._fatal = message
                print(f"[{self.mode} worker] {message}", file=sys.stderr, flush=True)

    def _write_frames(self) -> None:
        assert self.process.stdin is not None
        while True:
            item = self._frames.get()
            if item is None:
                try:
                    self.process.stdin.close()
                except (OSError, ValueError):
                    pass
                return
            sequence, frame, accept_input = item
            try:
                header, payload = encode_frame(sequence, frame, accept_input=accept_input)
                self.process.stdin.write(header)
                self.process.stdin.write(payload)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError, ValueError) as exc:
                self._report_fatal(f"Worker input error: {exc}")
                return

    def submit(self, sequence: int, frame: np.ndarray, *, accept_input: bool) -> None:
        if not self.ready or self._fatal is not None or self.process.poll() is not None:
            return
        item = (sequence, frame.copy(), accept_input)
        if enqueue_latest(self._frames, item):
            self.dropped_frames += 1

    def drain_results(self) -> list[dict]:
        with self._lock:
            records = list(self._results)
            self._results.clear()
        return records

    def close(self) -> None:
        self._closing = True
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
            try:
                self._frames.put_nowait(None)
            except queue.Full:
                pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._writer.join()
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
        self._reader.join()


def _draw_text(frame: np.ndarray, text: str, y: int, *, scale: float = 0.58,
               color=(245, 245, 245), thickness: int = 1) -> None:
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def run(camera_index: int, initial_mode: str, *, controller_enabled: bool = CONTROLLER_ENABLED_DEFAULT,
        controller_hold_seconds: float = CONTROLLER_HOLD_SECONDS,
        controller_region=CONTROLLER_REGION, signing_region=SIGNING_REGION,
        debug_controller: bool = False, serial_port: str | None = None,
        serial_baud: int = 9600) -> int:
    capture = cv2.VideoCapture(camera_index)
    workers: dict[str, InferenceWorker] = {}
    sender = None
    serial_queue: Queue = Queue(maxsize=1)
    serial_stop = threading.Event()
    serial_thread = None
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open webcam device {camera_index}.")
        actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Camera {camera_index}: using backend default; reported "
              f"{actual[0]}x{actual[1]}", flush=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        fullscreen = False
        source_size = actual if actual[0] > 0 and actual[1] > 0 else (640, 480)
        # Start from the reliable pre-resize behavior: show a camera-sized frame.
        # WINDOW_NORMAL lets the user resize this window and the next iterations
        # adapt once the GUI backend reports a plausible client rectangle.
        initial_size = source_size
        cv2.resizeWindow(WINDOW, *source_size)

        text = TextAccumulator()
        if serial_port:
            def serial_worker():
                connection = None
                try:
                    from tools.lpc2148_serial_sender import SerialTextSender
                    connection = SerialTextSender(serial_port, serial_baud)
                except Exception as exc:
                    print(f"LCD serial output disabled: {exc}", file=sys.stderr, flush=True)
                    return
                last = None
                while not serial_stop.is_set() or not serial_queue.empty():
                    try:
                        value = serial_queue.get(timeout=0.1)
                    except Empty:
                        continue
                    if value != last:
                        try:
                            connection.send(value)
                            last = value
                        except Exception as exc:
                            print(f"LCD serial output disconnected: {exc}", file=sys.stderr, flush=True)
                            break
                if connection is not None:
                    connection.close()
            serial_thread = threading.Thread(target=serial_worker, name="lcd-serial", daemon=True)
            serial_thread.start()
            print(f"Optional LCD serial output requested on {serial_port} at {serial_baud} baud", flush=True)
        last_serial_text = text.text
        pending = None
        candidate_status = "Pending"
        controller = ConfirmationController(controller_hold_seconds)
        cooldown_was_active = False
        cooldown_blocked_through_sequence = -1
        last_confirmed_label = None
        last_confirmed_at = 0.0
        controller_message = "Controller disabled" if not controller_enabled else "Controller ready"
        mode = initial_mode
        workers["word"] = InferenceWorker(
            "word", WORD_REPOSITORY, signing_region=signing_region)
        workers["alphabet"] = InferenceWorker(
            "alphabet", ALPHABET_REPOSITORY, controller_landmarks=controller_enabled,
            controller_region=controller_region, signing_region=signing_region)
        if controller_enabled:
            controller_message = "Left hand: show a clear gesture"
        latest_by_mode = {"word": None, "alphabet": None}
        sequence = 0
        mode_activation_sequence = 0
        first_frame = True
        last_render_dimensions = None
        while True:
            ok, raw_frame = capture.read()
            if not ok or raw_frame is None:
                raise RuntimeError(f"Camera {camera_index} stopped returning frames.")
            sequence += 1
            loop_now = time.monotonic()
            cooling_down = controller.in_cooldown(loop_now)
            if cooldown_was_active and not cooling_down:
                cooldown_blocked_through_sequence = sequence
                cooldown_was_active = False
            frame = cv2.flip(raw_frame, 1)
            if first_frame:
                print(f"Camera frame dimensions: raw={raw_frame.shape[1]}x{raw_frame.shape[0]}, "
                      f"mirrored/worker={frame.shape[1]}x{frame.shape[0]}, "
                      f"display={frame.shape[1]}x{frame.shape[0]}", flush=True)
                first_frame = False
            for worker_mode, worker in workers.items():
                if worker.ready:
                    worker.submit(sequence, frame, accept_input=(mode == worker_mode))
                for record in worker.drain_results():
                    latest_by_mode[worker_mode] = record
                    candidate = candidate_from_result(
                        record, selected_mode=mode,
                        minimum_sequence=mode_activation_sequence,
                    )
                    candidate = candidate_outside_cooldown(
                        candidate, controller=controller, now=loop_now,
                        blocked_through_sequence=cooldown_blocked_through_sequence,
                    )
                    updated_pending, changed = update_pending_candidate(pending, candidate)
                    if updated_pending is not pending:
                        pending = updated_pending
                    if changed:
                        candidate_status = "Pending"
                        controller.reset()

            controller_gesture, handedness_reliable = None, False
            active_alphabet = latest_by_mode.get("alphabet")
            controller_sequence = None
            if controller_enabled and isinstance(active_alphabet, dict):
                controller_sequence = active_alphabet.get("controller_sequence")
                fresh_controller = (isinstance(controller_sequence, int)
                                    and controller_sequence == active_alphabet.get("sequence")
                                    and sequence - controller_sequence <= 3)
                controller_gesture = active_alphabet.get("controller_gesture") if fresh_controller else None
                handedness_reliable = (active_alphabet.get("controller_handedness") is True
                                       and fresh_controller)
                controller_message = ("Controller ready" if handedness_reliable
                                      else "Controller waiting for clear left hand")
            elif controller_enabled:
                controller_message = "Controller waiting for hand detector"
            action = (controller_action(controller_gesture, pending=pending,
                                        reliable=handedness_reliable,
                                        now=loop_now, controller=controller,
                                        observation_sequence=controller_sequence)
                      if controller_enabled else None)
            if action is not None and pending is not None:
                if apply_candidate_action(text, pending, action):
                    candidate_status = "Accepted" if action == "confirm" else "Rejected"
                    if action == "confirm":
                        controller.begin_cooldown(loop_now)
                        last_confirmed_label = pending.label
                        last_confirmed_at = loop_now
                        cooldown_was_active = True
                    pending = None
                    controller_message = (f"Confirmed {last_confirmed_label}"
                                          if action == "confirm" else "Rejected")
            elif controller.in_cooldown(loop_now):
                remaining = max(1, int(controller.cooldown_remaining(loop_now) + 0.999))
                confirmation_note = (
                    f" · Confirmed {last_confirmed_label}"
                    if last_confirmed_label and loop_now - last_confirmed_at < 1.5 else "")
                controller_message = f"Cooldown: {remaining}s{confirmation_note}"
            elif controller.state == "cooldown":
                controller_message = "Waiting for release"
            elif controller.state in ("waiting for release", "holding open palm"):
                controller_message = controller.state.title()
            elif debug_controller:
                reason = ((active_alphabet or {}).get("controller_reason", "waiting"))
                if not handedness_reliable:
                    controller_message = f"Debug: no action ({reason})"
                elif pending is None:
                    controller_message = f"Debug: {controller_gesture}; no pending candidate"
                else:
                    controller_message = f"Debug: {controller_gesture}; pending, holding/release-gated"
                controller_message += f" ({controller.state})"

            display_width, display_height = _render_dimensions(frame, WINDOW, source_size)
            display = fit_camera_frame(frame, display_width, display_height)
            if controller_enabled:
                _draw_regions(display, frame.shape, controller_region, signing_region)
            word_worker = workers.get("word")
            alphabet_worker = workers.get("alphabet")
            worker_status = {
                "word": (f"error: {word_worker.fatal}" if word_worker and word_worker.fatal else
                         "ready" if word_worker and word_worker.ready else "loading"),
                "alphabet": (f"error: {alphabet_worker.fatal}" if alphabet_worker and alphabet_worker.fatal else
                             "ready" if alphabet_worker and alphabet_worker.ready else "loading"),
            }
            fields = presentation_fields(
                mode, latest_by_mode, worker_status, text,
                minimum_sequence=mode_activation_sequence,
                candidate=pending, candidate_status=candidate_status,
            )
            margin = max(12, min(24, display_height // 45))
            line_height = max(28, min(42, display_height // 15))
            field_y = {name: margin + line_height * (index + 1) for index, name in
                       enumerate(("Input mode", "Word", "Letter", "Candidate", "Status"))}
            field_colors = {"Input mode": (0, 220, 160), "Word": (255, 240, 200),
                            "Letter": (200, 255, 220), "Candidate": (0, 220, 160),
                            "Status": (245, 245, 245)}
            for name, value in fields[:-1]:
                _draw_text(display, f"{name}: {value}", field_y[name],
                           scale=max(0.48, min(0.66, display_height / 820)),
                           color=field_colors[name], thickness=2 if name == "Input mode" else 1)
            _draw_text(display, f"Text: {fields[-1][1]}", display.shape[0] - margin * 2 - 24,
                       scale=max(0.52, min(0.82, display_height / 660)),
                       thickness=2, color=(255, 255, 255))
            _draw_text(display,
                       "Left open palm: Confirm | Relaxed hand: Release",
                       display.shape[0] - margin * 3 - 40,
                       scale=max(0.35, min(0.52, display_height / 1050)))
            _draw_text(display,
                       f"{controller_message} | 1 WORD | 2 SPELL | Space | Backspace | C clear | Q quit | F fullscreen",
                       display.shape[0] - margin, scale=max(0.35, min(0.52, display_height / 1050)))
            if debug_controller:
                debug_hands = (active_alphabet or {}).get("controller_hands", [])
                _draw_text(display,
                           f"DEBUG hands={debug_hands} signing=Right controller=Left "
                           f"gesture={controller_gesture} state={controller.state} "
                           f"seq={controller_sequence} pending={bool(pending)} "
                           f"reason={(active_alphabet or {}).get('controller_reason', 'waiting')}",
                           margin + line_height * 6,
                           scale=max(0.36, min(0.48, display_height / 1000)),
                           color=(0, 255, 255))
            if display.ndim != 3 or display.shape[0] < 1 or display.shape[1] < 1:
                display = frame.copy()
                display_width, display_height = frame.shape[1], frame.shape[0]
            dimensions = (frame.shape, (display_width, display_height), display.shape,
                          "fullscreen" if fullscreen else "windowed")
            if dimensions != last_render_dimensions:
                print("Display sizing: "
                      f"source={frame.shape[1]}x{frame.shape[0]} "
                      f"target={display_width}x{display_height} "
                      f"rendered={display.shape[1]}x{display.shape[0]} "
                      f"mode={dimensions[3]}", flush=True)
                last_render_dimensions = dimensions
            cv2.imshow(WINDOW, display)
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                    break
            except cv2.error:
                pass
            raw_key = cv2.waitKey(1)
            if raw_key < 0:
                continue
            key = raw_key & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("f"), ord("F")):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
                )
                continue
            if controller.in_cooldown(time.monotonic()):
                continue
            if key in (ord("1"), ord("2")):
                selected = "word" if key == ord("1") else "alphabet"
                if selected != mode:
                    mode = switch_input_mode(text, mode, selected)
                    mode_activation_sequence = sequence + 1
                    pending = None
                    candidate_status = "Pending"
                    controller.reset()
            else:
                key_action = apply_candidate_key(text, pending, raw_key)
                if key_action is not None:
                    candidate_status = "Accepted" if key_action == "confirm" else "Rejected"
                    pending = None
                    controller.reset()
                else:
                    apply_text_key(text, raw_key)
            if text.text != last_serial_text:
                last_serial_text = text.text
                if serial_port:
                    try:
                        serial_queue.put_nowait(last_serial_text)
                    except queue.Full:
                        try:
                            serial_queue.get_nowait()
                        except queue.Empty:
                            pass
                        serial_queue.put_nowait(last_serial_text)
    finally:
        serial_stop.set()
        if serial_thread is not None:
            serial_thread.join()
        for worker in workers.values():
            worker.close()
        capture.release()
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mode", choices=("word", "spell", "alphabet"), default="word",
                        help="input mode to accept; switch with 1/2 (alphabet is a SPELL alias)")
    parser.add_argument("--no-left-controller", action="store_true",
                        help="disable experimental left-hand confirmation")
    parser.add_argument("--controller-hold", type=float, default=CONTROLLER_HOLD_SECONDS,
                        help="seconds a controller gesture must remain stable")
    parser.add_argument("--controller-region", nargs=4, type=float, default=CONTROLLER_REGION,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="normalized left-controller region in mirrored preview")
    parser.add_argument("--signing-region", nargs=4, type=float, default=SIGNING_REGION,
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="normalized right-signing region in mirrored preview")
    parser.add_argument("--debug-controller", action="store_true",
                        help="show temporary hand and controller diagnostics")
    parser.add_argument("--serial-port", help="optional LPC2148 LCD UART port")
    parser.add_argument("--serial-baud", type=int, default=9600)
    args = parser.parse_args(argv)
    try:
        mode = "alphabet" if args.mode in ("spell", "alphabet") else "word"
        return run(args.camera, mode, controller_enabled=not args.no_left_controller,
                   controller_hold_seconds=args.controller_hold,
                   controller_region=tuple(args.controller_region),
                   signing_region=tuple(args.signing_region),
                   debug_controller=args.debug_controller,
                   serial_port=args.serial_port, serial_baud=args.serial_baud)
    except Exception as exc:
        print(f"Sign2Text could not continue: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
