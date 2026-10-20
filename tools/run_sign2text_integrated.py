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

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sign2text.integrated import (
    TextAccumulator, apply_text_key, apply_worker_commit, encode_frame, enqueue_latest,
    presentation_fields, switch_input_mode, valid_worker_result,
)
from sign2text.recognizers.alphabet.adapter import DEFAULT_REPOSITORY as ALPHABET_REPOSITORY
from sign2text.recognizers.words.adapter import DEFAULT_REPOSITORY as WORD_REPOSITORY


WINDOW = "Sign2Text — WORD / SPELL"
ENVIRONMENTS = {"word": PROJECT_ROOT / ".venv-repair",
                "alphabet": PROJECT_ROOT / ".venv-alphabet"}


class InferenceWorker:
    """Own one subprocess interpreter; never opens or reads a camera."""

    def __init__(self, mode: str, repository: Path):
        env_python = ENVIRONMENTS[mode] / "bin" / "python"
        if not env_python.is_file():
            raise FileNotFoundError(f"Required {mode} interpreter is missing: {env_python}")
        self.mode = mode
        self.process = subprocess.Popen(
            [str(env_python), "-u", str(PROJECT_ROOT / "tools" / "sign2text_inference_worker.py"),
             "--mode", mode, "--repository", str(repository)],
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


def run(camera_index: int, initial_mode: str) -> int:
    capture = cv2.VideoCapture(camera_index)
    workers: dict[str, InferenceWorker] = {}
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open webcam device {camera_index}.")
        actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print(f"Camera {camera_index}: using backend default; reported "
              f"{actual[0]}x{actual[1]}", flush=True)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        fullscreen = False
        if actual[0] > 0 and actual[1] > 0:
            scale = min(960 / actual[0], 540 / actual[1])
            cv2.resizeWindow(WINDOW, max(320, int(actual[0] * scale)),
                             max(240, int(actual[1] * scale)))

        text = TextAccumulator()
        mode = initial_mode
        workers["word"] = InferenceWorker("word", WORD_REPOSITORY)
        workers["alphabet"] = InferenceWorker("alphabet", ALPHABET_REPOSITORY)
        latest_by_mode = {"word": None, "alphabet": None}
        sequence = 0
        mode_activation_sequence = 0
        first_frame = True
        while True:
            ok, raw_frame = capture.read()
            if not ok or raw_frame is None:
                raise RuntimeError(f"Camera {camera_index} stopped returning frames.")
            sequence += 1
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
                    apply_worker_commit(
                        text, record, selected_mode=mode,
                        minimum_sequence=mode_activation_sequence,
                    )

            display = frame.copy()
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
            )
            field_y = {"Input mode": 35, "Word": 78, "Letter": 112, "Status": 150}
            field_colors = {"Input mode": (0, 220, 160), "Word": (255, 240, 200),
                            "Letter": (200, 255, 220), "Status": (245, 245, 245)}
            for name, value in fields[:-1]:
                _draw_text(display, f"{name}: {value}", field_y[name],
                           scale=0.66 if name == "Input mode" else 0.58,
                           color=field_colors[name], thickness=2 if name == "Input mode" else 1)
            _draw_text(display, f"Text: {fields[-1][1]}", display.shape[0] - 52,
                       scale=0.82, thickness=2, color=(255, 255, 255))
            _draw_text(display,
                       "1 WORD | 2 SPELL | Space separator | Backspace undo | C clear | Q quit",
                       display.shape[0] - 17, scale=0.44)
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
            if key in (ord("1"), ord("2")):
                selected = "word" if key == ord("1") else "alphabet"
                if selected != mode:
                    mode = switch_input_mode(text, mode, selected)
                    mode_activation_sequence = sequence + 1
            else:
                apply_text_key(text, raw_key)
    finally:
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
    args = parser.parse_args(argv)
    try:
        mode = "alphabet" if args.mode in ("spell", "alphabet") else "word"
        return run(args.camera, mode)
    except Exception as exc:
        print(f"Sign2Text could not continue: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
