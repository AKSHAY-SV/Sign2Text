#!/usr/bin/env python3
"""Private stdin/stdout inference worker for the integrated UI (no camera access)."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sign2text.integrated import read_frame
from sign2text.recognizers.words.landmarker import create_integrated_word_landmarker

_PROTOCOL_OUT = sys.stdout


def _emit(record: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(record, separators=(",", ":")) + "\n")
    _PROTOCOL_OUT.flush()


def run_word(repository: Path) -> None:
    from sign2text.recognizers.words import (
        StableWordText, WordRecognizer, WordReleaseGate, WordSequenceBuffer,
        has_detected_hand,
    )
    recognizer = WordRecognizer(repository)
    sequence = WordSequenceBuffer(recognizer.sequence_length, recognizer.feature_size)
    stable = StableWordText(recognizer.class_names, stable_updates=8, confidence_threshold=0.60)
    release_gate = WordReleaseGate()
    landmarker = None
    accepting = False
    try:
        landmarker = create_integrated_word_landmarker(
            recognizer.source, mediapipe_version=mp.__version__
        )
        _emit({"type": "ready", "mode": "word",
               "classes": list(recognizer.class_names)})
        for message in iter_frames():
            if message.accept_input and not accepting:
                # Do not let background-mode frames seed an accepted word window.
                sequence.clear()
                stable.update(None, 0.0)
            accepting = message.accept_input
            rgb = cv2.cvtColor(message.frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not has_detected_hand(result):
                sequence.clear()
                stable.clear()  # Re-arm the same word after hand release.
                release_gate.reset()
                _emit({"type": "result", "mode": "word", "sequence": message.sequence,
                       "status": "no hand detected", "prediction": None,
                       "top": [], "progress": stable.progress, "committed": None,
                       "input_enabled": message.accept_input})
                continue
            sequence.append(recognizer.extract_frame(result))
            if not sequence.ready:
                stable.update(None, 0.0)
                _emit({"type": "result", "mode": "word", "sequence": message.sequence,
                       "status": f"collecting {sequence.count}/{recognizer.sequence_length}",
                       "prediction": None, "top": [], "progress": stable.progress,
                       "committed": None, "input_enabled": message.accept_input})
                continue
            prediction = recognizer.predict_sequence(sequence.array())
            best = prediction.best if prediction.status == "ok" else None
            status = prediction.status
            if message.accept_input and best:
                stable_commit = stable.update(best.label, best.confidence)
                committed = stable_commit and release_gate.accept(best.label)
                if best.confidence < stable.confidence_threshold:
                    status = "low confidence"
                elif committed:
                    status = "accepted"
                elif stable.progress >= 1.0:
                    status = "release to repeat"
                else:
                    status = "stabilizing"
            else:
                stable.update(None, 0.0)
                committed = False
                if best:
                    status = ("low confidence" if best.confidence < stable.confidence_threshold
                              else "prediction")
            _emit({"type": "result", "mode": "word", "sequence": message.sequence,
                   "status": status,
                   "prediction": ({"label": best.label, "confidence": best.confidence}
                                  if best else None),
                   "top": [{"label": item.label, "confidence": item.confidence}
                           for item in prediction.predictions],
                   "progress": stable.progress,
                   "committed": best.label if committed and best else None,
                   "input_enabled": message.accept_input})
    finally:
        if landmarker is not None:
            landmarker.close()
        recognizer.close()


def run_alphabet(repository: Path) -> None:
    from sign2text.recognizers.alphabet import AlphabetRecognizer, StableTextBuffer

    recognizer = AlphabetRecognizer(repository)
    stable = StableTextBuffer(commit_frames=10, confidence_threshold=0.75)
    hands_api = mp.solutions.hands
    accepting = False
    try:
        with hands_api.Hands(
            static_image_mode=False, model_complexity=1, max_num_hands=1,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as hands:
            _emit({"type": "ready", "mode": "alphabet",
                   "classes": list(recognizer.class_names)})
            for message in iter_frames():
                if message.accept_input and not accepting:
                    # Start J/Z motion tracking from the selected SPELL period.
                    recognizer.reset()
                    stable.update(None, 0.0)
                accepting = message.accept_input
                height, width = message.frame.shape[:2]
                rgb = cv2.cvtColor(message.frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                detected = hands.process(rgb)
                landmarks = (detected.multi_hand_landmarks[0]
                             if detected.multi_hand_landmarks else None)
                if landmarks is None:
                    recognizer.reset()
                    stable.update(None, 0.0, hand_present=False)
                    _emit({"type": "result", "mode": "alphabet",
                           "sequence": message.sequence, "status": "no hand detected",
                           "prediction": None, "top": [], "progress": stable.progress,
                           "committed": None, "input_enabled": message.accept_input})
                    continue
                result = recognizer.process_hand(landmarks, width, height, infer=True)
                committed = None
                status = result.status
                if result.updated and result.best and message.accept_input:
                    accepted = stable.update(result.best.label, result.best.confidence,
                                             moving=result.moving)
                    if accepted:
                        committed = result.best.label
                    if result.best.confidence < stable.confidence_threshold:
                        status = "low confidence"
                    elif committed:
                        status = "accepted"
                    elif result.moving:
                        status = "motion"
                    elif stable.progress >= 1.0:
                        status = "release to repeat"
                    else:
                        status = "stabilizing"
                elif not message.accept_input:
                    stable.update(None, 0.0, moving=result.moving)
                elif not result.best:
                    stable.update(None, 0.0, moving=result.moving)
                best = result.best
                _emit({"type": "result", "mode": "alphabet",
                       "sequence": message.sequence, "status": status,
                       "prediction": ({"label": best.label,
                                       "confidence": best.confidence,
                                       "motion_resolved": best.motion_resolved}
                                      if best else None),
                       "top": [{"label": item.label, "confidence": item.confidence}
                               for item in result.predictions],
                       "progress": stable.progress, "committed": committed,
                       "input_enabled": message.accept_input})
    finally:
        recognizer.close()


def iter_frames():
    while True:
        message = read_frame(sys.stdin.buffer)
        if message is None:
            return
        yield message


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("word", "alphabet"), required=True)
    parser.add_argument("--repository", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        # Upstream source modules print model and label diagnostics to stdout.
        # Keep the private JSON-lines protocol clean while preserving those logs
        # visibly on the terminal through stderr.
        with redirect_stdout(sys.stderr):
            if args.mode == "word":
                run_word(args.repository)
            else:
                run_alphabet(args.repository)
        return 0
    except Exception as exc:
        # Keep diagnostics on stderr, protocol status on stdout. No fatal native
        # abort is caught here; this handler is for ordinary Python exceptions.
        print(f"{args.mode} inference worker failed: {exc}", file=sys.stderr, flush=True)
        _emit({"type": "fatal", "mode": args.mode, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
