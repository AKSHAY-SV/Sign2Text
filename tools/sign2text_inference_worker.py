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

from sign2text.integrated import (
    HAND_IDENTITY_CONFIDENCE, HAND_OVERLAP_RATIO, read_frame,
)
from sign2text.recognizers.words.landmarker import create_integrated_word_landmarker

_PROTOCOL_OUT = sys.stdout


def _emit(record: dict) -> None:
    _PROTOCOL_OUT.write(json.dumps(record, separators=(",", ":")) + "\n")
    _PROTOCOL_OUT.flush()


def run_word(repository: Path, *, signing_region=(0.34, 0.0, 1.0, 1.0)) -> None:
    from sign2text.recognizers.words import (
        StableWordText, WordRecognizer, WordReleaseGate, WordSequenceBuffer,
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
            right_hand = getattr(result, "right_hand_landmarks", None)
            center = ((sum(point.x for point in right_hand) / len(right_hand),
                       sum(point.y for point in right_hand) / len(right_hand))
                      if right_hand else None)
            x0, y0, x1, y1 = signing_region
            right_in_region = (center is not None and x0 <= center[0] <= x1
                               and y0 <= center[1] <= y1)
            if not right_in_region:
                sequence.clear()
                stable.clear()  # Re-arm the same word after hand release.
                release_gate.reset()
                _emit({"type": "result", "mode": "word", "sequence": message.sequence,
                       "status": ("right hand outside signing region" if right_hand
                                  else "right signing hand not detected"), "prediction": None,
                       "top": [], "progress": stable.progress, "committed": None,
                       "input_enabled": message.accept_input,
                       "signing_hand": "Right" if right_hand else None,
                       "signing_hand_verified": False})
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
                   "signing_hand": "Right", "signing_hand_verified": True,
                   "input_enabled": message.accept_input})
    finally:
        if landmarker is not None:
            landmarker.close()
        recognizer.close()


def run_alphabet(repository: Path, *, controller_landmarks: bool = False,
                 controller_region=(0.0, 0.0, 0.34, 1.0),
                 signing_region=(0.34, 0.0, 1.0, 1.0)) -> None:
    from sign2text.recognizers.alphabet import AlphabetRecognizer, StableTextBuffer

    recognizer = AlphabetRecognizer(repository)
    stable = StableTextBuffer(commit_frames=10, confidence_threshold=0.75)
    hands_api = mp.solutions.hands
    accepting = False
    try:
        with hands_api.Hands(
            static_image_mode=False, model_complexity=1,
            max_num_hands=2 if controller_landmarks else 1,
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
                hands_found = detected.multi_hand_landmarks or []
                groups = getattr(detected, "multi_handedness", None) or []
                parsed = []
                if len(groups) == len(hands_found):
                    for group in groups:
                        if group.classification:
                            item = group.classification[0]
                            parsed.append((item.label, item.score))
                if len(parsed) != len(hands_found):
                    parsed = []
                def hand_center(hand):
                    return (sum(point.x for point in hand.landmark) / len(hand.landmark),
                            sum(point.y for point in hand.landmark) / len(hand.landmark))
                def in_region(center, region):
                    x0, y0, x1, y1 = region
                    return x0 <= center[0] <= x1 and y0 <= center[1] <= y1
                right = [i for i, (label, score) in enumerate(parsed)
                         if label == "Right" and score >= HAND_IDENTITY_CONFIDENCE
                         and in_region(hand_center(hands_found[i]), signing_region)]
                landmarks = hands_found[right[0]] if len(right) == 1 else None
                signing_hand = "Right" if landmarks is not None else None
                signing_hand_confidence = parsed[right[0]][1] if landmarks is not None else 0.0
                controller_gesture, controller_reliable = None, False
                controller_reason = "handedness missing or ambiguous"
                controller_hands = []
                if controller_landmarks:
                    if len(groups) != len(hands_found) or len(groups) not in (1, 2):
                        parsed = []
                    if len(parsed) != len(hands_found):
                        parsed = []
                    if parsed:
                        controller_hands = [
                            {"label": label, "score": float(score),
                             "center": hand_center(hands_found[i])}
                            for i, (label, score) in enumerate(parsed)
                        ]
                    if (parsed and all(label in ("Left", "Right") and score >= HAND_IDENTITY_CONFIDENCE
                                       for label, score in parsed)):
                        left = [i for i, (label, _) in enumerate(parsed) if label == "Left"]
                        overlap = False
                        if len(hands_found) == 2:
                            def bounds(hand):
                                xs = [point.x for point in hand.landmark]
                                ys = [point.y for point in hand.landmark]
                                return min(xs), min(ys), max(xs), max(ys)
                            a, b = bounds(hands_found[0]), bounds(hands_found[1])
                            area = max(1e-8, min((a[2]-a[0])*(a[3]-a[1]),
                                                (b[2]-b[0])*(b[3]-b[1])))
                            common = max(0.0, min(a[2], b[2])-max(a[0], b[0])) * max(
                                0.0, min(a[3], b[3])-max(a[1], b[1]))
                            overlap = common / area > HAND_OVERLAP_RATIO
                        left_in_region = [i for i in left
                                          if in_region(hand_center(hands_found[i]), controller_region)]
                        if len(left_in_region) == 1 and not overlap:
                            controller_reliable = True
                            hand = hands_found[left_in_region[0]]
                            p = hand.landmark
                            if len(p) >= 21:
                                opened = [p[t].y < p[j].y for t, j in
                                          ((8, 6), (12, 10), (16, 14), (20, 18))]
                                if all(opened):
                                    controller_gesture = "open_palm"
                                else:
                                    controller_gesture = "neutral"
                            controller_reason = "ok"
                        elif overlap:
                            controller_reason = "hands overlap"
                        elif len(left_in_region) != 1:
                            controller_reason = "ambiguous left-hand identity"
                    else:
                        if parsed:
                            controller_reason = "low handedness confidence"
                else:
                    controller_hands = []
                if landmarks is None:
                    recognizer.reset()
                    stable.update(None, 0.0, hand_present=False)
                    _emit({"type": "result", "mode": "alphabet",
                           "sequence": message.sequence, "status": "no hand detected",
                           "prediction": None, "top": [], "progress": stable.progress,
                           "committed": None, "input_enabled": message.accept_input,
                           "controller_gesture": controller_gesture,
                           "controller_handedness": controller_reliable,
                           "controller_sequence": message.sequence,
                           "controller_reason": controller_reason,
                           "controller_hands": controller_hands,
                           "signing_hand": signing_hand,
                           "signing_hand_confidence": signing_hand_confidence})
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
                       "controller_gesture": controller_gesture,
                       "controller_handedness": controller_reliable,
                       "controller_sequence": message.sequence,
                       "controller_reason": controller_reason,
                       "controller_hands": controller_hands,
                       "signing_hand": signing_hand,
                       "signing_hand_confidence": signing_hand_confidence,
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
    parser.add_argument("--controller-landmarks", action="store_true")
    parser.add_argument("--controller-region", nargs=4, type=float,
                        default=(0.0, 0.0, 0.34, 1.0))
    parser.add_argument("--signing-region", nargs=4, type=float,
                        default=(0.34, 0.0, 1.0, 1.0))
    args = parser.parse_args(argv)
    try:
        # Upstream source modules print model and label diagnostics to stdout.
        # Keep the private JSON-lines protocol clean while preserving those logs
        # visibly on the terminal through stderr.
        with redirect_stdout(sys.stderr):
            if args.mode == "word":
                run_word(args.repository, signing_region=args.signing_region)
            else:
                run_alphabet(args.repository, controller_landmarks=args.controller_landmarks,
                             controller_region=args.controller_region,
                             signing_region=args.signing_region)
        return 0
    except Exception as exc:
        # Keep diagnostics on stderr, protocol status on stdout. No fatal native
        # abort is caught here; this handler is for ordinary Python exceptions.
        print(f"{args.mode} inference worker failed: {exc}", file=sys.stderr, flush=True)
        _emit({"type": "fatal", "mode": args.mode, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
