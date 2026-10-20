#!/usr/bin/env python3
"""Run the pretrained isolated ASL word recognizer in its own webcam window."""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sign2text.recognizers.words import (
    StableWordText,
    WordRecognizer,
    WordSequenceBuffer,
    has_detected_hand,
)
from sign2text.recognizers.words.adapter import DEFAULT_REPOSITORY
from sign2text.recognizers.words.mediapipe_compat import install_holistic_empty_packet_guard


WINDOW = "ASL Isolated Word Recognition"
PREDICTION_INTERVAL = 1
LOW_CONFIDENCE_THRESHOLD = 0.60


def create_local_landmarker(source, *, install_compatibility_guard=True):
    """Create the source model's IMAGE-mode Holistic task without any downloading."""
    if not source.HOLISTIC_TASK_PATH.is_file():
        raise FileNotFoundError(
            f"MediaPipe task model missing: {source.HOLISTIC_TASK_PATH}. "
            "No download was attempted."
        )
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # MediaPipe 0.10.21 can emit empty optional landmark packets. Its default
    # result builder passes those packets to a native proto getter, which aborts
    # instead of returning an empty detection result. Normalize those packets to
    # typed empty landmark protos before the public detect() path converts them.
    if install_compatibility_guard:
        try:
            install_holistic_empty_packet_guard()
        except Exception as exc:
            raise RuntimeError(
                f"Could not install the MediaPipe {mp.__version__} empty-packet guard: {exc}"
            ) from exc

    options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(source.HOLISTIC_TASK_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_face_suppression_threshold=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_suppression_threshold=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def draw_pose_and_hands(frame, results, source):
    """Draw body and hand landmarks only; selected face points remain model inputs."""
    source.draw_skeleton(frame, results.pose_landmarks, source.POSE_CONNECTIONS,
                         (0, 255, 255), (0, 180, 255), point_radius=2)
    source.draw_skeleton(frame, results.left_hand_landmarks, source.HAND_CONNECTIONS,
                         (0, 255, 0), (0, 180, 0), point_radius=3)
    source.draw_skeleton(frame, results.right_hand_landmarks, source.HAND_CONNECTIONS,
                         (255, 80, 80), (255, 80, 80), point_radius=3)


def run(camera, repository, prediction_interval, stable_updates, confidence_threshold,
        show_landmarks=False):
    recognizer = WordRecognizer(repository)
    sequence = WordSequenceBuffer(recognizer.sequence_length, recognizer.feature_size)
    text = StableWordText(recognizer.class_names, stable_updates, confidence_threshold)
    result = None
    frame_index = 0
    last_inference_frame = -prediction_interval
    task_landmarker = None
    capture = None
    window_shown = False
    last_prediction_at = None
    update_gaps = []
    try:
        task_landmarker = create_local_landmarker(recognizer.source)
        capture = cv2.VideoCapture(camera)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open webcam {camera}.")
        cv2.namedWindow(WINDOW, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        while True:
            if window_shown:
                try:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    pass
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam.")
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = task_landmarker.detect(image)
            if show_landmarks:
                draw_pose_and_hands(frame, results, recognizer.source)
            frame_index += 1

            has_hand = has_detected_hand(results)
            if not has_hand:
                sequence.clear()
                text.update(None, 0.0)
                result = None
                status = "No hand detected"
            else:
                features = recognizer.extract_frame(results)
                sequence.append(features)
                if not sequence.ready:
                    result = None
                    status = f"Collecting frames {sequence.count}/{recognizer.sequence_length}"
                elif frame_index - last_inference_frame >= prediction_interval:
                    result = recognizer.predict_sequence(sequence.array())
                    last_inference_frame = frame_index
                    if result.status == "ok" and result.best:
                        text.update(result.best.label, result.best.confidence)
                        now = time.perf_counter()
                        if last_prediction_at is not None:
                            update_gaps.append(now - last_prediction_at)
                        last_prediction_at = now
                        status = "Low confidence" if result.best.confidence < confidence_threshold else "Prediction"
                    else:
                        text.update(None, 0.0)
                        status = result.status
                else:
                    if result is None:
                        status = "Prediction pending"
                    elif result.status != "ok":
                        status = result.status
                    elif result.best and result.best.confidence < confidence_threshold:
                        status = "Low confidence"
                    else:
                        status = "Prediction"

            cv2.putText(frame, "ISOLATED ASL WORD", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 230, 160), 2, cv2.LINE_AA)
            cv2.putText(frame, status, (16, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.58, (255, 255, 255), 1, cv2.LINE_AA)
            if result and result.status == "ok" and result.best:
                color = (0, 200, 100) if result.best.confidence >= confidence_threshold else (0, 165, 255)
                cv2.putText(frame, f"Live: {result.best.label.upper()}  {result.best.confidence:.1%}",
                            (16, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
                for rank, prediction in enumerate(result.predictions, start=1):
                    cv2.putText(frame,
                                f"{rank}. {prediction.label.upper()}  {prediction.confidence:.1%}",
                                (16, 132 + (rank - 1) * 24), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Stable: {text.progress:.0%}", (16, 218),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Recognized: {text.text or '(none)'}", (16, frame.shape[0] - 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Q quit | C clear", (16, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)
            window_shown = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                text.clear()

    finally:
        if capture is not None:
            capture.release()
        if task_landmarker is not None:
            task_landmarker.close()
        recognizer.close()
        cv2.destroyAllWindows()
        if update_gaps:
            print(f"Prediction update interval: {np.mean(update_gaps):.2f}s average")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--prediction-interval", type=int, default=PREDICTION_INTERVAL,
                        help="refresh predictions every N camera frames (default: 5)")
    parser.add_argument("--stable-updates", type=int, default=8,
                        help="matching confident prediction updates before display commit")
    parser.add_argument("--confidence-threshold", type=float, default=LOW_CONFIDENCE_THRESHOLD)
    parser.add_argument("--show-landmarks", action="store_true",
                        help="draw pose and hand landmarks for diagnostics")
    args = parser.parse_args(argv)
    if args.prediction_interval < 1:
        parser.error("--prediction-interval must be at least 1")
    if args.stable_updates < 1:
        parser.error("--stable-updates must be at least 1")
    if not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be between 0 and 1")
    return run(args.camera, args.repository, args.prediction_interval,
               args.stable_updates, args.confidence_threshold, args.show_landmarks)


if __name__ == "__main__":
    raise SystemExit(main())
