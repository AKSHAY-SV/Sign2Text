#!/usr/bin/env python3
"""Run the pretrained ASL isolated-sign recognizer with a Holistic webcam feed."""

import argparse
import json
import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_model200.keras")
LABEL_PATH = os.path.join(PROJECT_ROOT, "models", "index_to_gloss_200.json")
SEQUENCE_LENGTH = 150
LANDMARKS_PER_FRAME = 86
FEATURES_PER_LANDMARK = 2
FEATURES_PER_FRAME = LANDMARKS_PER_FRAME * FEATURES_PER_LANDMARK
LOW_CONFIDENCE_THRESHOLD = 0.50  # Display guidance only; it does not reject classes.

FACE_LANDMARK_INDICES = (
    33, 133, 159, 145, 153, 144,
    362, 263, 386, 374, 380, 373,
    70, 63, 105, 66, 107,
    295, 282, 320, 285, 318,
    1, 168, 197, 4,
    78, 308, 13, 14, 81, 311,
)
POSE_LANDMARK_INDICES = tuple(
    i for i in range(33) if i not in set(range(0, 11)) | set(range(23, 33))
)


def load_resources():
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    with open(LABEL_PATH, encoding="utf-8") as labels_file:
        glosses = json.load(labels_file)
    expected_shape = (None, SEQUENCE_LENGTH, FEATURES_PER_FRAME)
    if tuple(model.input_shape) != expected_shape:
        raise ValueError(f"Expected model input {expected_shape}, got {model.input_shape}")
    if len(model.output_shape) != 2 or model.output_shape[-1] != 271:
        raise ValueError(f"Expected 271 model outputs, got {model.output_shape}")
    return model, glosses


def extract_landmarks(results):
    """Extract pose, left hand, right hand, and selected face x/y in training order."""
    if results.pose_landmarks:
        pose = [(lm.x, lm.y) for i, lm in enumerate(results.pose_landmarks.landmark)
                if i in POSE_LANDMARK_INDICES]
    else:
        pose = [(0.0, 0.0)] * 12

    if results.left_hand_landmarks:
        left_hand = [(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark]
    else:
        left_hand = [(0.0, 0.0)] * 21

    if results.right_hand_landmarks:
        right_hand = [(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark]
    else:
        right_hand = [(0.0, 0.0)] * 21

    if results.face_landmarks:
        face_all = results.face_landmarks.landmark
        face = [(face_all[i].x, face_all[i].y) for i in FACE_LANDMARK_INDICES]
    else:
        face = [(0.0, 0.0)] * len(FACE_LANDMARK_INDICES)

    features = np.asarray(pose + left_hand + right_hand + face, dtype=np.float32)
    if features.shape != (LANDMARKS_PER_FRAME, FEATURES_PER_LANDMARK):
        raise ValueError(f"Expected (86, 2) landmarks, got {features.shape}")
    return features


def normalize_frame(features):
    """Apply preprocessing_split.py's training-time normalization to one frame."""
    frame = np.asarray(features, dtype=np.float32).copy()
    if frame.shape != (LANDMARKS_PER_FRAME, FEATURES_PER_LANDMARK):
        raise ValueError(f"Expected (86, 2) features, got {frame.shape}")

    def distance(a, b):
        delta = a - b
        return float(np.sqrt(delta[0] ** 2 + delta[1] ** 2))

    def anchor_norm(target, scale, reference):
        return (target - reference) / (scale + 0.01)

    def hand_normalize(hand):
        xmin, xmax = np.min(hand[:, 0]), np.max(hand[:, 0])
        ymin, ymax = np.min(hand[:, 1]), np.max(hand[:, 1])
        width, height = xmax - xmin, ymax - ymin
        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        hand[:, 0] = (hand[:, 0] - center_x) / width if width else 0
        hand[:, 1] = (hand[:, 1] - center_y) / height if height else 0

    neck = np.abs((frame[0] + frame[1]) / 2)
    frame[:] = anchor_norm(frame, distance(frame[0], frame[1]), neck)
    frame[54:] = anchor_norm(frame[54:], distance(frame[0], frame[1]), frame[79])
    frame[[2, 4, 6, 8, 10]] = anchor_norm(
        frame[[2, 4, 6, 8, 10]], distance(frame[0], frame[2]), frame[0]
    )
    frame[[3, 5, 7, 9, 11]] = anchor_norm(
        frame[[3, 5, 7, 9, 11]], distance(frame[1], frame[3]), frame[1]
    )
    hand_normalize(frame[12:33])
    hand_normalize(frame[33:54])
    return frame


def build_model_input(frames):
    """Flatten an exact chronological 150-frame sequence to the model contract."""
    sequence = np.asarray(frames, dtype=np.float32)
    expected = (SEQUENCE_LENGTH, LANDMARKS_PER_FRAME, FEATURES_PER_LANDMARK)
    if sequence.shape != expected:
        raise ValueError(f"Expected sequence {expected}, got {sequence.shape}")
    return np.ascontiguousarray(sequence.reshape(1, SEQUENCE_LENGTH, FEATURES_PER_FRAME))


def mapped_predictions(probabilities, glosses, count=3):
    """Return the highest-probability mapped labels, without relabeling unknowns."""
    scores = np.asarray(probabilities)
    candidates = [i for i in range(scores.shape[-1]) if str(i) in glosses]
    candidates.sort(key=lambda i: float(scores[i]), reverse=True)
    return [(i, glosses[str(i)], float(scores[i])) for i in candidates[:count]]


def prediction_description(probabilities, glosses):
    scores = np.asarray(probabilities)
    index = int(np.argmax(scores))
    confidence = float(scores[index])
    label = glosses.get(str(index))
    status = "mapped" if label is not None else f"unmapped output {index}"
    return label, index, confidence, status, mapped_predictions(scores, glosses)


def is_prediction_frame(frame_number, prediction_interval):
    """Predict on a full 150-frame buffer, then refresh at the requested cadence."""
    return (frame_number >= SEQUENCE_LENGTH and
            (frame_number - SEQUENCE_LENGTH) % prediction_interval == 0)


def draw_visible_landmarks(frame, results, holistic, drawing):
    """Draw body pose and hands only; face landmarks remain inference features."""
    drawing.draw_landmarks(frame, results.pose_landmarks, holistic.POSE_CONNECTIONS)
    drawing.draw_landmarks(frame, results.left_hand_landmarks, holistic.HAND_CONNECTIONS)
    drawing.draw_landmarks(frame, results.right_hand_landmarks, holistic.HAND_CONNECTIONS)


def inference_worker(model, jobs, results, stop_event, busy_event):
    """Run serial model inference away from the camera/MediaPipe thread."""
    while not stop_event.is_set() or not jobs.empty():
        try:
            frame_number, model_input = jobs.get(timeout=0.1)
        except queue.Empty:
            continue
        started = time.perf_counter()
        try:
            probabilities = model.predict(model_input, verbose=0)[0]
            result = (frame_number, probabilities, time.perf_counter() - started, None)
        except Exception as exc:  # keep the camera preview alive if inference fails
            result = (frame_number, None, time.perf_counter() - started, exc)
        finally:
            busy_event.clear()
        # Keep only the newest completed output if the UI was briefly busy.
        try:
            results.put_nowait(result)
        except queue.Full:
            try:
                results.get_nowait()
            except queue.Empty:
                pass
            results.put_nowait(result)
        jobs.task_done()


def collect_timing(timings, name, elapsed):
    timings[name].append(elapsed * 1000.0)


def report_timings(timings, prediction_updates, update_intervals):
    print("\nTiming summary (milliseconds; mean / p95):")
    for name, values in timings.items():
        if values:
            print(f"  {name}: {np.mean(values):.1f} / {np.percentile(values, 95):.1f} ({len(values)} samples)")
    if update_intervals:
        intervals_ms = np.asarray(update_intervals) * 1000.0
        print("  time between prediction updates: "
              f"{np.mean(intervals_ms):.1f} / {np.percentile(intervals_ms, 95):.1f} "
              f"({prediction_updates} updates)")
    else:
        print("  time between prediction updates: no prediction updates")


def main(argv=None):
    import cv2
    import mediapipe as mp
    import tensorflow as tf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-interval", type=int, default=5,
        help="run inference every N frames after the 150-frame buffer fills (default: 5)",
    )
    parser.add_argument("--camera", type=int, default=0, help="webcam device index")
    args = parser.parse_args(argv)
    if args.prediction_interval < 1:
        parser.error("--prediction-interval must be at least 1")

    print("Loading model and labels...")
    model, glosses = load_resources()
    print(f"Model input={model.input_shape}; outputs={model.output_shape[-1]}; "
          f"mapped labels={len(glosses)}; TensorFlow GPU devices={tf.config.list_physical_devices('GPU')}")
    print(f"Updating predictions every {args.prediction_interval} frames after buffer fills.")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}. Check camera access and try again.")

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    sequence = deque(maxlen=SEQUENCE_LENGTH)
    inference_jobs = queue.Queue(maxsize=1)
    inference_results = queue.Queue(maxsize=1)
    inference_stop = threading.Event()
    inference_busy = threading.Event()
    worker = threading.Thread(
        target=inference_worker,
        args=(model, inference_jobs, inference_results, inference_stop, inference_busy),
        name="sign-model-inference",
        daemon=True,
    )
    label = None
    confidence = 0.0
    prediction_status = "insufficient frames"
    top_predictions = []
    prediction_updates = 0
    last_prediction_time = None
    update_intervals = []
    timing_names = (
        "camera capture", "Holistic", "extraction + normalization", "buffer update",
        "model inference", "drawing", "render + key handling", "overall frame loop",
    )
    timings = {name: [] for name in timing_names}
    frame_number = 0
    worker.start()

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=0,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.9,
        ) as holistic:
            while True:
                loop_start = time.perf_counter()
                started = time.perf_counter()
                ok, frame = cap.read()
                collect_timing(timings, "camera capture", time.perf_counter() - started)
                if not ok:
                    print("Could not read a frame from the webcam.", file=sys.stderr)
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                started = time.perf_counter()
                results = holistic.process(rgb)
                collect_timing(timings, "Holistic", time.perf_counter() - started)

                started = time.perf_counter()
                features = normalize_frame(extract_landmarks(results))
                collect_timing(timings, "extraction + normalization", time.perf_counter() - started)

                started = time.perf_counter()
                sequence.append(features)
                frame_number += 1
                collect_timing(timings, "buffer update", time.perf_counter() - started)

                try:
                    result_frame, probabilities, inference_seconds, inference_error = (
                        inference_results.get_nowait()
                    )
                except queue.Empty:
                    pass
                else:
                    collect_timing(timings, "model inference", inference_seconds)
                    if inference_error is not None:
                        prediction_status = "inference error"
                        print(f"Inference failed: {inference_error}", file=sys.stderr)
                    else:
                        label, index, confidence, mapping_status, top_predictions = prediction_description(
                            probabilities, glosses
                        )
                        prediction_status = mapping_status
                        if confidence < LOW_CONFIDENCE_THRESHOLD and label is not None:
                            prediction_status = "low confidence"
                        now = time.perf_counter()
                        if last_prediction_time is not None:
                            update_intervals.append(now - last_prediction_time)
                        last_prediction_time = now
                        prediction_updates += 1

                has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                if len(sequence) < SEQUENCE_LENGTH:
                    prediction_status = "insufficient frames"
                elif is_prediction_frame(frame_number, args.prediction_interval):
                    if not inference_busy.is_set():
                        model_input = build_model_input(sequence)
                        inference_busy.set()
                        try:
                            inference_jobs.put_nowait((frame_number, model_input))
                            if label is None:
                                prediction_status = "prediction pending"
                        except queue.Full:
                            inference_busy.clear()

                started = time.perf_counter()
                # Face landmarks stay in `features` for inference, but are not drawn.
                draw_visible_landmarks(frame, results, mp_holistic, mp_drawing)

                if prediction_status == "insufficient frames":
                    main_text = f"Waiting for frames: {len(sequence)}/{SEQUENCE_LENGTH}"
                elif prediction_status.startswith("unmapped"):
                    main_text = f"Unknown label ({prediction_status})"
                elif prediction_status == "prediction pending":
                    main_text = "Prediction pending"
                elif prediction_status == "inference error":
                    main_text = "Inference error"
                elif label is None:
                    main_text = "No mapped prediction"
                else:
                    main_text = f"Sign: {label} ({prediction_status})"
                cv2.putText(frame, main_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (16, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                hand_status = "Hand detected" if has_hand else "No hand detected"
                cv2.putText(frame, hand_status, (16, 88), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                for rank, (idx, name, score) in enumerate(top_predictions, start=1):
                    cv2.putText(frame, f"{rank}. {name}: {score:.1%}", (16, 116 + (rank - 1) * 23),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, "Q: quit", (16, frame.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                collect_timing(timings, "drawing", time.perf_counter() - started)

                started = time.perf_counter()
                cv2.imshow("ASL Sign Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                collect_timing(timings, "render + key handling", time.perf_counter() - started)
                collect_timing(timings, "overall frame loop", time.perf_counter() - loop_start)
                if key in (ord("q"), ord("Q")):
                    break
    finally:
        inference_stop.set()
        worker.join(timeout=2.0)
        cap.release()
        cv2.destroyAllWindows()
        report_timings(timings, prediction_updates, update_intervals)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
