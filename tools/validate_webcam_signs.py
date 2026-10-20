"""Continuous webcam validation for the pretrained isolated-sign recognizer."""

import json
import os
import sys
from collections import deque

import cv2
import mediapipe as mp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from launch_demo import (  # noqa: E402
    LOW_CONFIDENCE_THRESHOLD,
    SEQUENCE_LENGTH,
    build_model_input,
    draw_visible_landmarks,
    extract_landmarks,
    is_prediction_frame,
    load_resources,
    mapped_predictions,
    normalize_frame,
    prediction_description,
)


def main():
    model, glosses = load_resources()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return 1

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    frame_number = 0
    frames_with_landmarks = 0
    previous_index = None
    prediction_stability_count = 0
    predicted_label = None
    predicted_index = None
    confidence = 0.0
    status = "insufficient frames"
    top_predictions = []
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

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
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    print("ERROR: Failed to grab frame")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = holistic.process(rgb)

                landmarks = normalize_frame(extract_landmarks(results))
                sequence.append(landmarks)
                frame_number += 1
                has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
                if results.pose_landmarks or has_hand:
                    frames_with_landmarks += 1

                if len(sequence) < SEQUENCE_LENGTH:
                    status = "insufficient frames"
                elif is_prediction_frame(frame_number, 5):
                    probabilities = model.predict(build_model_input(sequence), verbose=0)[0]
                    predicted_label, predicted_index, confidence, status, _ = (
                        prediction_description(probabilities, glosses)
                    )
                    top_predictions = mapped_predictions(probabilities, glosses)
                    if confidence < LOW_CONFIDENCE_THRESHOLD and predicted_label is not None:
                        status = "low confidence"
                    if predicted_index == previous_index:
                        prediction_stability_count += 1
                    else:
                        prediction_stability_count = 1
                    previous_index = predicted_index

                draw_visible_landmarks(frame, results, mp_holistic, mp_drawing)
                if status == "insufficient frames":
                    headline = f"Waiting for frames: {len(sequence)}/{SEQUENCE_LENGTH}"
                elif status.startswith("unmapped"):
                    headline = f"Unknown label ({status})"
                elif predicted_label is None:
                    headline = "No mapped prediction"
                else:
                    headline = f"Sign: {predicted_label} ({status})"

                cv2.putText(frame, headline, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "Hand detected" if has_hand else "No hand detected",
                            (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)
                for rank, (index, label, score) in enumerate(top_predictions, 1):
                    cv2.putText(frame, f"{rank}. {label}: {score:.1%}",
                                (10, 112 + (rank - 1) * 23), cv2.FONT_HERSHEY_SIMPLEX,
                                0.48, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Stable predictions: {prediction_stability_count}",
                            (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("ASL Sign Recognition Validation", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s") and predicted_index is not None:
                    result = {
                        "timestamp": cv2.getTickCount(),
                        "predicted_index": predicted_index,
                        "predicted_label": predicted_label,
                        "confidence": confidence,
                        "status": status,
                        "top_mapped_predictions": [
                            {"index": int(i), "label": label, "confidence": score}
                            for i, label, score in top_predictions
                        ],
                        "sequence_length": len(sequence),
                        "frames_with_landmarks": frames_with_landmarks,
                        "landmark_feature_count": 86,
                        "input_feature_count": 172,
                    }
                    with open("sign_prediction_result.json", "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2)
                    print(json.dumps(result, indent=2))
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("Webcam closed. Resources cleaned up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
