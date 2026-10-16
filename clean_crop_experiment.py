"""Isolated webcam experiment using clean RGB hand crops and confirmed text."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from config.settings import CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, MODEL_PATH, CONFIRMATION_HOLD_SECONDS
from src.camera.camera import Webcam
from src.detection.detector import HandDetector
from src.model.cnn_classifier import CnnClassifier, preprocess_image
from src.prediction.predictor import PredictionStabilizer
from src.text.buffer import TextBuffer
from src.vision.display import to_display_uint8
from src.vision.hand_crop import crop_hand_with_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--show-landmarks", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="show the exact model input and top-3 scores")
    parser.add_argument("--save-dir", type=Path, default=Path("/tmp/sign2text_diagnostic_crops"))
    args = parser.parse_args()

    model_dir = Path("models/candidate_asl_letters_v2")
    classifier = CnnClassifier(model_dir / "best.keras", model_dir / "metadata.json")
    meta = classifier.metadata
    if (classifier.image_size != 64 or classifier.class_names != meta["class_names"]
            or classifier.num_classes != 29 or len(classifier.class_names) != 29
            or list(classifier.model.input_shape[1:]) != [64, 64, 3]
            or meta.get("input_shape") != [64, 64, 3]
            or meta.get("color_order") != "RGB"
            or meta.get("normalization") != "float32 pixel values divided by 255, range [0,1]"):
        raise ValueError("Candidate model preprocessing or class metadata does not match the expected RGB contract.")
    detector = HandDetector(model_path=str(MODEL_PATH), num_hands=1)
    camera = Webcam(camera_index=args.camera, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    buffer = TextBuffer()
    stabilizer = PredictionStabilizer(probability_history_size=3, min_consistent_predictions=3,
                                      confirmation_hold_seconds=CONFIRMATION_HOLD_SECONDS)
    raw_label = confirmed_label = None
    raw_score = confirmed_score = None
    last_crop = last_input = last_scores = last_geometry = None
    show_landmarks = args.show_landmarks
    print("Clean-crop experiment: Enter commits confirmed letter; Space inserts word separator; Backspace deletes; C clears; L toggles landmarks; S saves current crop; Q quits.")
    try:
        tf.config.set_visible_devices([], "GPU")
        camera.open()
        while True:
            frame = camera.read()
            if frame is None:
                continue
            frame = cv2.flip(frame, 1)
            detection = detector.detect(frame)
            last_crop = last_input = last_scores = last_geometry = None
            raw_label = confirmed_label = None
            raw_score = confirmed_score = None
            if detection.hands:
                hand = detection.hands[0]
                last_geometry = crop_hand_with_bounds(frame, hand.landmarks)
            if last_geometry is not None:
                last_crop = last_geometry.image
                last_input = preprocess_image(last_crop, (classifier.image_size, classifier.image_size))
                if last_input.shape != (64, 64, 3) or last_input.dtype != np.float32 or not np.isfinite(last_input).all() or last_input.min() < 0 or last_input.max() > 1:
                    raise ValueError("Invalid CNN input tensor contract.")
                last_scores = classifier.predict_proba(last_input)
                if (last_scores is None or last_scores.shape != (len(classifier.class_names),)
                        or not np.all(np.isfinite(last_scores)) or np.any(last_scores < 0)
                        or not np.isclose(float(last_scores.sum()), 1.0, atol=1e-3)):
                    raise ValueError("Classifier returned invalid class probabilities.")
                idx = int(np.argmax(last_scores))
                raw_label, raw_score = classifier.class_names[idx], float(last_scores[idx])
                confirmed_label = stabilizer.update_probabilities(last_scores, classifier.class_names)
                confirmed_score = stabilizer.current_confidence
            else:
                stabilizer.reset()

            # Keep the camera preview pristine. Optional landmarks use a separate copy/window.
            cv2.imshow("Clean crop experiment", frame)
            if show_landmarks:
                from src.vision.visualizer import HandVisualizer
                landmark_view = HandVisualizer().draw(frame.copy(), detection, show_landmarks=True)
                cv2.imshow("Landmarks (preview only)", landmark_view)
            else:
                try:
                    cv2.destroyWindow("Landmarks (preview only)")
                except cv2.error:
                    pass
            if args.diagnostic:
                if last_input is not None:
                    crop_view = cv2.resize(last_crop, (320, 320), interpolation=cv2.INTER_NEAREST)
                    input_view = cv2.cvtColor(cv2.resize(to_display_uint8(last_input), (320, 320), interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR)
                    cv2.imshow("Hand crop before resize (BGR)", crop_view)
                    cv2.imshow("Exact CNN RGB input", input_view)
                    top = np.argsort(last_scores)[-3:][::-1]
                    lines = [
                        f"Source frame: {last_geometry.source_shape[1]}x{last_geometry.source_shape[0]}",
                        f"Crop bounds L,T,R,B: {last_geometry.bounds}",
                        f"Landmark bounds: {last_geometry.landmark_bounds}",
                        f"Crop before resize: {last_crop.shape[1]}x{last_crop.shape[0]} BGR",
                        f"CNN input: {last_input.shape} RGB {last_input.dtype} [0,1]",
                        f"Raw: {raw_label} {raw_score:.1%} | confirmed: {confirmed_label or 'waiting'}",
                        *[f"Top {rank}: {classifier.class_names[int(i)]} {last_scores[i]:.1%}" for rank, i in enumerate(top, 1)],
                        f"Text: {buffer.sentence}",
                    ]
                    print(f"frame={last_geometry.source_shape[1]}x{last_geometry.source_shape[0]} crop={last_geometry.bounds} crop_shape={last_crop.shape} model_input={last_input.shape} {last_input.dtype} RGB")
                else:
                    cv2.imshow("Hand crop before resize (BGR)", np.zeros((320,320,3), np.uint8))
                    cv2.imshow("Exact CNN RGB input", np.zeros((320,320,3), np.uint8))
                    lines = ["NO HAND DETECTED" if not detection.hands else "INVALID HAND CROP: inference rejected",
                             f"Source frame: {frame.shape[1]}x{frame.shape[0]}",
                             "Crop bounds: unavailable", "CNN input: unavailable", f"Text: {buffer.sentence}"]
                panel = np.zeros((250, 680, 3), dtype=np.uint8)
                for row, line in enumerate(lines):
                    cv2.putText(panel, line, (8, 22 + 23 * row), cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
                cv2.imshow("Crop geometry and predictions", panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (10, 13):
                committed = stabilizer.commit(confirmed_label, explicit=True)
                if committed and committed not in ("nothing", "del", "space"):
                    buffer.add_sign(committed)
                    print(f"Committed {committed}: {buffer.sentence}")
                elif committed == "del":
                    buffer.backspace()
                elif committed == "space":
                    buffer.add_space()
                else:
                    print("No new confirmed letter; use neutral/no-hand before repeating it.")
            elif key == 32:
                buffer.add_space()
                stabilizer.rearm()
            elif key in (8, 127):
                buffer.backspace()
                stabilizer.rearm()
            elif key in (ord("c"), ord("C")):
                buffer.clear()
                stabilizer.rearm()
            elif key in (ord("l"), ord("L")):
                show_landmarks = not show_landmarks
            elif key in (ord("s"), ord("S")) and last_input is not None:
                args.save_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000_000:09d}"
                image_path = args.save_dir / f"{stamp}.png"
                cv2.imwrite(str(image_path), cv2.cvtColor(to_display_uint8(last_input), cv2.COLOR_RGB2BGR))
                top = np.argsort(last_scores)[-3:][::-1]
                report = {"model": str(classifier.model_path), "metadata": str(classifier.metadata_path), "source_frame_shape": list(last_geometry.source_shape), "crop_bounds_ltrb": list(last_geometry.bounds), "landmark_bounds_ltrb": list(last_geometry.landmark_bounds), "crop_shape_bgr": list(last_crop.shape), "input_shape_rgb": list(last_input.shape), "dtype": str(last_input.dtype), "normalization": "float32 /255", "scores": [{"class": classifier.class_names[int(i)], "score": float(last_scores[i])} for i in top]}
                image_path.with_suffix(".json").write_text(json.dumps(report, indent=2))
                print(f"Saved {image_path}: " + ", ".join(f"{x['class']} {x['score']:.1%}" for x in report["scores"]))
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
