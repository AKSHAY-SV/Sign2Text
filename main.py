"""Single-window static ASL fingerspelling webcam demo."""

from __future__ import annotations

import argparse
from math import floor, ceil
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    CAMERA_FPS, CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH,
    MODEL_PATH, PROBABILITY_HISTORY_SIZE,
)
from src.camera.camera import Webcam
from src.detection.detector import HandDetector
from src.prediction.predictor import PredictionStabilizer
from src.reference_recognizer import ReferenceSignRecognizer, preprocess_hand
from src.text.buffer import TextBuffer
from src.vision.display import to_display_uint8
from src.vision.hand_crop import crop_hand_with_bounds
from src.model.cnn_classifier import CnnClassifier, preprocess_image


WINDOW = "Sign2Text"
REFERENCE_CONFIDENCE_THRESHOLD = 0.80
REFERENCE_STABLE_FRAMES = 12


def _insert(buffer: TextBuffer, label: str) -> bool:
    """Apply one confirmed model class to the text buffer."""
    if len(label) == 1 and label.isalnum():
        buffer.add_sign(label)
        return True
    if label == "space":
        buffer.add_space()
        return True
    if label == "del":
        buffer.backspace()
        return True
    return False  # nothing is deliberately non-textual


def _panel(lines: list[str], width: int = 390, height: int = 230) -> np.ndarray:
    panel = np.full((height, width, 3), 28, dtype=np.uint8)
    for index, line in enumerate(lines):
        cv2.putText(panel, line[:52], (10, 24 + 24 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (235, 235, 235), 1, cv2.LINE_AA)
    return panel


def _compose(frame: np.ndarray, diagnostic: bool, crop: np.ndarray | None,
             input_rgb: np.ndarray | None, lines: list[str]) -> np.ndarray:
    """Compose the camera and optional diagnostics into one application window."""
    height, width = frame.shape[:2]
    camera_view = cv2.resize(frame, (width, height))
    if diagnostic:
        panel_width = 390
        panel_height = 220
        crop_panel = np.zeros((panel_height, panel_width, 3), np.uint8) if crop is None else cv2.resize(crop, (panel_width, panel_height))
        if input_rgb is None:
            input_panel = np.zeros((panel_height, panel_width, 3), np.uint8)
        else:
            rgb8 = to_display_uint8(input_rgb)
            input_panel = cv2.cvtColor(cv2.resize(rgb8, (panel_width, panel_height)), cv2.COLOR_RGB2BGR)
        right = np.vstack((crop_panel, input_panel, _panel(lines, panel_width, 460)))
        left = cv2.resize(camera_view, (int(right.shape[0] * width / height), right.shape[0]))
        return np.hstack((left, right))
    return camera_view


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", action="store_true",
                        help="show crop, exact model input and top-three scores inside the main window")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX)
    parser.add_argument("--candidate-fallback", action="store_true",
                        help="use the preserved project 64x64 candidate CNN instead of the reference recognizer")
    parser.add_argument("--show-landmarks", action="store_true",
                        help="draw landmarks on the preview copy (never on classifier input)")
    args = parser.parse_args()

    candidate = None
    if args.candidate_fallback:
        model_dir = Path(__file__).resolve().parent / "models/candidate_asl_letters_v2"
        candidate = CnnClassifier(model_dir / "best.keras", model_dir / "metadata.json")
        class_names = candidate.class_names
        print(f"Fallback model: {candidate.model_path}\nMetadata: {candidate.metadata_path}")
    else:
        recognizer = ReferenceSignRecognizer()
        class_names = recognizer.classes
        print(f"Reference model: {recognizer.model_path}\nClass map: {recognizer.mapping_path}")

    camera = Webcam(camera_index=args.camera, width=CAMERA_WIDTH,
                    height=CAMERA_HEIGHT, fps=CAMERA_FPS)
    detector = HandDetector(model_path=str(MODEL_PATH), num_hands=1)
    buffer = TextBuffer()
    stabilizer = PredictionStabilizer(
        confidence_threshold=REFERENCE_CONFIDENCE_THRESHOLD,
        probability_history_size=PROBABILITY_HISTORY_SIZE,
        min_consistent_predictions=REFERENCE_STABLE_FRAMES,
        confirmation_hold_seconds=0.0,
    )
    print("ENTER capture confirmed class | SPACE separator | BACKSPACE delete | C clear | Q/close quit")

    crop = input_rgb = scores = geometry = None
    raw_label = None
    confirmed = None
    try:
        camera.open()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            frame = camera.read()
            if frame is None:
                continue
            source_frame = frame
            detection = detector.detect(frame)
            crop = input_rgb = scores = geometry = None
            raw_label = confirmed = None
            raw_score = None

            if detection.hands:
                # The selected reference model was collected with its unmirrored
                # camera image and right hand; keep that convention for inference.
                hand = next((item for item in detection.hands if item.label.lower() == "right"), None)
                if hand is not None:
                    if candidate is None:
                        points = hand.landmarks
                        h, w = source_frame.shape[:2]
                        xs = np.asarray([point.x * w for point in points], dtype=np.float64)
                        ys = np.asarray([point.y * h for point in points], dtype=np.float64)
                        if (len(points) == 21 and np.isfinite(xs).all() and np.isfinite(ys).all()
                                and np.all((0 <= xs) & (xs <= w)) and np.all((0 <= ys) & (ys <= h))):
                            left, top = max(0, floor(float(xs.min())) - 60), max(0, floor(float(ys.min())) - 60)
                            right, bottom = min(w, ceil(float(xs.max())) + 61), min(h, ceil(float(ys.max())) + 61)
                            if right > left and bottom > top:
                                geometry = (left, top, right, bottom)
                                crop = source_frame[top:bottom, left:right]
                                batch, scores = recognizer.predict_crop(crop)
                                input_rgb = batch[0]
                    else:
                        geometry = crop_hand_with_bounds(source_frame, hand.landmarks)
                        if geometry is not None:
                            crop = geometry.image
                            input_rgb = preprocess_image(crop, (candidate.image_size, candidate.image_size))
                            scores = candidate.predict_proba(input_rgb)
                if scores is not None:
                    idx = int(np.argmax(scores))
                    raw_label = class_names[idx]
                    raw_score = float(scores[idx])
                    confirmed = stabilizer.update_probabilities(scores, class_names)
                else:
                    stabilizer.reset()
            else:
                stabilizer.reset()

            # UI is drawn only on a separate preview copy, never the model tensor.
            preview = cv2.flip(source_frame, 1)
            if args.show_landmarks and detection.hands:
                from src.vision.visualizer import HandVisualizer
                preview = cv2.flip(HandVisualizer().draw(source_frame.copy(), detection, show_landmarks=True), 1)
            display_lines = [
                "LIVE CAMERA PREVIEW",
                f"RAW: {raw_label or ('NO RIGHT HAND' if detection.hands else 'NO HAND DETECTED')}" + (f" {raw_score:.1%}" if raw_score is not None else ""),
                f"CONFIRMED: {confirmed or 'waiting'}",
                f"TEXT: {buffer.sentence}",
                "ENTER capture | SPACE word | BACKSPACE | C clear | Q quit",
            ]
            cv2.rectangle(preview, (0, 0), (preview.shape[1], 142), (20, 20, 20), -1)
            for row, line in enumerate(display_lines):
                cv2.putText(preview, line, (10, 24 + row * 25), cv2.FONT_HERSHEY_SIMPLEX,
                            .55, (230, 240, 255), 1, cv2.LINE_AA)

            diag_lines = [f"Frame {source_frame.shape[1]}x{source_frame.shape[0]}"]
            if geometry is None:
                diag_lines += ["NO RIGHT HAND DETECTED" if detection.hands else "NO HAND DETECTED",
                               "Crop: unavailable", "Input: unavailable"]
                shown_crop = shown_input = None
            else:
                bounds = geometry.bounds if candidate is not None else geometry
                diag_lines += [f"Crop LTRB: {bounds}",
                               f"Crop: {crop.shape[1]}x{crop.shape[0]} BGR",
                               f"Input: {input_rgb.shape} RGB float32 /255"]
                top = np.argsort(scores)[-3:][::-1]
                diag_lines += [f"Top {rank}: {class_names[int(i)]} {scores[i]:.1%}"
                               for rank, i in enumerate(top, 1)]
                shown_crop, shown_input = crop, input_rgb
            composed = _compose(preview, args.diagnostic, shown_crop, shown_input, diag_lines)
            cv2.imshow(WINDOW, composed)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (10, 13):
                committed = stabilizer.commit(confirmed, explicit=True)
                if committed and _insert(buffer, committed):
                    print(f"Captured {committed}: {buffer.sentence}")
                elif committed == "nothing":
                    print("Neutral sign; text unchanged.")
            elif key == 32:
                buffer.add_space()
                stabilizer.rearm()
            elif key in (8, 127):
                buffer.backspace()
                stabilizer.rearm()
            elif key in (ord("c"), ord("C")):
                buffer.clear()
                stabilizer.rearm()

            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
