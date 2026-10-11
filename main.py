"""
Sign2Text - Main Entry Point

Real-time sign language to text conversion system.

Pipeline:
    Laptop Camera
    → OpenCV frame capture
    → MediaPipe Hand Landmarker (HandDetector)
    → Feature Extraction (FeatureExtractor)
    → ML Classifier (SignClassifier)
    → Prediction Stabilization (PredictionStabilizer)
    → Text Buffer (TextBuffer)
    → Serial UART → LPC2148 → LCD

The laptop performs ALL processing. The LPC2148 only receives text
and displays it on its LCD. No image processing, MediaPipe, ML, or
sign recognition runs on the LPC2148.

Usage:
    python main.py

Controls:
    SPACE - Finalize current sign and add to text buffer
    BACKSPACE - Delete last character/word
    CLEAR (C) - Clear entire text buffer
    Q - Quit application
    L - Toggle landmark overlay (show/hide hand skeleton)
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.camera.camera import Webcam
from src.detection.detector import HandDetector
from src.features.extractor import FeatureExtractor
from src.model.cnn_classifier import CnnClassifier
from src.prediction.predictor import RealTimePredictor, PredictionStabilizer
from src.text.buffer import TextBuffer
from src.communication.serial import SerialCommunicator
from src.vision.visualizer import HandVisualizer

from config.settings import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_FPS,
    CAMERA_WARMUP_FRAMES,
    CAMERA_MAX_READ_RETRIES,
    WINDOW_NAME,
    MIRROR_FEED,
    CONFIDENCE_THRESHOLD,
    DEBOUNCE_FRAMES,
    MAX_SAME_SIGN_REPEAT,
    SERIAL_PORT,
    BAUD_RATE,
    SERIAL_TIMEOUT,
    SHOW_FPS,
    SHOW_CONFIDENCE,
    SHOW_PREDICTION,
    SHOW_TEXT_BUFFER,
    MODEL_PATH,
    MODELS_DIR,
    CLASSIFIER_TYPE,
    SHOW_LANDMARKS,
    CNN_INFERENCE_INTERVAL,
    ENABLE_TIMING_MEASUREMENTS,
)

# Constants
_KEY_QUIT = ord("q")
KEY_SPACE = 32
KEY_BACKSPACE = 8
KEY_CLEAR = ord("c")
KEY_ENTER = 13


def signal_handler(signum, frame) -> None:
    """Handle OS signals (Ctrl+C) gracefully."""
    print(f"\nReceived signal {signum}. Exiting gracefully...")


def main() -> None:
    """Run the real-time Sign2Text application."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Initialize text buffer
    text_buffer = TextBuffer()

    # Initialize serial communication with LPC2148
    serial_comm = SerialCommunicator(
        port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
        timeout=SERIAL_TIMEOUT,
    )

    # Initialize camera
    try:
        camera = Webcam(
            camera_index=CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            warmup_frames=CAMERA_WARMUP_FRAMES,
            max_read_retries=CAMERA_MAX_READ_RETRIES,
        )
        camera.open()
    except RuntimeError as e:
        print(f"Error: {e}")
        print("  Falling back to default camera index 0...")
        camera = Webcam(
            camera_index=0,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=CAMERA_FPS,
            warmup_frames=CAMERA_WARMUP_FRAMES,
            max_read_retries=CAMERA_MAX_READ_RETRIES,
        )
        camera.open()

    # Initialize hand detector
    detector = HandDetector(
        model_path=str(MODEL_PATH),
        num_hands=2,
    )

    # Initialize hand visualizer
    visualizer = HandVisualizer()

    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Initialize classifier
    from src.model.classifier import SignClassifier

    classifier_model_path = str(MODELS_DIR / "sign_classifier.pkl")
    classifier = SignClassifier(
        model_path=classifier_model_path if Path(classifier_model_path).is_file() else None,
        classifier_type=CLASSIFIER_TYPE,
    )

    # Load the CNN classifier for sign recognition.
    cnn_model_path = str(
        MODELS_DIR / "candidate_asl_letters_v2" / "best.keras"
    )
    cnn_classifier = None
    if Path(cnn_model_path).is_file():
        try:
            cnn_classifier = CnnClassifier(model_path=cnn_model_path)
            metadata_path = (
                MODELS_DIR / "candidate_asl_letters_v2" / "metadata.json"
            )
            if metadata_path.is_file():
                cnn_classifier.load_metadata(str(metadata_path))
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"CNN disabled: failed to load compatible model: {exc}")
    else:
        print(f"CNN unavailable: model file not found at {cnn_model_path}")

    # Initialize prediction stabilizer
    stabilizer = PredictionStabilizer(
        stable_frames=DEBOUNCE_FRAMES,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    # Initialize real-time predictor (CNN classifier passed separately;
    #   predictor falls back to classical ML if CNN not trained)
    predictor = RealTimePredictor(
        detector=detector,
        extractor=extractor,
        classifier=classifier,
        cnn_classifier=cnn_classifier,
        stabilizer=stabilizer,
    )

    if cnn_classifier and cnn_classifier.is_trained():
        print(f"CNN classifier loaded: {cnn_classifier.num_classes} classes, image_size={cnn_classifier.image_size}")
        val_accuracy = cnn_classifier.metadata.get("final_val_accuracy")
        if val_accuracy is not None:
            print(f"Recorded validation accuracy: {val_accuracy:.1%}; treat predictions cautiously.")
    else:
        print("No compatible CNN model is available. Recognition is disabled until one is installed.")

    # Main loop variables
    running = True
    frame_count = 0
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    current_fps = 0

    # CNN inference control
    cnn_inference_interval = CNN_INFERENCE_INTERVAL if (cnn_classifier and cnn_classifier.is_trained()) else 1
    last_cnn_prediction: Optional[str] = None
    last_cnn_confidence: Optional[float] = None
    last_cnn_detection_timestamp: Optional[int] = None
    frames_since_cnn_inference = 0

    # Timing measurements
    timing_stats = {
        'camera_read': [],
        'landmark_detection': [],
        'cnn_inference': [],
        'display_drawing': [],
        'total_frame': []
    }

    print("Sign2Text starting...")
    print("=" * 60)
    print("Press SPACE to capture/sign and add to text")
    print("Press BACKSPACE to delete last character/word")
    print("Press CLEAR (C) to clear entire text")
    print("Press L to toggle landmark overlay")
    print("Press Q to quit")
    print("=" * 60)

    # Local variable for landmark display toggle
    show_landmarks = SHOW_LANDMARKS

    try:
        while running:
            frame_start_time = time.perf_counter()
            
            # Read frame from camera
            camera_read_start = time.perf_counter()
            frame = camera.read()
            camera_read_time = time.perf_counter() - camera_read_start
            
            if frame is None:
                print("Failed to read frame from camera.", file=sys.stderr)
                break

            # Mirror the feed so the display matches natural hand motion
            if MIRROR_FEED:
                frame = cv2.flip(frame, 1)

            # Run landmark detection
            landmark_start = time.perf_counter()
            detection = predictor._detector.detect(frame)
            landmark_time = time.perf_counter() - landmark_start

            # Run prediction pipeline (CNN-based if model available, else classical ML)
            cnn_inference_time = 0.0
            cnn_inference_this_frame = False
            
            diagnostic_reason = "No hand detected" if detection.num_hands == 0 else ""
            if predictor._cnn_classifier is not None and predictor._cnn_classifier.is_trained():
                if detection.num_hands == 0:
                    predictor._stabilizer.update(None, None)
                    last_cnn_prediction, last_cnn_confidence = None, None
                    last_cnn_detection_timestamp = None
                    frames_since_cnn_inference = 0
                    prediction, confidence = None, None
                else:
                    frames_since_cnn_inference += 1
                
                if detection.num_hands and frames_since_cnn_inference >= cnn_inference_interval:
                    # Run CNN inference
                    cnn_start = time.perf_counter()
                    prediction, confidence = predictor.predict_from_frame_cnn(frame, detection=detection)
                    cnn_inference_time = time.perf_counter() - cnn_start
                    cnn_inference_this_frame = True
                    frames_since_cnn_inference = 0
                    last_cnn_prediction = prediction
                    last_cnn_confidence = confidence
                    last_cnn_detection_timestamp = detection.timestamp_ms
                elif detection.num_hands:
                    # Reuse last CNN prediction
                    prediction = last_cnn_prediction
                    confidence = last_cnn_confidence
                if prediction is None and detection.num_hands:
                    diagnostic_reason = "CNN returned no valid prediction"
                elif prediction is not None:
                    diagnostic_reason = ""
                stable_label = predictor._stabilizer.current_stable_label
                if stable_label is None and detection.num_hands and prediction is not None:
                    diagnostic_reason = "Holding sign; waiting for stable frames"
                prediction = stable_label
                if prediction is None:
                    confidence = last_cnn_confidence
            else:
                # The classical classifier consumes landmark vectors and is a
                # separate model format; don't imply it is a CNN fallback.
                prediction, confidence = (None, None)
                if detection.num_hands:
                    diagnostic_reason = "No CNN loaded; no compatible landmark model loaded"

            # Update FPS counter
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps_end_time = cv2.getTickCount()
                freq = cv2.getTickFrequency()
                elapsed = (fps_end_time - fps_start_time) / freq
                current_fps = 10.0 / elapsed if elapsed > 0 else 0
                fps_start_time = cv2.getTickCount()

            # Display window with UI overlay
            display_frame = frame.copy()

            # Draw UI panel
            # Background panel
            cv2.rectangle(display_frame, (0, 0), (int(display_frame.shape[1] * 0.8), 180),
                          (40, 40, 40), -1)
            cv2.rectangle(display_frame, (0, 0), (int(display_frame.shape[1] * 0.8), 180),
                          (255, 255, 255), 1)

            # FPS display
            if SHOW_FPS:
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    display_frame,
                    fps_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Hand detection status
            num_hands = detection.num_hands if detection else 0
            hands_text = f"Hands: {num_hands}"
            color = (0, 255, 0) if num_hands > 0 else (0, 0, 255)
            cv2.putText(
                display_frame,
                f"Hands: {num_hands}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

            model_status = "CNN ready" if cnn_classifier and cnn_classifier.is_trained() else "Model unavailable"
            cv2.putText(display_frame, model_status, (10, 125), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 220, 255) if "unavailable" in model_status.lower() else (0, 255, 0), 1, cv2.LINE_AA)
            if diagnostic_reason:
                cv2.putText(display_frame, diagnostic_reason[:70], (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 180, 255), 1, cv2.LINE_AA)
            # Render skeleton on top of the UI panel so landmarks remain visible
            # even when a hand is near the top of the camera frame.
            if show_landmarks:
                display_frame = visualizer.draw(display_frame, detection, show_landmarks=True)

            # Prediction display
            if SHOW_PREDICTION and prediction is not None:
                label_text = f"Sign: {prediction}"
                if confidence is not None:
                    label_text += f" ({confidence:.1%})"
                cv2.putText(
                    display_frame,
                    label_text,
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if SHOW_PREDICTION and prediction is None:
                cv2.putText(display_frame, "Sign: --", (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 180, 255), 2, cv2.LINE_AA)

            if SHOW_CONFIDENCE and confidence is not None:
                conf_text = f"Conf: {confidence:.1%}"
                cv2.putText(
                    display_frame,
                    conf_text,
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Text buffer display
            if SHOW_TEXT_BUFFER:
                sentence = text_buffer.sentence
                if sentence:
                    # Truncate if too long for display
                    display_text = sentence[:30] + ("..." if len(sentence) > 30 else "")
                    buf_text = f"Text: {display_text}"
                    cv2.putText(
                        display_frame,
                        buf_text,
                        (10, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )

            # Performance timing display
            if ENABLE_TIMING_MEASUREMENTS:
                timing_text = [
                    f"Cam: {camera_read_time*1000:.1f}ms",
                    f"Landmark: {landmark_time*1000:.1f}ms",
                    f"CNN: {cnn_inference_time*1000:.1f}ms" if cnn_inference_this_frame else "CNN: reused",
                    f"Draw: {(time.perf_counter() - frame_start_time)*1000:.1f}ms"
                ]
                y_pos = 130
                for text in timing_text:
                    cv2.putText(
                        display_frame,
                        text,
                        (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                    y_pos += 18

            # Instructions at bottom
            cv2.putText(
                display_frame,
                "SPACE: capture  BACKSPACE: delete  CLEAR: clear  L: landmarks  Q: quit",
                (10, display_frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Show the frame
            cv2.imshow(WINDOW_NAME, display_frame)

            # Record total frame time
            total_frame_time = time.perf_counter() - frame_start_time
            if ENABLE_TIMING_MEASUREMENTS:
                timing_stats['camera_read'].append(camera_read_time)
                timing_stats['landmark_detection'].append(landmark_time)
                timing_stats['cnn_inference'].append(cnn_inference_time)
                timing_stats['display_drawing'].append(time.perf_counter() - (frame_start_time + camera_read_time + landmark_time + cnn_inference_time))
                timing_stats['total_frame'].append(total_frame_time)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == _KEY_QUIT:
                running = False
                print("Quit requested.")

            elif key == KEY_SPACE:
                # Finalize current sign and add to text buffer
                stabilized = predictor._stabilizer.current_stable_label
                if stabilized and confidence is not None and confidence >= CONFIDENCE_THRESHOLD and stabilized != "nothing":
                    text_buffer.add_sign(stabilized)
                    # Send to LPC2148
                    current_sentence = text_buffer.sentence
                    if serial_comm.is_connected():
                        serial_comm.send_text(current_sentence)
                    print(f"Sent to LPC2148: {current_sentence}")
                else:
                    print("No confident non-neutral sign to add.")

            elif key == KEY_BACKSPACE:
                # Delete last character/word
                text_buffer.backspace()
                # Send backspace command to LPC2148
                if serial_comm.is_connected():
                    serial_comm.send_text("BACKSPACE")
                print("Backspace executed.")

            elif key == KEY_CLEAR:
                # Clear entire text buffer
                text_buffer.clear()
                # Send clear command to LPC2148
                if serial_comm.is_connected():
                    serial_comm.send_text("CLEAR")
                print("Text cleared.")

            elif key == ord('l') or key == ord('L'):
                # Toggle landmark display (local variable)
                show_landmarks = not show_landmarks
                print(f"Landmark overlay: {'ON' if show_landmarks else 'OFF'}")

            elif key == KEY_ENTER:
                # Finalize and send current sentence
                current_sentence = text_buffer.sentence
                if current_sentence and serial_comm.is_connected():
                    serial_comm.send_text(current_sentence)
                    print(f"Sent via ENTER: {current_sentence}")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received.")

    finally:
        # Print timing statistics
        if ENABLE_TIMING_MEASUREMENTS and timing_stats['total_frame']:
            print("\n=== Timing Statistics ===")
            for key, values in timing_stats.items():
                if values:
                    avg = sum(values) / len(values) * 1000
                    max_val = max(values) * 1000
                    min_val = min(values) * 1000
                    print(f"  {key}: avg={avg:.1f}ms, min={min_val:.1f}ms, max={max_val:.1f}ms, samples={len(values)}")

        # Clean up resources (safe if any component was never initialized)
        if 'predictor' in locals() and predictor is not None:
            predictor.reset_prediction()

        if 'camera' in locals() and camera is not None:
            camera.release()

        if 'detector' in locals() and detector is not None:
            detector.close()

        cv2.destroyAllWindows()

        if 'serial_comm' in locals() and serial_comm is not None:
            serial_comm.close()

        print("Sign2Text exited gracefully.")


if __name__ == "__main__":
    main()
