#!/usr/bin/env python3
"""Launch the separate fingerspelling alphabet webcam recognizer."""

import argparse
import sys
from pathlib import Path

import cv2
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sign2text.recognizers.alphabet import AlphabetRecognizer, StableTextBuffer
from sign2text.recognizers.alphabet.adapter import DEFAULT_REPOSITORY


WINDOW = "ASL Alphabet Fingerspelling"
LOW_CONFIDENCE_THRESHOLD = 0.50


def run(camera: int, repository: Path, refresh_interval: int,
        commit_frames: int, commit_confidence: float, show_landmarks: bool = False) -> int:
    recognizer = AlphabetRecognizer(repository)
    buffer = StableTextBuffer(commit_frames, commit_confidence)
    hands_api = mp.solutions.hands
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        capture.release()
        recognizer.close()
        raise SystemExit(f"Could not open webcam {camera}.")

    frame_number = 0
    result = None
    window_shown = False
    try:
        with hands_api.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
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
                frame = cv2.flip(frame, 1)  # matches the alphabet repository's camera path
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                detected = hands.process(rgb)
                rgb.flags.writeable = True
                frame_number += 1
                landmarks = detected.multi_hand_landmarks[0] if detected.multi_hand_landmarks else None

                if landmarks is None:
                    recognizer.reset()
                    buffer.update(None, 0.0, hand_present=False)
                    status = "No hand detected"
                    result = None
                else:
                    should_infer = (frame_number - 1) % refresh_interval == 0
                    result = recognizer.process_hand(landmarks, width, height, infer=should_infer)
                    if result.updated and result.best:
                        buffer.update(
                            result.best.label,
                            result.best.confidence,
                            moving=result.moving,
                        )
                    status = result.status
                    if show_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, landmarks, hands_api.HAND_CONNECTIONS,
                            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                            mp.solutions.drawing_styles.get_default_hand_connections_style(),
                        )

                cv2.putText(frame, "ASL FINGERSPELLING", (18, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.72, (0, 220, 150), 2, cv2.LINE_AA)
                cv2.putText(frame, status.upper(), (18, 62), cv2.FONT_HERSHEY_SIMPLEX,
                            0.58, (255, 255, 255), 1, cv2.LINE_AA)
                if result and result.best:
                    if result.best.confidence < LOW_CONFIDENCE_THRESHOLD:
                        prefix = "Low confidence"
                    else:
                        prefix = "Letter"
                    if result.best.motion_resolved:
                        reading = (f"{prefix}: {result.best.label.upper()} "
                                   f"(motion rule; pose {result.best.confidence:.1%})")
                    else:
                        reading = f"{prefix}: {result.best.label.upper()} {result.best.confidence:.1%}"
                    color = ((0, 220, 150) if result.best.confidence >= commit_confidence
                             else (0, 180, 255))
                    cv2.putText(frame, reading,
                                (18, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                    for rank, prediction in enumerate(result.predictions, start=1):
                        cv2.putText(frame,
                                    f"{rank}. {prediction.label.upper()}  {prediction.confidence:.1%}",
                                    (18, 137 + 25 * (rank - 1)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frame, f"Stable: {buffer.progress:.0%}", (18, 220),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
                cv2.putText(frame, f"Text: {buffer.text or '(empty)'}", (18, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Q quit | C clear | Backspace undo | Space separator | R repeat",
                            (18, height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                            (210, 210, 210), 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, frame)
                window_shown = True
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (8, 127):
                    buffer.backspace()
                elif key in (ord("c"), ord("C")):
                    buffer.clear()
                elif key == ord(" "):
                    buffer.space()
                elif key in (ord("r"), ord("R")):
                    buffer.repeat()
    finally:
        capture.release()
        recognizer.close()
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--repository", type=Path, default=DEFAULT_REPOSITORY)
    parser.add_argument("--refresh-interval", type=int, default=1,
                        help="run ONNX inference every N frames (default: 1)")
    parser.add_argument("--commit-frames", type=int, default=10)
    parser.add_argument("--commit-confidence", type=float, default=0.75)
    parser.add_argument("--show-landmarks", action="store_true",
                        help="draw hand landmarks for diagnostics")
    args = parser.parse_args(argv)
    if args.refresh_interval < 1:
        parser.error("--refresh-interval must be at least 1")
    if args.commit_frames < 1:
        parser.error("--commit-frames must be at least 1")
    if not 0 <= args.commit_confidence <= 1:
        parser.error("--commit-confidence must be between 0 and 1")
    return run(args.camera, args.repository, args.refresh_interval,
               args.commit_frames, args.commit_confidence, args.show_landmarks)


if __name__ == "__main__":
    raise SystemExit(main())
