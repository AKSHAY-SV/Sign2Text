import cv2
import json
import time
import numpy as np
from pathlib import Path
from tensorflow import keras

from src.model.cnn_classifier import preprocess_image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models/candidate_asl_letters_v2/best.keras"
META_PATH = ROOT / "models/candidate_asl_letters_v2/metadata.json"

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise FileNotFoundError("Candidate model or metadata is missing.")

metadata = json.loads(META_PATH.read_text())
classes = metadata["class_names"]
size = int(metadata["image_size"])

print("Loading candidate CNN...")
model = keras.models.load_model(MODEL_PATH, compile=False)

assert model.input_shape[1:] == (size, size, 3), model.input_shape
assert model.output_shape[-1] == len(classes), model.output_shape

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam /dev/video0.")

print("\nWebcam test started.")
print("Put one ASL alphabet handshape in view.")
print("Press Q to quit.")
print("Press S to save a screenshot of the current frame.")

last_label = None
last_conf = 0.0
last_time = 0.0

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Could not read webcam frame.")
            break

        # Limit inference to approximately 5 times per second.
        now = time.time()
        if now - last_time >= 0.2:
            # Full-frame test first: isolates the CNN from hand-crop errors.
            resized = cv2.resize(
                frame, (size, size), interpolation=cv2.INTER_AREA
            )
            image = preprocess_image(resized, (size, size))
            scores = model.predict(image[None, ...], verbose=0)[0]

            top_indices = np.argsort(scores)[-3:][::-1]
            idx = int(top_indices[0])
            last_label = classes[idx]
            last_conf = float(scores[idx])
            last_time = now

            top_text = " | ".join(
                f"{classes[int(i)]}: {float(scores[i]):.1%}"
                for i in top_indices
            )
            print(f"Top 3: {top_text}")

        cv2.putText(
            frame,
            f"Candidate: {last_label} ({last_conf:.1%})",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "FULL FRAME TEST - press Q to quit",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Sign2Text Candidate CNN Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            out = ROOT / f"candidate_test_{int(time.time())}.jpg"
            cv2.imwrite(str(out), frame)
            print("Saved:", out)

finally:
    cap.release()
    cv2.destroyAllWindows()
