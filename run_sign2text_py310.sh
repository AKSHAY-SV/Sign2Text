#!/bin/bash
# run_sign2text_py310.sh - ASL Live Translation Launcher (Python 3.10 isolation)
#
# Launches the real webcam sign recognition pipeline using .venv-repair
# which has MediaPipe Holistic 0.10.21 compatible with Python 3.12.3.
# Collects 150 real webcam frames, preprocesses them using the original
# training-time pipeline, and runs pretrained model inference.
#
# Requirements:
#   - .venv-repair must exist alongside .venv
#   - models/best_model200.keras and models/index_to_gloss_200.json must exist
#
# Known limitation: index_to_gloss_200.json maps only 209 of 271 model outputs.
# Indices 209-270 are unmapped and will be reported without invented labels.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================="
echo "ASL Live Translation - Real Webcam Pipeline"
echo "============================================================="
echo ""

# Check if .venv-repair exists
if [ -d ".venv-repair" ]; then
    echo "Using .venv-repair virtual environment (MediaPipe 0.10.21 + Python 3.12)"
    PYTHON=".venv-repair/bin/python3"
else
    echo "ERROR: .venv-repair directory not found"
    echo "Expected at: $SCRIPT_DIR/.venv-repair"
    echo "The repository requires Python 3.10.17 with MediaPipe 0.10.9 for full pipeline."
    echo "See diagnostics/mediapipe_failure.txt for details."
    exit 1
fi

# Verify model exists
if [ ! -f "models/best_model200.keras" ]; then
    echo "ERROR: Model file models/best_model200.keras not found"
    exit 1
fi

# Verify labels exist
if [ ! -f "models/index_to_gloss_200.json" ]; then
    echo "ERROR: Label file models/index_to_gloss_200.json not found"
    exit 1
fi

echo "Model and labels verified."
echo "Collecting 150 real webcam frames using MediaPipe Holistic..."
echo "Place your hand in view of the webcam."
echo "Press Ctrl+C to exit after the pipeline completes."
echo ""

# Run the Python inference pipeline
"$PYTHON" - << 'PYEOF'
import mediapipe as mp
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
from launch_demo import normalize_frame

# Load model and label mapping
model = load_model('models/best_model200.keras')
with open('models/index_to_gloss_200.json') as f:
    gloss_dict = json.load(f)

print(f"Model loaded: input={model.input_shape}, output={model.output_shape}")
print(f"Label mapping: {len(gloss_dict)} of 271 output indices have labels")

# MediaPipe setup
mp_holistic = mp.solutions.holistic

# Extract landmarks function matching the original preprocessing
def extract_landmarks(results):
    """
    Extract pose (filtered), left/right hands, and selected face landmarks
    from MediaPipe results, returning x,y coordinates.
    Matches the original extract_landmarks from live_translation.py.
    """
    important_face_landmarks = [
        33, 133, 159, 145, 153, 144,          # Right eye
        362, 263, 386, 374, 380, 373,         # Left eye
        70, 63, 105, 66, 107,                 # Right eyebrow
        295, 282, 320, 285, 318,              # Left eyebrow
        1, 168, 197, 4,                       # Nose bridge and tip
        78, 308, 13, 14, 81, 311              # Mouth and lips
    ]

    # Indices to remove from pose landmarks
    indices_to_remove1 = np.arange(23, 33)
    indices_to_remove2 = np.arange(0, 11)
    pose_indices_to_remove = set(indices_to_remove1.tolist() + indices_to_remove2.tolist())
    pose_indices_to_keep = [i for i in range(33) if i not in pose_indices_to_remove]

    # Pose landmarks (x, y only)
    if results.pose_landmarks:
        pose = [(lm.x, lm.y) 
                for i, lm in enumerate(results.pose_landmarks.landmark) 
                if i in pose_indices_to_keep]
    else:
        pose = [(0, 0)] * len(pose_indices_to_keep)

    # Face landmarks (x, y only)
    if results.face_landmarks:
        face_all = results.face_landmarks.landmark
        face = [(face_all[i].x, face_all[i].y) for i in important_face_landmarks]
    else:
        face = [(0, 0)] * len(important_face_landmarks)

    # Left hand (x, y only)
    if results.left_hand_landmarks:
        left_hand = [(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark]
    else:
        left_hand = [(0, 0)] * 21

    # Right hand (x, y only)
    if results.right_hand_landmarks:
        right_hand = [(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark]
    else:
        right_hand = [(0, 0)] * 21

    # Combine all: pose + left_hand + right_hand + face
    # Total: 12 + 21 + 21 + 32 = 86 landmarks * 2 coords = 172 features
    combined = np.array(pose + left_hand + right_hand + face)
    return combined

# Pad video function matching preprocessing_split.py pad_video
def pad_video(X):
    padded_length = 150
    data_array = np.zeros((padded_length, X.shape[1], X.shape[2]))
    for landmark in range(0, X.shape[1]):
        for coordinate in range(0, X.shape[2]):
            data_array[:, landmark, coordinate] = np.tile(X[:, landmark, coordinate], int(padded_length / X.shape[0] + 2))[:padded_length]
    return data_array

# Process a single frame
def process_frame(frame, holistic):
    """Process a webcam frame: flip, convert to RGB, run holistic, extract landmarks."""
    frame = cv2.flip(frame, 1)  # Mirror the frame
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    numpy_data = extract_landmarks(results)
    return numpy_data

# Initialize holistic
with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.9
) as holistic:

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        exit(1)

    print("Webcam opened successfully. Collecting 150 frames...")
    print("Please show your hand to the camera.")
    print()

    landmark_sequences = []
    frame_count = 0

    # Collect 150 consecutive frames
    while cap.isOpened() and frame_count < 150:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame")
            break

        # Process frame and extract landmarks
        numpy_data = normalize_frame(process_frame(frame, holistic))

        # Store the frame's landmarks
        landmark_sequences.append(numpy_data.astype(np.float32))
        frame_count += 1

        # Progress indicator
        if frame_count % 30 == 0:
            print(f"Collected {frame_count}/150 frames...")

    cap.release()
    cv2.destroyAllWindows()

    if not landmark_sequences:
        raise SystemExit("No webcam frames were captured; inference was not run.")

    print(f"\nCollected {len(landmark_sequences)} frames total")
    if len(landmark_sequences) < 150:
        print(f"Warning: Only {len(landmark_sequences)} frames collected (expected 150)")

    # Stack all frames into array
    data = np.array(landmark_sequences)  # shape: (n_frames, 86, 2)

    # Reshape/truncate/pad to (150, 172) matching model input
    if len(data) == 150:
        # Reshape from (150, 86, 2) to (150, 172)
        data_flat = data.reshape(150, -1)  # flatten each frame: 86*2 = 172
    elif len(data) > 150:
        # Truncate to first 150 frames, then flatten
        data = data[:150]
        data_flat = data.reshape(150, -1)
    else:
        # Match training-time pad_video by repeating the whole clip sequence.
        data = np.tile(data, (150 // len(data) + 2, 1, 1))[:150]
        data_flat = data.reshape(150, -1)

    print(f"Feature array shape: {data_flat.shape} (expected: (150, 172))")

    # Add batch dimension: (1, 150, 172)
    model_input = np.expand_dims(data_flat, axis=0)
    print(f"Model input shape: {model_input.shape}")

    # Run inference
    print("Running model inference...")
    predictions = model.predict(model_input, verbose=0)
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[0][predicted_index])

    print(f"\nPredicted output index: {predicted_index}")
    print(f"Confidence: {confidence:.4f}")

    # Map to label using the JSON mapping
    if str(predicted_index) in gloss_dict:
        label = gloss_dict[str(predicted_index)]
        print(f"Predicted label: {label}")
    else:
        # The JSON maps 209 of 271 outputs (indices 0-208)
        # Indices 209-270 have no mapping in the provided JSON
        label = f"UNMAPPED - index {predicted_index} has no label in index_to_gloss_200.json"
        print(label)
        print(f"Note: Model has 271 outputs but only 209 labels are mapped (0-208)")
        print(f"Missing mappings: indices 209-270 (62 total classes)")

    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
PYEOF

echo ""
echo "============================================================="
echo "End of pipeline"
echo "============================================================="
