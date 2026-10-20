# ASL Alphabet Fingerspelling Mode

This is a separate, optional alphabet recognizer. It does not load or change the
Sign2Text GRU checkpoint or its gloss map. It uses the cloned
`MediaPipe-ASL-sign-language-recognition` repository's ONNX model, `a`–`z` label
order, preprocessing, smoothing, and J/Z motion code in place. The clone and its
weights are not copied into this project.

## Verified input and outputs

The cloned training code defines a custom CNN with three double-convolution blocks
(32, 64, and 128 channels), each followed by batch normalization, ReLU, max pooling,
and dropout; global average pooling; a 128-unit dense block; then 26 raw logits. Its
export code names the main output `logits`. Inference input is a float32 batch of
RGB skeleton images with shape `(1, 96, 96, 3)` and pixel values 0–255; the graph
performs its own `/255` rescaling. A frame is transformed with the cloned
`preprocessing.py`: 21 MediaPipe x/y points → pixel coordinates → centered,
aspect-preserving 70%-fill normalization → colored RGB skeleton on a 192 canvas →
96×96 `INTER_AREA` resize. The adapter applies softmax to logits and averages the
last eight probability vectors. For J/Z (and the clone's compatible handshape
corrections), the UI identifies motion-rule resolution; the percentage remains
the model's static-pose score and is not a calibrated confidence in the motion rule.

The model has all 26 lowercase `a`–`z` classes. J and Z have additional rule-based
motion recognition in the cloned `motion.py`; digits and words are not model classes.
The camera is mirrored before MediaPipe, as in the original alphabet demo. The
preview draws hand landmarks only, not face landmarks.

## Dependencies and isolated environment

The existing project environments were inspected: NumPy 1.26.4, MediaPipe 0.10.21,
and OpenCV 4.11 are present, but ONNX Runtime is absent. The clone's requirements
specify `opencv-python`; the existing project uses `opencv-contrib-python`, and
installing both into one environment can overwrite shared `cv2` files. The clone
also explicitly expects the MediaPipe 0.10 `solutions` API. For that reason, no
packages were installed and no existing environment was changed. Set up this
dedicated environment only if you want to run live inference:

```bash
cd /home/akshay-sv/live-asl-translation
python3.12 -m venv .venv-alphabet
.venv-alphabet/bin/python -m pip install 'numpy<2' 'opencv-python==4.11.0.86' 'mediapipe==0.10.21' onnxruntime
```

These commands affect only `.venv-alphabet`. They install inference dependencies
only; TensorFlow and training dependencies are not needed. Verify dependency
resolution before using it with:

```bash
.venv-alphabet/bin/python -m pip check
```

## Launch

```bash
./run_sign2text_alphabet.sh
```

Optional controls: `--camera 1`, `--refresh-interval 2`, `--commit-frames 10`,
and `--commit-confidence 0.75`. Predictions are letter hypotheses, not validated
recognition results. The buffer commits a letter only after consecutive agreeing
inference updates above the threshold; use `C` to clear, Backspace to undo, Space
to insert a separator, and `R` to repeat the previous character. `Q` exits.

The original word/gloss mode remains available with:

```bash
./run_sign2text.sh
```

## Independent OpenCode audit

In a fresh OpenCode session, ask for a read-only audit of this integration. Have it
verify the clone's ONNX artifact and `class_names.txt`, the adapter's preprocessing
call chain and tensor contract, the independent GRU launcher/checkpoint, and the
offline test results. Ask it not to edit files, install packages, download models,
run training, or access the webcam. It should report that live inference is pending
until ONNX Runtime is installed in `.venv-alphabet` and tested locally.

## License note

The cloned repository licenses its code under MIT. Its LICENSE explicitly says that
the ONNX weights are not covered by that code license; the weights derive mainly
from an ASL Alphabet dataset identified as GPL-2.0, plus two CC0 datasets. The
cloned README recommends checking those dataset terms before redistributing the
weights. This integration references the local clone and does not copy or
redistribute the checkpoint.
