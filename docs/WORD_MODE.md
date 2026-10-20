# Isolated ASL Word Mode

This mode loads the pretrained LSTM and its own `actions.npy`, `mean.npy`, and
`std.npy` from `/home/akshay-sv/asl-recognition-lstm`. It is independent of the
alphabet ONNX model and the existing GRU gloss recognizer. It uses a separate
process and owns the webcam while running.

## Verified model and classes

The checkpoint loaded read-only as a Keras model with input `(None, 40, 444)` and
output `(None, 24)`. The serialized `actions.npy` array provides the authoritative
class order:

`bath, bye, dad, drink, food, happy, hello, hungry, mad, milk, mom, no, please,
sad, see, sick, sleep, thankyou, thirsty, wait, water, where, yes, yourself`.

`hello` is output index 6. `world` is not present in the checkpoint labels. These
are isolated sign classes; the model does not perform sentence translation.

The Keras architecture loaded from `best.keras` contains 128-unit and 64-unit LSTM
layers followed by 128-unit and 64-unit dense layers and a 24-unit softmax output.
The confidence shown is the model's maximum softmax score, not a correctness
guarantee. The checked-in evaluation report records 71.23% accuracy and an
`hello` F1 of 0.815 on its test split; that is source-repository evaluation, not a
claim about this laptop's webcam performance.

## Preprocessing contract

The adapter reuses the source app module's `extract_keypoints()` and
`normalize_sequence()` functions, avoiding a second implementation of the
preprocessing. Each frame has 444 float32 values in this order:

1. Pose: 33 landmarks × x/y/z = 99
2. Left hand: 21 × x/y/z = 63
3. Right hand: 21 × x/y/z = 63
4. Selected face: 73 × x/y/z = 219

Missing pose, either hand, or selected face values are zero-filled, and non-finite
features become zero as in the source app. The model's `mean.npy` and `std.npy`
are used for per-feature normalization. Inference uses a rolling window of exactly
40 frame vectors and calls the model with `(1, 40, 444)`. No shorter sequence is
passed to the checkpoint. The selected face features remain in the model input;
the preview draws pose and hands without drawing face points.

Training samples with 40 or more frames were uniformly resampled to 40 points;
shorter training clips repeated their final frame. The source live app instead
classifies its latest 40 consecutive webcam frames. This adapter follows that live
sequence behavior and keeps its tensor shape and feature preprocessing unchanged.

## Dependencies and environments

No dependency manifest is present in the word repository. Its README's install
list includes TensorFlow, OpenCV, MediaPipe, and NumPy along with training/reporting
packages. For inference, TensorFlow, NumPy, OpenCV, and MediaPipe Tasks are used;
Pandas, scikit-learn, Matplotlib, and tqdm are training or reporting dependencies.

The existing `.venv-repair` is Python 3.12.3 with TensorFlow 2.19.0, MediaPipe
0.10.21, NumPy 1.26.4, and OpenCV contrib 4.11.0.86. `pip check` reported no
broken requirements. A read-only import loaded the source app, model, labels, and
statistics, and the local `holistic_landmarker.task` detector initialized and
closed successfully. The `.venv-alphabet` remains separate and unchanged. No
packages were installed and `.venv-words` was not created because the existing
`.venv-repair` passed these checks.

If you later want a dedicated environment, do not run these commands until you
approve the setup plan:

```bash
cd /home/akshay-sv/live-asl-translation
python3.12 -m venv .venv-words
.venv-words/bin/python -m pip install 'tensorflow==2.19.0' 'mediapipe==0.10.21' 'numpy==1.26.4' 'opencv-python==4.11.0.86'
.venv-words/bin/python -m pip check
```

Those commands affect only `.venv-words`; they do not share packages with either
existing environment. The current launcher uses `.venv-words` if present, otherwise
the verified `.venv-repair`.

## Run the word demo

```bash
cd /home/akshay-sv/live-asl-translation
./run_sign2text_words.sh
```

The app waits for 40 frames, predicts every frame (matching the source live app),
and updates the displayed word after eight consecutive predictions at or above
0.60 confidence. Adjust with
`--prediction-interval`, `--stable-updates`, and `--confidence-threshold`. Hold the
isolated sign `hello` to test the intended milestone. Press `C` to clear and `Q`
to exit. The displayed field contains one stable isolated class; it does not join
separate predictions into a sentence.

The task model is loaded only from the checked-in `model/holistic_landmarker.task`;
this launcher does not download assets. Face points are retained as inference
features but are not drawn. Camera, task detector, model references, and GUI are
released on exit.

The source repository includes an MIT license and identifies Google ASL Signs as
its dataset; the raw dataset is not included. Review the dataset's terms before
redistributing any artifacts beyond this local integration.

## Relationship to the other modes

Alphabet mode remains `./run_sign2text_alphabet.sh`; the original GRU word/gloss
mode remains `./run_sign2text.sh`. Keep them as separate processes and do not run
two webcam modes simultaneously. Only expose an eventual mode selector after each
launcher is independently verified; do not merge model inputs or class maps.
