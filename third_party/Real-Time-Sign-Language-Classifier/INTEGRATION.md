# Integrated recognition engine

Source: `Umaima-Manzoor/Real-Time-Sign-Language-Classifier`, checkout commit
`8c71df1226cf8470f951aa8e1cd6381812036d33`.

The original `scripts/recognition.py` is retained here for provenance. The
single-window Sign2Text loop calls the included checkpoint through
`src/reference_recognizer.py`, using the same 300x300 white canvas, aspect
preservation, 128x128 resize, BGR-to-RGB conversion, and float32 `[0,1]`
normalization specified by the reference recognition and training code.
Hand crop bounds are supplied by Sign2Text's existing MediaPipe Tasks detector
so the reference script's independent camera window and CVZone detector are
not launched.

Integrated assets (copied byte-for-byte from the reference checkout):

- `models/reference_rt_sign_classifier/best_model.h5`
- `models/reference_rt_sign_classifier/class_mapping.pkl`

The mapping is ordered `A` through `Z`, then digits `0` through `9`; the loaded
model accepts `(128, 128, 3)` and returns 36 scores. The reference repository's
MIT `LICENSE` is included in this directory. It permits use, modification, and
distribution subject to retaining the copyright and permission notice. The
reference README describes its dataset as collected specifically for the
project and does not identify a third-party dataset dependency. No reference
dataset or optional text-to-speech dependency was copied.

The original Sign2Text candidate model and metadata remain available with
`python3 main.py --candidate-fallback`.
