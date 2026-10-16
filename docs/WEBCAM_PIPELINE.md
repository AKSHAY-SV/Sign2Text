# Webcam crop experiment and classifier boundaries

Run the isolated candidate demo from the repository root:

```bash
python3 clean_crop_experiment.py --diagnostic
```

It mirrors the camera frame before both hand detection and crop extraction, keeps
landmark drawing on the preview only, and displays the exact resized RGB tensor
sent to `models/candidate_asl_letters_v2/best.keras`. Press `S` to save that
64x64 RGB input and its top-three scores locally. Diagnostic samples default to
`/tmp/sign2text_diagnostic_crops`.

The candidate CNN consumes RGB photographs, normalized to float32 `[0, 1]`, with
class order defined by its adjacent metadata. Do not pass grayscale threshold
masks or landmark renderings to this model.

## Landmark-only alternative

`punpuniacitizen/MediaPipe-ASL-sign-language-recognition` uses a shared
landmark-normalization and skeleton-rendering path (`preprocessing.py`) for its
own 96x96 input and its own ONNX model/class labels. Sharing the renderer between
its training and inference paths is a useful design property. Its model and
representation are not compatible with this project's 64x64 RGB-photo CNN.
Treat it as a separate future classifier option: it would need its own matching
training data, model, class mapping, and inference adapter before it could be
selected here.

The webcam experiment currently uses the detector's 21 image landmarks and a
2.5x square expansion. The installed MediaPipe Tasks result does not expose a
hand bounding box, so review saved A/B/V/F crops for wrist coverage and hand
scale before drawing conclusions about classifier generalization.
