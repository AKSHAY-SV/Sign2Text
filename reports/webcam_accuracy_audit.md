# Webcam Accuracy Audit

Date: 2026-10-11

## Findings ranked by likely impact

1. **Training/live image domain mismatch remains the largest accuracy risk.** The candidate is trained and validated on centered dataset photos. The live system sends a landmark-derived crop with background/context. The metadata itself notes that this needs camera validation. No labeled webcam set exists in the repository, so the reported 99.23% dataset validation accuracy is not a webcam accuracy estimate.
2. **The prior live classifier selection was ambiguous.** `main.py` instantiated a Random Forest classifier and separately loaded a CNN candidate; the CNN path was fixed in the loop, but the selection was implicit and the classical classifier object was misleadingly passed into the predictor. The app now loads a named CNN directory and matching metadata: `asl_letters_v2` by default and `candidate_asl_letters_v2` with `--candidate`.
3. **Class index mapping was not fully validated.** The CNN loader checked model output count but accepted a metadata `class_to_idx` mapping inconsistent with `class_names`; it could silently attach wrong labels to output indices. It now validates the two mappings and requires sibling metadata for an explicitly selected model.
4. **Live preprocessing differed from training.** Training uses TensorFlow bilinear resize with antialiasing; inference used OpenCV `INTER_AREA`. Inference now uses the training resize operation, then RGB and float32 `/255` normalization.
5. **Crop robustness/composition can hurt recognition.** The crop uses the first detected hand, a square box 1.6 times the landmark bounding extent, clipped to frame boundaries. It has no padding if fingers approach the image edge and may still clip fingertips because MediaPipe landmarks do not cover the full silhouette. Invalid or empty detections now return no crop safely. Left and right hands use the same geometry. No hand-specific mirroring is added.
6. **Temporal logic affects display delay, not raw classification.** The confidence threshold is 0.6 and five consecutive eligible observations are needed. CNN inference runs every three frames and cached outputs are reused between inferences. At 30 camera FPS, five observations can take around 0.5 seconds or longer, depending on inference cadence. The confidence is a softmax maximum and is not calibrated.

## Camera and mirroring trace

The camera returns BGR. With current `MIRROR_FEED = False`, the captured frame is not flipped. If enabled, `main.py` flips once before detection, crop, and display. MediaPipe receives one BGR-to-RGB conversion internally for detection. The crop continues to use the same already-flipped frame and normalized landmark coordinates, so coordinate alignment is preserved. CNN preprocessing separately converts the cropped BGR pixels to RGB; this is channel conversion, not a spatial flip. MediaPipe handedness labels are not used for the CNN. The candidate training pipeline applies random horizontal flips, which supports both orientations but does not demonstrate equal live accuracy.

The CNN path crops `detection.hands[0]`; it does not use the full frame. When no CNN loads, the main loop disables recognition; it does not fall back to the instantiated Random Forest.

## Files changed

- `src/model/cnn_classifier.py`: validate model/metadata dimensions and index mappings, pair explicitly selected checkpoints with sibling metadata, use training-compatible resize, validate probability output.
- `src/prediction/predictor.py`: reject invalid landmark crop inputs and retain the last CNN input scores for diagnostics.
- `main.py`: explicitly select the established model directory or candidate via CLI; add optional crop/top-three diagnostic window and print exact paths.
- `tests/test_webcam_pipeline.py`: lightweight contract tests for candidate loading, preprocessing, mirroring geometry, invalid crops, metadata order, and stabilization.

The untracked candidate checkpoint, its metadata, backups, and the pre-existing `test_candidate_webcam.py` were preserved.

## Verification results

- `python3 -m py_compile main.py src/model/cnn_classifier.py src/prediction/predictor.py tests/test_webcam_pipeline.py`: passed.
- Standalone assertions: passed for candidate loading, exact candidate metadata pairing, input `(None, 64, 64, 3)`, output `(None, 29)`, class order, empty/invalid crop handling, mirrored-coordinate crop dimensions, confidence thresholding, and stable-frame logic.
- `pytest` suite: not executed because `pytest` is not installed in the active Python environment.
- Preprocessing RGB pixel check: passed for a constant red BGR image; output was RGB `[1.0, 0.0, 0.0]` with float32 shape `(64, 64, 3)` and range `[0,1]`.
- A TensorFlow model `predict` latency benchmark was attempted but hung in inference after successful model loading and was stopped. No latency/FPS result is claimed. CUDA was unavailable; webcam hardware was not exercised.
- No suitable labeled webcam images were present, so webcam accuracy and left/right B accuracy remain unknown.

Actual candidate paths:

- Model: `models/candidate_asl_letters_v2/best.keras`
- Metadata: `models/candidate_asl_letters_v2/metadata.json`
- Input: RGB `(64,64,3)`, float32, bilinear antialiased resize, values divided by 255 into `[0,1]`.

## Run and revert

Run the candidate with diagnostics:

```bash
python3 main.py --candidate --diagnostic
```

Run the existing separate trained model directory without diagnostic display:

```bash
python3 main.py
```

Revert the experimental code changes by reverting `main.py`, `src/model/cnn_classifier.py`, and `src/prediction/predictor.py`, and removing the newly added `tests/test_webcam_pipeline.py` and this report. The candidate model files were not changed.

## Recommended next experiment

Do not retrain on the same dataset yet. First record labeled webcam examples for all static classes under both hands, multiple distances/backgrounds, and the exact runtime crop pipeline. Split by recording session/person rather than random adjacent frames to avoid leakage. Compare candidate predictions on the saved crops, inspect per-class precision/recall and confusion pairs (including B/N), and report latency using a warmed-up inference benchmark. Keep these camera captures outside Git. Only train a new model if that independent evaluation shows a repeatable camera-domain failure; if justified, add a camera-crop fine-tuning set with a held-out session test split and a separate output directory.

## Follow-up: no-hand and stale-crop safeguards (2026-10-11)

The previous main loop already obtains a fresh `HandDetection` on each frame, but reused cached predictions between CNN calls and rendered `Sign: --` when no hand was detected. The crop routine had no exact landmark-count or normalized-coordinate validation; malformed coordinates could be clipped to an unrelated visible part of the image. There is no face/full-frame crop fallback. The predictor now clears current state and crop buffers on missing/invalid hands, requires the detection dimensions to match the current frame, rejects non-increasing detector timestamps, requires 21 finite in-frame points, and validates positive crop dimensions. The main display says `NO HAND DETECTED` and diagnostic mode shows `NO VALID HAND CROP`; for valid inference it displays the exact 64x64 RGB tensor passed to the CNN with top three scores. `MIRROR_FEED` remains unchanged and detection/crop/display all use the same frame.

Follow-up verification: syntax compilation and `git diff --check` are run for the updated files. `pytest` remains unavailable in the environment. A focused TensorFlow-backed runtime check was attempted but stalled during TensorFlow startup and was stopped; webcam behavior was not tested.

## Follow-up: V-sign stability investigation (2026-10-11)

Code review found that the main loop displayed only the current argmax label and re-used it between model calls; the existing stabilizer counted consecutive labels and did not use the other class scores. The pipeline now averages two consecutive full probability vectors, keeps the existing 0.6 minimum confidence, and requires two consistent smoothed observations before showing a label. It displays `UNCERTAIN` while class evidence is weak or changing. These settings are in `config/settings.py`; at the configured inference cadence this adds roughly one to several inference intervals depending on how close competing class scores are. A hand disappearance still immediately clears smoothing and accepted labels.

Crop geometry inspection also found that square landmark boxes were clipped independently at image edges, potentially turning into rectangles that were then stretched to 64x64. The crop now keeps a fixed square extent and uses replicated edge pixels only for the portion outside the camera frame. This avoids aspect distortion while retaining all available image pixels; it cannot recover fingers that are physically outside the camera view.

Diagnostic mode now reports raw top-three classes/scores, stable label or uncertain state, crop validity/dimensions, and CNN tensor shape/dtype at most once per second. Press `S` to save the current valid 64x64 CNN input crop plus a JSON score record under `/tmp/sign2text_webcam_samples`; capture is opt-in and no full frame is saved. This supports manual side-by-side inspection of V and the N/W/H confusions. No webcam recordings or labeled examples were available during this audit, so no V accuracy improvement is claimed and the model was not retrained.

Focused checks executed from the source passed for smoothing/consensus, low confidence and malformed probability vectors, valid and invalid square crops (including frame-edge padding), and display dtype conversion. `python3 -m py_compile` and `git diff --check` passed. The full pytest suite could not run because pytest is not installed; actual webcam accuracy remains unmeasured.
