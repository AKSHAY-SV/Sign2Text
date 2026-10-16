# Reproducible ASL alphabet training on Kaggle

This trains static alphabet/fingerspelling classes from the 29 folders `A`–`Z`,
`del`, `nothing`, and `space`. It does not train a sentence translator. J and Z
are motion signs, so a single image cannot reliably represent their movement.

## Kaggle setup

1. Create a Kaggle notebook, enable a GPU accelerator, and attach the ASL
   Alphabet dataset whose root contains the 29 class directories.
2. Clone this repository into the notebook (with Kaggle Internet enabled), or
   upload `train_cnn.py`, `src/model/image_preprocessing.py`, `config/`, and
   `src/` as notebook data.
3. Kaggle normally provides TensorFlow, NumPy, OpenCV, and scikit-learn. If a
   dependency is missing, run `pip install -r requirements-kaggle.txt`.
4. Run the training command below. It discovers the mounted dataset directory,
   audits all classes, groups perceptual duplicates before a deterministic
   stratified split, then trains with early stopping and best-validation-loss
   checkpointing. The log prints the script path, Git revision, branch, dirty
   state, and TensorFlow/scikit-learn versions so a Kaggle run can be checked
   against the intended checkout.

```bash
git clone https://github.com/AKSHY-SV/Sign2Text.git /kaggle/working/Sign2Text
cd /kaggle/working/Sign2Text
DATASET_DIR="$(find /kaggle/input -type d -name asl_alphabet_train -print -quit)"
test -n "$DATASET_DIR" || { echo 'Dataset directory not found under /kaggle/input'; exit 1; }
python train_cnn.py \
  --dataset "$DATASET_DIR" \
  --output-dir /kaggle/working/asl_webcam_candidate \
  --image-size 128 \
  --batch-size 64 \
  --epochs 30 \
  --validation-split 0.2 \
  --seed 42 \
  --patience 5
```

For a bounded pipeline smoke test, use a balanced subset and a fixed epoch
budget:

```bash
python train_cnn.py \
  --dataset "$DATASET_DIR" \
  --output-dir /kaggle/working/asl_cnn_smoke \
  --image-size 64 \
  --batch-size 32 \
  --smoke \
  --smoke-per-class 100 \
  --smoke-epochs 8
```

Smoke mode uses BatchNormalization momentum 0.7 and runs the requested epochs
without early stopping or learning-rate reduction. It still saves the best
validation-loss checkpoint and fails if its predictions collapse, do not beat
the majority-class accuracy, or its loss does not beat the uniform-class
baseline. The output includes per-class split counts, prediction diagnostics,
and `smoke_train_validation_grid.jpg` for checking actual images against their
folder labels. This smoke gate checks that the training path learns; it does
not predict full-dataset or webcam accuracy.

The output directory contains `best.keras`, its exact `metadata.json`,
`classification_report.json`, `confusion_matrix.csv`, and dataset audit files.
To download them, use Kaggle's notebook **Output** section after **Save
Version**, or zip the output directory in `/kaggle/working` and download that
artifact. Keep `best.keras` beside `metadata.json` when installing locally.
Alternatively, create one downloadable archive with
`cd /kaggle/working && zip -r asl_webcam_candidate.zip asl_webcam_candidate`.

## Local installation and webcam validation

Copy the Kaggle output into `models/asl_webcam_candidate/`. The output directory
is gitignored so checkpoints and generated metadata are not accidentally
committed. Existing checkpoints and metadata remain in place.

Run the new model through the one-window webcam app:

```bash
python3 main.py --model-dir models/asl_webcam_candidate --diagnostic
```

The `--diagnostic` view shows the source preview, clean hand crop, exact resized
RGB input, and top-three predictions in one window. It uses the same
`preprocess_rgb_tensor` implementation as training. Compare live performance
with separate, locally saved labeled hand crops; validation on the source
dataset does not establish webcam accuracy.

To evaluate a labeled webcam-crop directory arranged as
`crops/A/*.png`, `crops/B/*.png`, etc., run:

```bash
python3 scripts/evaluate_webcam_crops.py \
  --crops /path/to/labeled_webcam_crops \
  --model-dir models/asl_webcam_candidate \
  --output-dir reports/asl_webcam_candidate_eval
```

Only use hand crops in that evaluator; it does not capture or upload webcam
images. Review the resulting per-class report and confusion matrix before
promoting any model.
