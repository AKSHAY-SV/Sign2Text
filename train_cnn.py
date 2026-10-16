"""Reproducible CNN training for static ASL alphabet images.

The output directory is explicit so experiments cannot silently replace the
currently deployed model. Images are RGB, resized to a square, and scaled to
[0, 1].
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedGroupKFold

from config.settings import DATASET_DIR, MODELS_DIR
from src.model.image_preprocessing import NORMALIZATION, RESIZE_METHOD, preprocess_rgb_tensor

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPECIAL = ["del", "nothing", "space"]
EXPECTED_CLASSES = LETTERS + SPECIAL
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass


def audit_dataset(root: Path, cache_path: Path | None = None) -> tuple[list[str], dict[str, list[Path]], dict[str, int], dict[str, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    classes = [p.name for p in class_dirs]
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Expected classes {EXPECTED_CLASSES}; found {classes}")
    files: dict[str, list[Path]] = {}
    invalid: dict[str, int] = {}
    dimensions: dict[str, int] = {}
    import cv2
    fingerprint = []
    files_by_dir = {}
    for directory in class_dirs:
        items = sorted(p for p in directory.iterdir()
                       if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        files_by_dir[directory.name] = items
        fingerprint.append([directory.name, [[p.name, p.stat().st_size, p.stat().st_mtime_ns]
                                             for p in items]])
    if cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("fingerprint") == fingerprint:
                cached_files = {name: [Path(p) for p in paths] for name, paths in cached["files"].items()}
                if all(all(p.is_file() for p in paths) for paths in cached_files.values()):
                    if "dimensions" in cached:
                        return classes, cached_files, cached["invalid"], cached["dimensions"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    for directory in class_dirs:
        good, bad = [], 0
        for path in files_by_dir[directory.name]:
            try:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("not a decodable color image")
                shape = f"{image.shape[1]}x{image.shape[0]}"
                dimensions[shape] = dimensions.get(shape, 0) + 1
                good.append(path)
            except (OSError, ValueError):
                bad += 1
        files[directory.name] = good
        invalid[directory.name] = bad
        if len(good) < 100:
            raise ValueError(f"Class {directory.name} has only {len(good)} valid images")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"fingerprint": fingerprint,
                                          "files": {k: [str(p) for p in v] for k, v in files.items()},
                                          "invalid": invalid, "dimensions": dimensions}))
    return classes, files, invalid, dimensions


def split_files(files: dict[str, list[Path]], validation_split: float, seed: int,
                max_per_class: int | None = None):
    """Deterministically stratify while keeping perceptual duplicate groups intact."""
    if not 0.05 <= validation_split <= 0.5:
        raise ValueError("validation_split must be between 0.05 and 0.5")
    classes = list(files)
    paths, labels = [], []
    rng = random.Random(seed)
    for class_index, name in enumerate(classes):
        choices = list(files[name])
        rng.shuffle(choices)
        if max_per_class is not None:
            choices = choices[:max_per_class]
        paths.extend(choices)
        labels.extend([class_index] * len(choices))
    paths = np.asarray([str(p) for p in paths])
    labels = np.asarray(labels, dtype=np.int32)
    groups, duplicate_stats = perceptual_duplicate_groups(paths, seed)
    n_splits = max(2, min(20, int(round(1.0 / validation_split))))
    unique_groups = np.unique(groups).size
    if unique_groups < n_splits:
        raise ValueError(
            f"Cannot make a {n_splits}-fold duplicate-aware split from "
            f"only {unique_groups} unique image groups; add more distinct "
            "images per class or reduce validation_split."
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_indices, val_indices = next(splitter.split(paths, labels, groups))
    return (paths[train_indices], labels[train_indices], paths[val_indices],
            labels[val_indices], classes, duplicate_stats)


def perceptual_duplicate_groups(paths: np.ndarray, seed: int = SEED,
                                 max_hamming_distance: int = 2):
    """Group exact/near duplicate images using DCT perceptual hashes.

    Three disjoint hash bands guarantee that hashes at Hamming distance <=2
    share at least one bucket. Candidate pairs are then checked exactly. This
    avoids the quadratic all-pairs comparison on Kaggle-sized datasets.
    """
    import cv2

    n = len(paths)
    parent = list(range(n))
    rank = [0] * n

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    buckets: dict[tuple[int, int], list[tuple[int, int]]] = {}
    hashes = []
    invalid = 0
    # 21/21/22-bit partition: distance <=2 must leave one complete band equal.
    bands = ((0, 21), (21, 21), (42, 22))
    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            invalid += 1
            value = index  # shouldn't occur after audit; never group bad paths
            hashes.append(value)
            continue
        small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)[:8, :8].reshape(-1)
        median = float(np.median(dct[1:]))
        value = 0
        for bit, coefficient in enumerate(dct):
            if coefficient > median:
                value |= 1 << bit
        hashes.append(value)
        candidates: set[int] = set()
        for band_index, (offset, width) in enumerate(bands):
            key_bits = (value >> offset) & ((1 << width) - 1)
            candidates.update(other for other, _ in buckets.get((band_index, key_bits), ()))
        for other in candidates:
            if (value ^ hashes[other]).bit_count() <= max_hamming_distance:
                union(index, other)
        for band_index, (offset, width) in enumerate(bands):
            key_bits = (value >> offset) & ((1 << width) - 1)
            buckets.setdefault((band_index, key_bits), []).append((index, value))
    roots = [find(i) for i in range(n)]
    root_to_group = {root: group for group, root in enumerate(sorted(set(roots)))}
    group_ids = np.asarray([root_to_group[root] for root in roots], dtype=np.int64)
    _, counts = np.unique(group_ids, return_counts=True)
    stats = {
        "algorithm": "64-bit DCT pHash, Hamming distance <= 2, stratified group split",
        "seed": int(seed), "near_duplicate_groups": int(np.count_nonzero(counts > 1)),
        "images_in_duplicate_groups": int(counts[counts > 1].sum()) if np.any(counts > 1) else 0,
        "unique_groups": int(len(counts)), "invalid_hash_inputs": int(invalid),
    }
    return group_ids, stats


def make_dataset(paths, labels, image_size: int, batch_size: int, training: bool,
                 seed: int):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = preprocess_rgb_tensor(image, image_size)
        return image, label

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return ds.map(load, num_parallel_calls=2).batch(batch_size).prefetch(1).with_options(options)


def validate_training_labels(train_paths, train_labels, val_paths, val_labels,
                             class_names: list[str]) -> None:
    """Fail before fitting if split labels no longer match source class folders."""
    if len(train_paths) != len(train_labels) or len(val_paths) != len(val_labels):
        raise ValueError("Image path and label counts differ in the train/validation split.")
    num_classes = len(class_names)
    for side, paths, labels in (("training", train_paths, train_labels),
                                ("validation", val_paths, val_labels)):
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError(f"{side} split labels must be a non-empty 1-D array.")
        if np.any(labels < 0) or np.any(labels >= num_classes):
            raise ValueError(f"{side} split contains labels outside the class mapping.")
        mismatches = [(str(path), int(label), class_names[int(label)])
                      for path, label in zip(paths, labels)
                      if Path(path).parent.name != class_names[int(label)]]
        if mismatches:
            path, label, expected = mismatches[0]
            raise ValueError(
                f"{side} label mismatch: {path} is class {Path(path).parent.name!r}, "
                f"but label {label} maps to {expected!r}."
            )
        present = np.unique(labels)
        if len(present) != num_classes:
            raise ValueError(
                f"{side} split contains {len(present)}/{num_classes} classes; "
                "each class must appear in both splits."
            )


def check_smoke_acceptance(scores: np.ndarray, labels: np.ndarray,
                           num_classes: int, validation_loss: float) -> dict[str, Any]:
    """Check that a smoke checkpoint beats a uniform, majority-class guess."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim != 2 or scores.shape != (len(labels), num_classes):
        raise ValueError(
            f"Smoke prediction shape {scores.shape} does not match "
            f"({len(labels)}, {num_classes})."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Smoke predictions contain non-finite values.")
    if len(labels) == 0 or np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError("Smoke labels are empty or outside the configured class range.")
    predictions = scores.argmax(axis=1)
    predicted_classes = np.unique(predictions)
    accuracy = float(np.mean(predictions == labels))
    majority_baseline = float(np.bincount(labels.astype(np.int64), minlength=num_classes).max()
                              / len(labels))
    uniform_loss = float(np.log(num_classes))
    failures = []
    if len(predicted_classes) < 2:
        predicted_index = int(predicted_classes[0]) if len(predicted_classes) else -1
        failures.append(f"predictions collapsed to class index {predicted_index}")
    if accuracy <= majority_baseline:
        failures.append(f"validation accuracy {accuracy:.4f} did not beat majority baseline "
                        f"{majority_baseline:.4f}")
    if not np.isfinite(validation_loss) or validation_loss >= uniform_loss - 0.05:
        failures.append(f"validation loss {validation_loss:.4f} did not beat uniform-class "
                        f"loss {uniform_loss:.4f} by at least 0.05")
    return {
        "accepted": not failures,
        "failures": failures,
        "predicted_classes": predicted_classes.tolist(),
        "accuracy": accuracy,
        "majority_baseline_accuracy": majority_baseline,
        "validation_loss": float(validation_loss),
        "uniform_baseline_loss": uniform_loss,
    }


def class_counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=len(class_names))
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


def make_callbacks(best_path: Path, smoke: bool, patience: int):
    """Smoke mode has a fixed budget; full training retains adaptive stopping."""
    callbacks = [keras.callbacks.ModelCheckpoint(
        str(best_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1
    )]
    if not smoke:
        callbacks.extend([
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                          restore_best_weights=True, mode="min", verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                              min_lr=1e-5, verbose=1),
        ])
    return callbacks


def save_smoke_sample_grid(train_paths, train_labels, val_paths, val_labels,
                           class_names: list[str], output_path: Path) -> None:
    """Save two actual train and validation images per class with source labels."""
    import cv2

    cell, label_height = 128, 24
    canvas = np.full((len(class_names) * (cell + label_height), cell * 4, 3), 255, np.uint8)
    for class_index, class_name in enumerate(class_names):
        train_samples = [p for p, y in zip(train_paths, train_labels)
                         if int(y) == class_index][:2]
        val_samples = [p for p, y in zip(val_paths, val_labels)
                       if int(y) == class_index][:2]
        for column, (source, path) in enumerate(
                [("train", p) for p in train_samples] + [("validation", p) for p in val_samples]):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (cell, cell), interpolation=cv2.INTER_AREA)
            row_top = class_index * (cell + label_height)
            canvas[row_top:row_top + cell, column * cell:(column + 1) * cell] = image
            cv2.putText(canvas, f"{class_name} {source}", (column * cell + 3, row_top + cell + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write smoke sample grid to {output_path}")


def print_prediction_diagnostics(name: str, scores: np.ndarray, labels: np.ndarray,
                                 class_names: list[str]) -> None:
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    predictions = scores.argmax(axis=1)
    distribution = class_counts(predictions, class_names)
    print(f"{name} prediction distribution: "
          f"{ {label: count for label, count in distribution.items() if count} }")
    print(f"{name} sample accuracy: {np.mean(predictions == labels):.4f}")
    per_class = {
        class_names[index]: float(np.mean(predictions[labels == index] == index))
        for index in range(len(class_names)) if np.any(labels == index)
    }
    print(f"{name} per-class accuracy: {per_class}")
    example_rows = []
    for class_index in range(len(class_names)):
        matches = np.flatnonzero(labels == class_index)
        if len(matches):
            example_rows.append(int(matches[0]))
        if len(example_rows) == 3:
            break
    for row in example_rows:
        top = np.argsort(scores[row])[-3:][::-1]
        summary = [(class_names[int(index)], round(float(scores[row, index]), 4))
                   for index in top]
        print(f"{name} example {row}: true={class_names[int(labels[row])]}, top3={summary}")


def build_model(image_size: int, num_classes: int, learning_rate: float,
                augmentation: bool = True, batch_norm_momentum: float = 0.7):
    inputs = keras.Input((image_size, image_size, 3), name="rgb_image_01")
    x = inputs
    if augmentation:
        # No horizontal flips: mirrored hand orientation may alter sign appearance.
        x = layers.RandomRotation(0.04, fill_mode="reflect")(x)
        x = layers.RandomTranslation(0.04, 0.04, fill_mode="reflect")(x)
        x = layers.RandomZoom(0.06, fill_mode="reflect")(x)
        x = layers.RandomContrast(0.10)(x)
    x = layers.Conv2D(24, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization(momentum=batch_norm_momentum)(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    for filters in (32, 64, 96):
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=batch_norm_momentum)(x); x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(args) -> dict[str, Any]:
    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent
    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_revision = "unavailable (script is not in a Git checkout)"
    try:
        git_status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout
        git_dirty = bool(git_status.strip())
        git_branch = subprocess.run(
            ["git", "branch", "--show-current"], cwd=project_root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_dirty, git_branch = None, "unavailable"
    print(f"Training script: {Path(__file__).resolve()}")
    print(f"Git revision: {git_revision}; branch: {git_branch}; dirty: {git_dirty}")
    print(f"TensorFlow: {tf.__version__}; scikit-learn: {sklearn.__version__}")
    classes, files, invalid, source_dimensions = audit_dataset(
        args.dataset, args.output_dir / "dataset_audit.json")
    counts = {name: len(files[name]) for name in classes}
    print(f"Dataset: {sum(counts.values())} valid images; classes={classes}")
    print(f"Invalid image counts: {invalid}")
    max_per_class = args.smoke_per_class if args.smoke else None
    trp, trl, vap, val, class_names, duplicate_stats = split_files(
        files, args.validation_split, args.seed, max_per_class)
    validate_training_labels(trp, trl, vap, val, class_names)
    print(f"Split: {len(trp)} train, {len(vap)} validation; {duplicate_stats}")
    print(f"Train counts by class: {class_counts(trl, class_names)}")
    print(f"Validation counts by class: {class_counts(val, class_names)}")
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    train_ds = make_dataset(trp, trl, args.image_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(vap, val, args.image_size, args.batch_size, False, args.seed)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        grid_path = out / "smoke_train_validation_grid.jpg"
        save_smoke_sample_grid(trp, trl, vap, val, class_names, grid_path)
        print(f"Smoke sample grid: {grid_path}")
    best_path = out / "best.keras"
    model = build_model(args.image_size, len(class_names), args.learning_rate,
                        augmentation=not args.disable_augmentation,
                        batch_norm_momentum=args.batch_norm_momentum)
    cb = make_callbacks(best_path, args.smoke, args.patience)
    history = model.fit(train_ds, validation_data=val_ds, shuffle=False,
                        epochs=args.smoke_epochs if args.smoke else args.epochs,
                        callbacks=cb, verbose=2)
    best = keras.models.load_model(best_path, compile=False)
    best.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    scores = best.predict(val_ds, verbose=0)
    y_pred = scores.argmax(axis=1)
    val_loss, val_acc = best.evaluate(val_ds, verbose=0)
    cm = confusion_matrix(val, y_pred, labels=np.arange(len(class_names)))
    report = classification_report(val, y_pred, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    # Keep all training artifacts isolated. Caller may promote best.keras only
    # after review; existing models/sign_cnn.h5 is never overwritten here.
    cm_path = out / "confusion_matrix.csv"
    with cm_path.open("w", newline="") as matrix_file:
        writer = csv.writer(matrix_file)
        writer.writerow(["true\\pred", *class_names])
        writer.writerows([name, *row.tolist()] for name, row in zip(class_names, cm))
    report_path = out / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    smoke_check = None
    if args.smoke:
        train_probe_indices = np.concatenate([
            np.flatnonzero(trl == class_index)[:4] for class_index in range(len(class_names))
        ])
        train_probe_paths = trp[train_probe_indices]
        train_probe_labels = trl[train_probe_indices]
        train_probe_ds = make_dataset(train_probe_paths, train_probe_labels,
                                      args.image_size, args.batch_size, False, args.seed)
        train_scores = best.predict(train_probe_ds, verbose=0)
        print_prediction_diagnostics("Fixed train probe", train_scores,
                                     train_probe_labels, class_names)
        print_prediction_diagnostics("Validation", scores, val, class_names)
        smoke_check = check_smoke_acceptance(scores, val, len(class_names), val_loss)
        print("Smoke acceptance:", smoke_check)
    metadata = {
        "model_version": "2.0.0", "dataset": str(args.dataset.resolve()),
        "image_size": args.image_size, "input_shape": [args.image_size, args.image_size, 3],
        "source_image_dimensions": source_dimensions,
        "model_input_dimensions": [args.image_size, args.image_size, 3],
        "num_classes": len(class_names), "class_names": class_names,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "color_order": "RGB",
        "training_decoder_color_order": "RGB (TensorFlow decode_image)",
        "webcam_source_color_order": "BGR (OpenCV), converted to RGB before model input",
        "normalization": NORMALIZATION,
        "resize": RESIZE_METHOD,
        "batch_norm_momentum": args.batch_norm_momentum,
        "tensorflow_version": tf.__version__,
        "scikit_learn_version": sklearn.__version__,
        "augmentation": ([] if args.disable_augmentation else
                         ["rotation_0.04", "translation_0.04", "zoom_0.06", "contrast_0.10"]),
        "seed": args.seed, "validation_split": args.validation_split,
        "effective_validation_fraction": float(len(vap) / (len(trp) + len(vap))),
        "split_strategy": duplicate_stats,
        "train_samples": int(len(trp)), "validation_samples": int(len(vap)),
        "train_class_counts": class_counts(trl, class_names),
        "validation_class_counts": class_counts(val, class_names),
        "git_revision": git_revision,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "best_val_accuracy": float(val_acc), "best_val_loss": float(val_loss),
        "smoke_test": bool(args.smoke), "training_epochs_completed": len(history.history["loss"]),
        "smoke_acceptance": smoke_check,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "live_input_note": "Training uses centered dataset photos; live hand crop domain requires camera validation.",
        "dynamic_sign_limitation": "J and Z require motion and cannot be reliably recognized from one static image.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Best validation accuracy={val_acc:.4f}; loss={val_loss:.4f}")
    print("Per-class accuracy:")
    for i, name in enumerate(class_names):
        total = int(cm[i].sum()); correct = int(cm[i, i])
        print(f"  {name}: {correct}/{total} ({correct/max(total,1):.1%})")
    print(f"Saved best checkpoint: {best_path}")
    if smoke_check is not None and not smoke_check["accepted"]:
        raise RuntimeError("Smoke test failed: " + "; ".join(smoke_check["failures"]) +
                           ". Inspect the saved diagnostic grid and reports; do not promote this checkpoint.")
    return {"accuracy": float(val_acc), "loss": float(val_loss), "confusion_matrix": cm,
            "report": report, "metadata": metadata, "checkpoint": best_path}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "asl_webcam_candidate")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--batch-norm-momentum", type=float, default=0.7,
                        help="BatchNorm moving-stat momentum; lower values track short smoke runs faster")
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-class", type=int, default=40)
    parser.add_argument("--smoke-epochs", type=int, default=8)
    parser.add_argument("--disable-augmentation", action="store_true",
                        help="diagnostic only: train on unaugmented images")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
