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
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
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
                           num_classes: int) -> dict[str, Any]:
    """Reject smoke runs that execute but collapse to a single class."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim != 2 or scores.shape != (len(labels), num_classes):
        raise ValueError(
            f"Smoke prediction shape {scores.shape} does not match "
            f"({len(labels)}, {num_classes})."
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("Smoke predictions contain non-finite values.")
    predictions = scores.argmax(axis=1)
    predicted_classes = np.unique(predictions)
    accuracy = float(np.mean(predictions == labels))
    if len(predicted_classes) < 2:
        raise RuntimeError(
            "Smoke test failed: the checkpoint predicts only one class on "
            f"{len(labels)} balanced validation examples (class index "
            f"{int(predicted_classes[0]) if len(predicted_classes) else 'none'}). "
            "The pipeline ran, but this is not evidence of learning. Increase "
            "smoke epochs or investigate optimization/model behavior; do not "
            "promote this checkpoint."
        )
    return {"predicted_classes": predicted_classes.tolist(), "accuracy": accuracy}


def build_model(image_size: int, num_classes: int, learning_rate: float):
    inputs = keras.Input((image_size, image_size, 3), name="rgb_image_01")
    # No horizontal flips: mirrored hand orientation may alter sign appearance.
    x = layers.RandomRotation(0.04, fill_mode="reflect")(inputs)
    x = layers.RandomTranslation(0.04, 0.04, fill_mode="reflect")(x)
    x = layers.RandomZoom(0.06, fill_mode="reflect")(x)
    x = layers.RandomContrast(0.10)(x)
    x = layers.Conv2D(24, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D()(x)
    for filters in (32, 64, 96):
        x = layers.SeparableConv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x); x = layers.Activation("relu")(x)
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
    print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
    train_ds = make_dataset(trp, trl, args.image_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(vap, val, args.image_size, args.batch_size, False, args.seed)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best.keras"
    model = build_model(args.image_size, len(class_names), args.learning_rate)
    cb = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_loss",
                                        save_best_only=True, mode="min", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                      restore_best_weights=True, mode="min", verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2,
                                          min_lr=1e-5, verbose=1),
    ]
    history = model.fit(train_ds, validation_data=val_ds, shuffle=False,
                        epochs=args.smoke_epochs if args.smoke else args.epochs,
                        callbacks=cb, verbose=2)
    best = keras.models.load_model(best_path, compile=False)
    best.compile(optimizer=keras.optimizers.Adam(args.learning_rate),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    scores = best.predict(val_ds, verbose=0)
    y_pred = scores.argmax(axis=1)
    smoke_check = None
    if args.smoke:
        smoke_check = check_smoke_acceptance(scores, val, len(class_names))
        print("Smoke acceptance:", smoke_check)
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
        "augmentation": ["rotation_0.04", "translation_0.04", "zoom_0.06", "contrast_0.10"],
        "seed": args.seed, "validation_split": args.validation_split,
        "effective_validation_fraction": float(len(vap) / (len(trp) + len(vap))),
        "split_strategy": duplicate_stats,
        "train_samples": int(len(trp)), "validation_samples": int(len(vap)),
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
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-per-class", type=int, default=40)
    parser.add_argument("--smoke-epochs", type=int, default=8)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
