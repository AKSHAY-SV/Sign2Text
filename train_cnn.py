"""Reproducible CNN training for static ASL alphabet images.

The output directory is explicit so experiments cannot silently replace the
currently deployed model. Images are RGB, resized to a square, and scaled to
[0, 1].
"""
from __future__ import annotations

import argparse
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

from config.settings import DATASET_DIR, MODELS_DIR

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SPECIAL = ["del", "nothing", "space"]
EXPECTED_CLASSES = LETTERS + SPECIAL
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def audit_dataset(root: Path, cache_path: Path | None = None) -> tuple[list[str], dict[str, list[Path]], dict[str, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    classes = [p.name for p in class_dirs]
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Expected classes {EXPECTED_CLASSES}; found {classes}")
    files: dict[str, list[Path]] = {}
    invalid: dict[str, int] = {}
    import cv2
    fingerprint = []
    for directory in class_dirs:
        items = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        fingerprint.append((directory.name, len(items), sum(p.stat().st_size for p in items),
                            max((p.stat().st_mtime_ns for p in items), default=0)))
    if cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("fingerprint") == fingerprint:
                cached_files = {name: [Path(p) for p in paths] for name, paths in cached["files"].items()}
                if all(all(p.is_file() for p in paths) for paths in cached_files.values()):
                    return classes, cached_files, cached["invalid"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    for directory in class_dirs:
        good, bad = [], 0
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            try:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None or image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError("not a decodable color image")
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
                                          "invalid": invalid}))
    return classes, files, invalid


def split_files(files: dict[str, list[Path]], validation_split: float, seed: int,
                max_per_class: int | None = None):
    rng = random.Random(seed)
    train_paths, train_labels, val_paths, val_labels = [], [], [], []
    classes = list(files)
    for class_index, name in enumerate(classes):
        choices = list(files[name])
        rng.shuffle(choices)
        if max_per_class is not None:
            choices = choices[:max_per_class]
        n_val = max(1, int(round(len(choices) * validation_split)))
        val, train = choices[:n_val], choices[n_val:]
        train_paths.extend(map(str, train)); train_labels.extend([class_index] * len(train))
        val_paths.extend(map(str, val)); val_labels.extend([class_index] * len(val))
    # Deterministic global train shuffle; validation remains stable for reports.
    order = list(range(len(train_paths))); rng.shuffle(order)
    return (np.asarray(train_paths)[order], np.asarray(train_labels, np.int32)[order],
            np.asarray(val_paths), np.asarray(val_labels, np.int32), classes)


def make_dataset(paths, labels, image_size: int, batch_size: int, training: bool,
                 seed: int):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size], antialias=True)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    options = tf.data.Options()
    options.threading.private_threadpool_size = 2
    options.threading.max_intra_op_parallelism = 1
    return ds.map(load, num_parallel_calls=2).batch(batch_size).prefetch(1).with_options(options)


def build_model(image_size: int, num_classes: int, learning_rate: float):
    inputs = keras.Input((image_size, image_size, 3), name="rgb_image_01")
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.06, fill_mode="reflect")(x)
    x = layers.RandomZoom(0.08, fill_mode="reflect")(x)
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
    classes, files, invalid = audit_dataset(args.dataset, args.output_dir / "dataset_audit.json")
    counts = {name: len(files[name]) for name in classes}
    print(f"Dataset: {sum(counts.values())} valid images; classes={classes}")
    print(f"Invalid image counts: {invalid}")
    max_per_class = args.smoke_per_class if args.smoke else None
    trp, trl, vap, val, class_names = split_files(
        files, args.validation_split, args.seed, max_per_class)
    print(f"Split: {len(trp)} train, {len(vap)} validation; GPU={tf.config.list_physical_devices('GPU')}")
    train_ds = make_dataset(trp, trl, args.image_size, args.batch_size, True, args.seed)
    val_ds = make_dataset(vap, val, args.image_size, args.batch_size, False, args.seed)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "best.keras"
    model = build_model(args.image_size, len(class_names), args.learning_rate)
    cb = [
        keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
                                        save_best_only=True, mode="max", verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience,
                                      restore_best_weights=True, mode="max", verbose=1),
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
    val_loss, val_acc = best.evaluate(val_ds, verbose=0)
    cm = confusion_matrix(val, y_pred, labels=np.arange(len(class_names)))
    report = classification_report(val, y_pred, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True,
                                   zero_division=0)
    # Keep all training artifacts isolated. Caller may promote best.keras only
    # after review; existing models/sign_cnn.h5 is never overwritten here.
    cm_path = out / "confusion_matrix.csv"
    np.savetxt(cm_path, cm, delimiter=",", fmt="%d",
               header=",".join(class_names), comments="")
    report_path = out / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    metadata = {
        "model_version": "2.0.0", "dataset": str(args.dataset.resolve()),
        "image_size": args.image_size, "input_shape": [args.image_size, args.image_size, 3],
        "num_classes": len(class_names), "class_names": class_names,
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "color_order": "RGB", "normalization": "float32 pixel values divided by 255, range [0,1]",
        "resize": "bilinear antialiased square resize", "augmentation": ["horizontal_flip", "rotation_0.06", "zoom_0.08"],
        "seed": args.seed, "validation_split": args.validation_split,
        "train_samples": int(len(trp)), "validation_samples": int(len(vap)),
        "best_val_accuracy": float(val_acc), "best_val_loss": float(val_loss),
        "smoke_test": bool(args.smoke), "training_epochs_completed": len(history.history["loss"]),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "live_input_note": "Training uses centered dataset photos; live hand crop domain requires camera validation.",
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
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR / "asl_letters_v2")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
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
