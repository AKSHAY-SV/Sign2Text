"""Evaluate labeled local hand crops with a model and its matching metadata."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.cnn_classifier import CnnClassifier, preprocess_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def evaluate(crops_dir: Path, model_dir: Path, output_dir: Path) -> dict:
    classifier = CnnClassifier(model_dir / "best.keras", model_dir / "metadata.json")
    class_names = classifier.class_names
    y_true, y_pred = [], []
    per_label_counts: dict[str, int] = {}
    for class_dir in sorted(path for path in crops_dir.iterdir() if path.is_dir()):
        if class_dir.name not in class_names:
            raise ValueError(f"Crop label {class_dir.name!r} is absent from model metadata.")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode labeled crop: {path}")
            prepared = preprocess_image(image, (classifier.image_size, classifier.image_size))
            scores = classifier.predict_proba(prepared)
            y_true.append(class_names.index(class_dir.name))
            y_pred.append(int(np.argmax(scores)))
            per_label_counts[class_dir.name] = per_label_counts.get(class_dir.name, 0) + 1
    if not y_true:
        raise ValueError(f"No labeled images found under {crops_dir}.")
    labels = np.arange(len(class_names))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    result = {
        "model": str(classifier.model_path), "metadata": str(classifier.metadata_path),
        "crops_dir": str(crops_dir.resolve()), "sample_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "class_counts": per_label_counts, "class_names": class_names,
        "input_shape": [classifier.image_size, classifier.image_size, 3],
        "color_order": "RGB", "normalization": classifier.metadata.get("normalization"),
        "classification_report": report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(result, indent=2))
    with (output_dir / "confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        writer.writerows([name, *row.tolist()] for name, row in zip(class_names, matrix))
    print(f"Labeled crop accuracy: {result['accuracy']:.2%} ({len(y_true)} images)")
    for name in class_names:
        row = report[name]
        print(f"  {name}: {row['recall']:.1%} ({int(row['support'])} samples)")
    print(f"Reports saved to {output_dir}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True,
                        help="directory containing one subdirectory per true class")
    parser.add_argument("--model-dir", type=Path, default=Path("models/asl_webcam_candidate"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/asl_webcam_eval"))
    args = parser.parse_args()
    evaluate(args.crops, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
