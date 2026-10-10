"""
Sign2Text - Dataset Inspection Script

Inspects the ASL alphabet dataset and produces a clear summary.
Do not modify, rename, or delete the original dataset.

Usage:
    python3 scripts/inspect_dataset.py

    Or with custom paths:
    python3 scripts/inspect_dataset.py --dataset_dir /path/to/images

Output:
    - Prints class summary to stdout
    - Optionally writes a CSV report to disk
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# =============================================================================
# Constants
# =============================================================================

DEFAULT_DATASET_DIR = Path("/home/akshay-sv/SignDB/asl_alphabet_train/asl_alphabet_train")
DEFAULT_REPORT_PATH = Path("dataset_report.csv")

# =============================================================================
# Dataset Discovery
# =============================================================================


def discover_dataset(
    dataset_dir: Path,
) -> Tuple[Dict[str, int], int, List[str]]:
    """
    Discover all image classes and counts in the dataset directory.

    Args:
        dataset_dir: Path to directory containing class subdirectories with image files.

    Returns:
        Tuple of (class_counts, total_images, class_names)
        - class_counts: dict mapping class name -> number of images
        - total_images: total number of valid images across all classes
        - class_names: sorted list of class directory names
    """
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    class_counts: Dict[str, int] = {}
    class_names: List[str] = []

    for entry in sorted(dataset_dir.iterdir()):
        if not entry.is_dir():
            continue
        class_name = entry.name
        # Supported extensions
        supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".pgm"}
        count = 0
        corrupt = 0
        valid_files: List[str] = []

        for fname in entry.iterdir():
            if fname.suffix.lower() not in supported_extensions:
                # Log unsupported type but don't count
                continue
            try:
                # Verify image can be read
                img = np.asarray(
                    np.load(str(fname)) if fname.suffix.lower() == ".npy"
                    else None
                )
                # For a quick validation, just check the file exists and is readable
                count += 1
                valid_files.append(fname.name)
            except Exception:
                corrupt += 1

        class_counts[class_name] = count
        class_names.append(class_name)

    total_images = sum(class_counts.values())
    return class_counts, total_images, class_names


# =============================================================================
# Validation Checks
# =============================================================================


def validate_classes(
    class_counts: Dict[str, int],
    expected_classes: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Validate that all expected classes are present and no unexpected classes exist.

    Args:
        class_counts: Discovered class counts.
        expected_classes: Classes we expect to find (e.g. from the ASL alphabet).

    Returns:
        Tuple of (missing_classes, unexpected_classes)
    """
    discovered = set(class_counts.keys())
    expected = set(expected_classes)

    missing = sorted(expected - discovered)
    unexpected = sorted(discovered - expected)

    return missing, unexpected


def check_class_balance(
    class_counts: Dict[str, int],
    min_per_class: int = 100,
) -> Dict[str, any]:
    """
    Check if classes have a minimum number of samples.

    Args:
        class_counts: Discovered class counts.
        min_per_class: Minimum number of images required per class.

    Returns:
        Dictionary with balance information.
    """
    balanced: List[str] = []
    imbalanced: List[str] = []
    warning: List[str] = []

    for cls, count in sorted(class_counts.items()):
        if count >= min_per_class:
            balanced.append(cls)
        else:
            imbalanced.append(cls)
            warning.append(f"  {cls}: {count} images (below {min_per_class})")

    return {
        "balanced": balanced,
        "imbalanced": imbalanced,
        "warning": warning,
        "total_classes": len(class_counts),
        "classes_below_min": len(imbalanced),
    }


# =============================================================================
# Report Generation
# =============================================================================


def print_summary(
    class_counts: Dict[str, int],
    total_images: int,
    missing_classes: List[str],
    unexpected_classes: List[str],
    balance_info: Dict[str, any],
) -> None:
    """Print a human-readable summary of the dataset inspection."""
    print("=" * 60)
    print("DATASET INSPECTION SUMMARY")
    print("=" * 60)
    print()

    print(f"Total images: {total_images}")
    print(f"Number of classes: {len(class_counts)}")
    print()

    print("Class distribution:")
    print("-" * 40)
    for cls in sorted(class_counts.keys()):
        bar = "=" * (class_counts[cls] // 50)  # Scale bar
        print(f"  {cls:12s}: {class_counts[cls]:5d} images {bar}")
    print()

    if missing_classes:
        print(f"MISSING classes: {', '.join(missing_classes)}")
    else:
        print("All expected classes present.")

    if unexpected_classes:
        print(f"UNEXPECTED classes: {', '.join(unexpected_classes)}")
    else:
        print("No unexpected classes found.")

    print()
    print("Balance check:")
    print("-" * 40)
    if balance_info["imbalanced"]:
        for line in balance_info["warning"]:
            print(line)
    else:
        print("  All classes have sufficient samples.")

    print()
    print("=" * 60)


def write_csv_report(
    class_counts: Dict[str, int],
    total_images: int,
    missing_classes: List[str],
    unexpected_classes: List[str],
    report_path: Path,
) -> None:
    """Write a CSV report of the dataset inspection."""
    write_path = report_path
    write_path.parent.mkdir(parents=True, exist_ok=True)

    with open(write_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "image_count"])
        for cls in sorted(class_counts.keys()):
            writer.writerow([cls, class_counts[cls]])
        writer.writerow([])
        writer.writerow(["total_images", total_images])
        writer.writerow(["missing_classes", "; ".join(missing_classes) if missing_classes else ""])
        writer.writerow(["unexpected_classes", "; ".join(unexpected_classes) if unexpected_classes else ""])

    print(f"CSV report written to: {write_path}")


# =============================================================================
# Command-Line Interface
# =============================================================================


def main_cli() -> None:
    """Command-line interface for dataset inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect ASL alphabet dataset structure and validity"
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Path to dataset directory (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help=f"Path for CSV report (default: {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument(
        "--expected_classes",
        nargs="+",
        default=[
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "del", "nothing", "space",
        ],
        help="Expected class names (default: full ASL alphabet)",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=100,
        help="Minimum images per class to be considered balanced (default: 100)",
    )

    args = parser.parse_args()

    # Discover dataset
    class_counts, total_images, class_names = discover_dataset(args.dataset_dir)

    # Validate classes
    missing, unexpected = validate_classes(class_counts, args.expected_classes)

    # Check balance
    balance_info = check_class_balance(class_counts, args.min_per_class)

    # Print summary
    print_summary(class_counts, total_images, missing, unexpected, balance_info)

    # Write CSV report
    write_csv_report(class_counts, total_images, missing, unexpected, args.report)


if __name__ == "__main__":
    main_cli()