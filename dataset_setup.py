"""
Dataset preparation utility for YOLO training.

Creates a train/val/test split from a folder of images + YOLO labels and
writes the structure expected by this project.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a YOLO dataset split and directory layout."
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        type=Path,
        help="Directory containing all source images.",
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        type=Path,
        help="Directory containing YOLO .txt labels with matching file names.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output dataset directory (default: data).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Class names in dataset.yaml order, e.g. --classes water orange apple.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split (default: 42).",
    )
    return parser.parse_args()


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    image_files = sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    pairs: list[tuple[Path, Path]] = []
    missing_labels: list[Path] = []
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path)

    if missing_labels:
        print(f"Warning: {len(missing_labels)} images have no matching label and will be skipped.")

    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[tuple[Path, Path]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio > 1:
        raise ValueError("train_ratio + val_ratio must be <= 1.")

    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def ensure_structure(output_dir: Path) -> None:
    for split_name in ("train", "val", "test"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)


def copy_split_files(output_dir: Path, split_name: str, items: Iterable[tuple[Path, Path]]) -> None:
    image_dest = output_dir / "images" / split_name
    label_dest = output_dir / "labels" / split_name

    count = 0
    for image_path, label_path in items:
        shutil.copy2(image_path, image_dest / image_path.name)
        shutil.copy2(label_path, label_dest / label_path.name)
        count += 1
    print(f"{split_name}: copied {count} image/label pairs")


def write_dataset_yaml(output_dir: Path, classes: list[str]) -> None:
    yaml_path = output_dir / "dataset.yaml"
    lines = [
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    lines.extend([f"  - {name}" for name in classes])
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote dataset config: {yaml_path}")


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    labels_dir = args.labels_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    pairs = collect_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError("No valid image/label pairs found.")

    print(f"Found {len(pairs)} labeled image pairs.")

    ensure_structure(output_dir)
    split = split_pairs(pairs, args.train_ratio, args.val_ratio, args.seed)

    for split_name in ("train", "val", "test"):
        copy_split_files(output_dir, split_name, split[split_name])

    write_dataset_yaml(output_dir, args.classes)
    print("Dataset setup complete.")


if __name__ == "__main__":
    main()
