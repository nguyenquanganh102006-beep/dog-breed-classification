"""Convert a class-folder image dataset into a flat-feature CSV file."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/merged_dog_dataset_v2"),
        help="Directory containing one subdirectory per class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dog_dataset.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Use one grayscale channel instead of three RGB channels.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Maximum images per class; 0 processes every image.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print progress after this many processed images.",
    )
    return parser.parse_args()


def discover_images(
    data_dir: Path, limit_per_class: int
) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if limit_per_class > 0:
            images = images[:limit_per_class]
        samples.extend((image_path, class_dir.name) for image_path in images)
    return samples


def extract_pixels(image_path: Path, image_size: int, grayscale: bool) -> np.ndarray:
    color_mode = "L" if grayscale else "RGB"
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert(color_mode)
        image = ImageOps.fit(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def write_csv(
    samples: list[tuple[Path, str]],
    output_path: Path,
    image_size: int,
    grayscale: bool,
    progress_every: int,
) -> tuple[int, int, Counter[str]]:
    channels = 1 if grayscale else 3
    feature_count = image_size * image_size * channels
    class_counts: Counter[str] = Counter()
    processed = 0
    skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([f"feature_{index}" for index in range(feature_count)] + ["label"])

        for image_path, label in samples:
            try:
                features = extract_pixels(image_path, image_size, grayscale)
            except (OSError, ValueError, UnidentifiedImageError) as error:
                skipped += 1
                print(f"Skipping {image_path}: {error}")
                continue

            writer.writerow([f"{value:.6g}" for value in features] + [label])
            processed += 1
            class_counts[label] += 1
            if progress_every > 0 and processed % progress_every == 0:
                print(f"Processed {processed}/{len(samples)} images...")

    return processed, skipped, class_counts


def main() -> None:
    args = parse_args()
    if not args.data_dir.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.data_dir}")
    if args.image_size < 1 or args.limit_per_class < 0 or args.progress_every < 0:
        raise SystemExit("image-size must be positive; limits cannot be negative.")

    samples = discover_images(args.data_dir, args.limit_per_class)
    if not samples:
        raise SystemExit(f"No supported images found in: {args.data_dir}")

    channels = 1 if args.grayscale else 3
    feature_count = args.image_size * args.image_size * channels
    print(f"Found {len(samples)} images in {len({label for _, label in samples})} classes.")
    print(f"Output features per image: {feature_count}")
    print(f"Writing CSV to: {args.output}")

    processed, skipped, class_counts = write_csv(
        samples=samples,
        output_path=args.output,
        image_size=args.image_size,
        grayscale=args.grayscale,
        progress_every=args.progress_every,
    )
    print(f"Finished | processed: {processed} | skipped: {skipped}")
    print(f"Classes written: {len(class_counts)}")


if __name__ == "__main__":
    main()
