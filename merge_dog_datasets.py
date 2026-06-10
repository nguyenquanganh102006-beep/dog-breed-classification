"""Copy images from one class-folder dataset into another."""

from __future__ import annotations

import argparse
import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class MergeStats:
    copied: int = 0
    skipped_duplicates: int = 0
    renamed: int = 0
    created_classes: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/merged_dog_dataset"),
        help="Dataset whose images will be copied.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("data/merged_dog_dataset_v2"),
        help="Dataset that will receive the images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without copying any files.",
    )
    return parser.parse_args()


def same_file_content(first: Path, second: Path) -> bool:
    return first.is_file() and second.is_file() and filecmp.cmp(
        first, second, shallow=False
    )


def available_destination(source: Path, destination: Path) -> tuple[Path, bool]:
    """Return a free destination, or the existing path for duplicate content."""
    if not destination.exists() or same_file_content(source, destination):
        return destination, False

    counter = 1
    while True:
        candidate = destination.with_name(
            f"{destination.stem}__merged_{counter}{destination.suffix}"
        )
        if not candidate.exists() or same_file_content(source, candidate):
            return candidate, True
        counter += 1


def target_classes_by_name(target: Path) -> dict[str, Path]:
    classes: dict[str, Path] = {}
    for class_dir in target.iterdir():
        if not class_dir.is_dir():
            continue

        key = class_dir.name.casefold()
        if key in classes:
            raise SystemExit(
                "Target contains class directories that differ only by letter case: "
                f"{classes[key]} and {class_dir}"
            )
        classes[key] = class_dir
    return classes


def merge_datasets(source: Path, target: Path, dry_run: bool) -> MergeStats:
    stats = MergeStats()
    target.mkdir(parents=True, exist_ok=True)
    target_classes = target_classes_by_name(target)

    source_classes = sorted(path for path in source.iterdir() if path.is_dir())
    for source_class in source_classes:
        key = source_class.name.casefold()
        target_class = target_classes.get(key)
        if target_class is None:
            target_class = target / source_class.name
            target_classes[key] = target_class
            stats.created_classes += 1
            if not dry_run:
                target_class.mkdir(parents=True, exist_ok=True)

        images = sorted(
            path
            for path in source_class.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for image in images:
            relative_path = image.relative_to(source_class)
            requested_destination = target_class / relative_path
            destination, renamed = available_destination(
                image, requested_destination
            )

            if destination.exists() and same_file_content(image, destination):
                stats.skipped_duplicates += 1
                continue

            if not dry_run:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, destination)
            stats.copied += 1
            stats.renamed += int(renamed)

    return stats


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()

    if not source.is_dir():
        raise SystemExit(f"Source dataset not found: {source}")
    if source == target:
        raise SystemExit("Source and target datasets must be different directories.")

    stats = merge_datasets(source, target, args.dry_run)
    mode = "Dry run finished" if args.dry_run else "Merge finished"
    print(
        f"{mode} | copied: {stats.copied} | "
        f"duplicates skipped: {stats.skipped_duplicates} | "
        f"renamed: {stats.renamed} | classes created: {stats.created_classes}"
    )


if __name__ == "__main__":
    main()
