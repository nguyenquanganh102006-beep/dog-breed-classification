"""Sweep max-thresholds for the self-implemented decision tree."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from class_presets import selected_classes_for_preset
from data_utils import load_and_prepare_data
from metrics import classification_metrics
from model import DecisionTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument(
        "--class-preset",
        choices=("all", "diverse5", "diverse10"),
        default="all",
    )
    parser.add_argument("--no-scaler", action="store_true")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=100,
        help="Recommended for decision tree on image pixels; 0 disables PCA.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/decision_tree"),
    )
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument(
        "--max-threshold-values",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20],
        help="List of max-thresholds values to test.",
    )
    return parser.parse_args()


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} | top-1: {metrics['top1']:.2f}% | "
        f"top-3: {metrics['top3']:.2f}% | top-5: {metrics['top5']:.2f}%"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    for value in args.max_threshold_values:
        if value < 1:
            raise ValueError("max-threshold values must be positive.")

    data = load_and_prepare_data(
        data_path=args.data_path,
        label_column=args.label_column,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state,
        use_scaler=not args.no_scaler,
        pca_components=args.pca_components,
        artifacts_dir=args.artifacts_dir,
        num_classes=args.num_classes,
        selected_classes=selected_classes_for_preset(args.class_preset),
    )
    print(
        f"Split sizes | train: {len(data.X_train)} | val: {len(data.X_val)} | "
        f"test: {len(data.X_test)} | features: {data.X_train.shape[1]} | "
        f"classes: {data.num_classes}"
    )

    best_thresholds = None
    best_validation = None
    best_test = None

    for max_thresholds in args.max_threshold_values:
        print(
            f"\nTraining Decision Tree with max_depth={args.max_depth}, "
            f"max_thresholds={max_thresholds}"
        )
        model = DecisionTree(
            num_classes=data.num_classes,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            max_thresholds=max_thresholds,
        )
        model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
        validation_metrics = classification_metrics(
            model.predict_scores(data.X_val), data.y_val
        )
        test_metrics = classification_metrics(model.predict_scores(data.X_test), data.y_test)
        print_metrics(f"thresholds={max_thresholds} | Validation", validation_metrics)
        print_metrics(f"thresholds={max_thresholds} | Test      ", test_metrics)

        if best_validation is None or validation_metrics["top1"] > best_validation["top1"]:
            best_thresholds = max_thresholds
            best_validation = validation_metrics
            best_test = test_metrics

    print(f"\nBest max-thresholds by validation top-1: {best_thresholds}")
    print_metrics("Best validation", best_validation)
    print_metrics("Best test      ", best_test)


if __name__ == "__main__":
    main()
