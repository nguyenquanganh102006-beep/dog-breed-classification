"""Sweep C values for the self-implemented multiclass linear SVM."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from class_presets import selected_classes_for_preset
from data_utils import load_and_prepare_data
from metrics import classification_metrics
from model import LinearSVM


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
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/svm"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
        help="List of C values to test.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return requested


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} | top-1: {metrics['top1']:.2f}% | "
        f"top-3: {metrics['top3']:.2f}% | top-5: {metrics['top5']:.2f}%"
    )


def main() -> None:
    args = parse_args()
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    device = resolve_device(args.device)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.random_state)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

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

    best_c = None
    best_validation = None
    best_test = None

    for c_value in args.c_values:
        if c_value <= 0:
            raise ValueError("C values must be positive.")
        print(f"\nTraining SVM with C={c_value:g}")
        model = LinearSVM(
            num_classes=data.num_classes,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            c=c_value,
            batch_size=args.batch_size,
            device=device,
        )
        model.fit(data.X_train, data.y_train, data.X_val, data.y_val)
        validation_metrics = classification_metrics(
            model.predict_scores(data.X_val), data.y_val
        )
        test_metrics = classification_metrics(model.predict_scores(data.X_test), data.y_test)
        print_metrics(f"C={c_value:g} | Validation", validation_metrics)
        print_metrics(f"C={c_value:g} | Test      ", test_metrics)

        if best_validation is None or validation_metrics["top1"] > best_validation["top1"]:
            best_c = c_value
            best_validation = validation_metrics
            best_test = test_metrics

    print(f"\nBest C by validation top-1: {best_c:g}")
    print_metrics("Best validation", best_validation)
    print_metrics("Best test      ", best_test)


if __name__ == "__main__":
    main()
