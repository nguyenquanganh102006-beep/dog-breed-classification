"""Sweep KNN k values on a preprocessed CSV dataset."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from class_presets import selected_classes_for_preset
from data_utils import load_and_prepare_data
from metrics import classification_metrics


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
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/knn"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=49)
    parser.add_argument("--k-step", type=int, default=2)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return torch.device(requested)


def validate_k_values(k_values: list[int], train_size: int) -> None:
    if not k_values:
        raise ValueError("No k values selected.")
    if min(k_values) < 1:
        raise ValueError("k values must be positive.")
    if max(k_values) > train_size:
        raise ValueError(f"k cannot exceed train size {train_size}.")


def nearest_labels(
    X_query: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k_max: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    train_features = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    train_labels = torch.as_tensor(y_train, dtype=torch.long, device=device)
    labels_by_batch = []

    for start in range(0, len(X_query), batch_size):
        query = torch.as_tensor(
            X_query[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        distances = torch.cdist(query, train_features)
        indices = distances.topk(k_max, largest=False).indices
        labels_by_batch.append(train_labels[indices].cpu())

    return torch.cat(labels_by_batch, dim=0)


def scores_from_nearest_labels(
    labels: torch.Tensor,
    k: int,
    num_classes: int,
) -> np.ndarray:
    labels = labels[:, :k]
    scores = torch.zeros((labels.shape[0], num_classes), dtype=torch.float32)
    scores.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
    return scores.numpy()


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
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.random_state)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
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
    validate_k_values(k_values, len(data.X_train))
    print(
        f"Split sizes | train: {len(data.X_train)} | val: {len(data.X_val)} | "
        f"test: {len(data.X_test)} | features: {data.X_train.shape[1]} | "
        f"classes: {data.num_classes}"
    )
    print(f"Sweeping k values: {k_values[0]}..{k_values[-1]} step {args.k_step}")

    validation_neighbors = nearest_labels(
        data.X_val,
        data.X_train,
        data.y_train,
        max(k_values),
        args.batch_size,
        device,
    )
    test_neighbors = nearest_labels(
        data.X_test,
        data.X_train,
        data.y_train,
        max(k_values),
        args.batch_size,
        device,
    )

    best_k = None
    best_validation = None
    best_test = None
    for k in k_values:
        validation_metrics = classification_metrics(
            scores_from_nearest_labels(validation_neighbors, k, data.num_classes),
            data.y_val,
        )
        test_metrics = classification_metrics(
            scores_from_nearest_labels(test_neighbors, k, data.num_classes),
            data.y_test,
        )
        print_metrics(f"k={k:02d} | Validation", validation_metrics)
        print_metrics(f"k={k:02d} | Test      ", test_metrics)

        if best_validation is None or validation_metrics["top1"] > best_validation["top1"]:
            best_k = k
            best_validation = validation_metrics
            best_test = test_metrics

    print(f"Best k by validation top-1: {best_k}")
    print_metrics("Best validation", best_validation)
    print_metrics("Best test      ", best_test)


if __name__ == "__main__":
    main()
