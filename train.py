from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from class_presets import selected_classes_for_preset
from data_utils import load_and_prepare_data
from metrics import classification_metrics
from model import DecisionTree, KNN, LinearSVM, MLPMixerClassifier, ResidualMLP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a self-implemented classifier.")
    parser.add_argument("--model", choices=("knn", "svm", "decision_tree", "mlp"), required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--num-classes",
        type=int,
        default=0,
        help="Use the first N classes alphabetically; 0 uses every class.",
    )
    parser.add_argument(
        "--class-preset",
        choices=("all", "diverse5", "diverse10"),
        default="all",
        help="Use a predefined subset of visually distinct breeds.",
    )
    parser.add_argument("--no-scaler", action="store_true")
    parser.add_argument("--pca-components", type=int, default=0)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path to save the trained checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-samples-split", type=int, default=4)
    parser.add_argument("--max-thresholds", type=int, default=20)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--optimizer",
        choices=("sgd", "adamw"),
        default="sgd",
        help="Optimizer for MLP-Mixer; other models ignore this.",
    )
    parser.add_argument("--c", type=float, default=1.0, help="SVM C penalty.")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--mlp-type", choices=("mixer", "original"), default="mixer")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--token-mlp-size", type=int, default=128)
    parser.add_argument(
        "--channel-mlp-size",
        type=int,
        default=0,
        help="Channel MLP size for MLP-Mixer; 0 uses hidden-size * expansion.",
    )
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--feature-noise", type=float, default=0.01)
    parser.add_argument("--feature-drop", type=float, default=0.02)
    parser.add_argument("--crop-padding", type=int, default=4)
    parser.add_argument("--hflip-prob", type=float, default=0.5)
    parser.add_argument("--erase-prob", type=float, default=0.1)
    parser.add_argument("--erase-scale", type=float, default=0.15)
    parser.add_argument("--mixup-alpha", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--drop-path", type=float, default=0.05)
    parser.add_argument("--layer-scale", type=float, default=0.1)
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
        help="EMA decay for MLP-Mixer weights; 0 disables EMA.",
    )
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return requested


def build_model(args: argparse.Namespace, input_size: int, num_classes: int, device: str):
    if args.model == "knn":
        return KNN(k=args.k, batch_size=args.batch_size, device=device)
    if args.model == "svm":
        return LinearSVM(
            num_classes=num_classes,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            c=args.c,
            batch_size=args.batch_size,
            device=device,
        )
    if args.model == "decision_tree":
        return DecisionTree(
            num_classes=num_classes,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            max_thresholds=args.max_thresholds,
        )
    if args.mlp_type == "mixer":
        return MLPMixerClassifier(
            input_size=input_size,
            num_classes=num_classes,
            image_size=args.image_size,
            channels=args.channels,
            patch_size=args.patch_size,
            hidden_size=args.hidden_size,
            depth=args.depth,
            expansion=args.expansion,
            token_mlp_size=args.token_mlp_size,
            channel_mlp_size=args.channel_mlp_size,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            optimizer_name=args.optimizer,
            label_smoothing=args.label_smoothing,
            feature_noise=args.feature_noise,
            feature_drop=args.feature_drop,
            crop_padding=args.crop_padding,
            hflip_prob=args.hflip_prob,
            erase_prob=args.erase_prob,
            erase_scale=args.erase_scale,
            mixup_alpha=args.mixup_alpha,
            ema_decay=args.ema_decay,
            warmup_epochs=args.warmup_epochs,
            grad_clip=args.grad_clip,
            drop_path=args.drop_path,
            layer_scale=args.layer_scale,
            patience=args.patience,
            device=device,
        )
    return ResidualMLP(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=args.hidden_size,
        depth=args.depth,
        expansion=args.expansion,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        feature_noise=args.feature_noise,
        feature_drop=args.feature_drop,
        patience=args.patience,
        device=device,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model,
    args: argparse.Namespace,
    input_size: int,
    num_classes: int,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "network"):
        model_state = model.network.state_dict()
    elif hasattr(model, "_state_dict"):
        model_state = model._state_dict()
    else:
        model_state = None

    torch.save(
        {
            "model": args.model,
            "mlp_type": args.mlp_type if args.model == "mlp" else None,
            "input_size": input_size,
            "num_classes": num_classes,
            "args": vars(args),
            "model_state": model_state,
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to: {checkpoint_path}")


def main() -> None:
    args = parse_args()
    if args.model == "mlp" and args.mlp_type == "mixer":
        if not args.no_scaler:
            raise SystemExit("MLP-Mixer needs raw normalized pixels. Add --no-scaler.")
        if args.pca_components > 0:
            raise SystemExit("MLP-Mixer needs full image pixels. Do not use --pca-components.")

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

    model = build_model(args, data.X_train.shape[1], data.num_classes, device)
    model_name = f"{args.model}:{args.mlp_type}" if args.model == "mlp" else args.model
    print(f"Training model: {model_name}")
    model.fit(data.X_train, data.y_train, data.X_val, data.y_val)

    validation_metrics = classification_metrics(
        model.predict_scores(data.X_val), data.y_val
    )
    test_metrics = classification_metrics(model.predict_scores(data.X_test), data.y_test)
    print(
        f"Validation | top-1: {validation_metrics['top1']:.2f}% | "
        f"top-3: {validation_metrics['top3']:.2f}% | "
        f"top-5: {validation_metrics['top5']:.2f}%"
    )
    print(
        f"Test       | top-1: {test_metrics['top1']:.2f}% | "
        f"top-3: {test_metrics['top3']:.2f}% | "
        f"top-5: {test_metrics['top5']:.2f}%"
    )
    if args.checkpoint is not None:
        save_checkpoint(
            checkpoint_path=args.checkpoint,
            model=model,
            args=args,
            input_size=data.X_train.shape[1],
            num_classes=data.num_classes,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )


if __name__ == "__main__":
    main()
