"""Train a PyTorch-library residual MLP on a preprocessed CSV dataset."""

from __future__ import annotations

import argparse
import copy
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from class_presets import DIVERSE_5_DOG_CLASSES, DIVERSE_DOG_CLASSES
from data_utils import DatasetSplits, load_and_prepare_data


class FeatureAugmentation(nn.Module):
    def __init__(self, noise_std: float, feature_drop: float) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.feature_drop = feature_drop

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return features
        if self.noise_std > 0:
            features = features + torch.randn_like(features) * self.noise_std
        if self.feature_drop > 0:
            mask = torch.rand_like(features) > self.feature_drop
            features = features * mask / (1.0 - self.feature_drop)
        return features


class ImageVectorDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        image_size: int,
        channels: int,
        augment: bool,
    ) -> None:
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        self.image_size = image_size
        self.channels = channels
        self.augment = augment

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if not self.augment:
            return features, self.labels[index]

        image = features.reshape(self.image_size, self.image_size, self.channels)
        if torch.rand(()) < 0.5:
            image = image.flip(1)
        image = image.permute(2, 0, 1)
        image = self._random_crop(image, padding=4)
        if torch.rand(()) < 0.25:
            image = self._random_erasing(image)
        return image.permute(1, 2, 0).reshape(-1), self.labels[index]

    def _random_crop(self, image: torch.Tensor, padding: int) -> torch.Tensor:
        padded = nn.functional.pad(image, (padding, padding, padding, padding))
        top = int(torch.randint(0, padding * 2 + 1, ()).item())
        left = int(torch.randint(0, padding * 2 + 1, ()).item())
        return padded[:, top : top + self.image_size, left : left + self.image_size]

    @staticmethod
    def _random_erasing(image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        erase_height = max(1, int(height * float(torch.empty(()).uniform_(0.1, 0.25))))
        erase_width = max(1, int(width * float(torch.empty(()).uniform_(0.1, 0.25))))
        top = int(torch.randint(0, height - erase_height + 1, ()).item())
        left = int(torch.randint(0, width - erase_width + 1, ()).item())
        image = image.clone()
        image[:, top : top + erase_height, left : left + erase_width] = 0
        return image


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return values
        keep_probability = 1.0 - self.probability
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        mask = values.new_empty(shape).bernoulli_(keep_probability)
        return values * mask / keep_probability


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, expansion: int, dropout: float) -> None:
        super().__init__()
        expanded_size = hidden_size * expansion
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, expanded_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expanded_size, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.block(features)


class LibraryMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int,
        depth: int,
        expansion: int,
        dropout: float,
        feature_noise: float,
        feature_drop: float,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            FeatureAugmentation(feature_noise, feature_drop),
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            *[
                ResidualMLPBlock(hidden_size, expansion, dropout)
                for _ in range(depth)
            ],
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class MixerBlock(nn.Module):
    def __init__(
        self,
        num_patches: int,
        hidden_size: int,
        token_mlp_size: int,
        channel_mlp_size: int,
        dropout: float,
        drop_path: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        self.token_norm = nn.LayerNorm(hidden_size)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_patches, token_mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_mlp_size, num_patches),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(hidden_size)
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_size, channel_mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel_mlp_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.token_scale = nn.Parameter(torch.full((hidden_size,), layer_scale))
        self.channel_scale = nn.Parameter(torch.full((hidden_size,), layer_scale))
        self.drop_path = DropPath(drop_path)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        token_features = self.token_norm(patches).transpose(1, 2)
        token_features = self.token_mixing(token_features).transpose(1, 2)
        patches = patches + self.drop_path(token_features * self.token_scale)
        channel_features = self.channel_mixing(self.channel_norm(patches))
        return patches + self.drop_path(channel_features * self.channel_scale)


class MLPMixer(nn.Module):
    """MLP-only image model that preserves spatial information using patches."""

    def __init__(
        self,
        image_size: int,
        channels: int,
        patch_size: int,
        num_classes: int,
        hidden_size: int,
        depth: int,
        token_mlp_size: int,
        channel_mlp_size: int,
        dropout: float,
        feature_noise: float,
        feature_drop: float,
        drop_path: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image-size must be divisible by patch-size.")
        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_features = channels * patch_size * patch_size

        self.augmentation = FeatureAugmentation(feature_noise, feature_drop)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size)
        )
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_features),
            nn.Linear(patch_features, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_patches,
                    hidden_size,
                    token_mlp_size,
                    channel_mlp_size,
                    dropout,
                    drop_path * index / max(1, depth - 1),
                    layer_scale,
                )
                for index in range(depth)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
        )
        self.token_pool = nn.Linear(num_patches, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )
        self.apply(self._initialize_weights)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self.augmentation(features)
        images = features.reshape(
            -1, self.image_size, self.image_size, self.channels
        )
        patches = images.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().reshape(images.shape[0], -1, self.channels * self.patch_size**2)
        patches = self.patch_projection(patches) + self.position_embedding
        patches = self.blocks(patches)
        patches = self.classifier(patches)
        pooled = self.token_pool(patches.transpose(1, 2)).squeeze(-1)
        return self.head(pooled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/library_mlp"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/library_mlp.pt"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--architecture",
        choices=("residual", "mixer"),
        default="mixer",
        help="MLP architecture. Mixer preserves image patch structure.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Image width/height for Mixer; 0 infers it from feature count.",
    )
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--token-mlp-size", type=int, default=256)
    parser.add_argument("--channel-mlp-size", type=int, default=1024)
    parser.add_argument("--drop-path", type=float, default=0.05)
    parser.add_argument("--layer-scale", type=float, default=0.1)
    parser.add_argument("--no-spatial-augmentation", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.99)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--expansion", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--feature-noise", type=float, default=0.005)
    parser.add_argument("--feature-drop", type=float, default=0.01)
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.1,
        help="Mixup strength; 0 disables Mixup.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Linearly increase the learning rate during the first epochs.",
    )
    parser.add_argument("--patience", type=int, default=40)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    return torch.device(requested)


def infer_image_size(input_size: int, channels: int, requested_size: int) -> int:
    if requested_size > 0:
        if requested_size * requested_size * channels != input_size:
            raise ValueError(
                f"image-size={requested_size} and channels={channels} do not match "
                f"the {input_size} input features."
            )
        return requested_size
    inferred_size = int(round((input_size / channels) ** 0.5))
    if inferred_size * inferred_size * channels != input_size:
        raise ValueError(
            "Cannot infer square image dimensions. Set --image-size and --channels."
        )
    return inferred_size


def make_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
    image_size: int,
    channels: int,
    augment: bool,
) -> DataLoader:
    dataset = ImageVectorDataset(
        features,
        labels,
        image_size=image_size,
        channels=channels,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    mixup_alpha: float,
    ema_state: dict[str, torch.Tensor],
    ema_decay: float,
) -> float:
    model.train()
    total_loss = 0.0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if mixup_alpha > 0:
            mixup_weight = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(len(features), device=device)
            mixed_features = (
                mixup_weight * features + (1.0 - mixup_weight) * features[permutation]
            )
            mixed_labels = labels[permutation]
        else:
            mixup_weight = 1.0
            mixed_features = features
            mixed_labels = labels

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(mixed_features)
            loss = mixup_weight * criterion(logits, labels) + (
                1.0 - mixup_weight
            ) * criterion(logits, mixed_labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema_state, model, ema_decay)
        total_loss += loss.item() * len(features)
    return total_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, labels)

        predictions = logits.topk(k=min(5, logits.shape[1]), dim=1).indices
        matches = predictions.eq(labels[:, None])
        top1_correct += matches[:, :1].sum().item()
        top3_correct += matches[:, :3].sum().item()
        top5_correct += matches.sum().item()
        total_loss += loss.item() * len(features)

    size = len(loader.dataset)
    return (
        total_loss / size,
        100.0 * top1_correct / size,
        100.0 * top3_correct / size,
        100.0 * top5_correct / size,
    )


def build_loaders(
    data: DatasetSplits,
    args: argparse.Namespace,
    pin_memory: bool,
    image_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    return (
        make_loader(
            data.X_train,
            data.y_train,
            args.batch_size,
            True,
            args.workers,
            pin_memory,
            image_size,
            args.channels,
            not args.no_spatial_augmentation,
        ),
        make_loader(
            data.X_val,
            data.y_val,
            args.batch_size,
            False,
            args.workers,
            pin_memory,
            image_size,
            args.channels,
            False,
        ),
        make_loader(
            data.X_test,
            data.y_test,
            args.batch_size,
            False,
            args.workers,
            pin_memory,
            image_size,
            args.channels,
            False,
        ),
    )


def learning_rate_for_epoch(
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    maximum_lr: float,
) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return maximum_lr * epoch / warmup_epochs
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    progress = (epoch - warmup_epochs - 1) / cosine_epochs
    minimum_lr = maximum_lr * 0.01
    return minimum_lr + 0.5 * (maximum_lr - minimum_lr) * (
        1.0 + math.cos(math.pi * progress)
    )


def make_optimizer(
    model: nn.Module, learning_rate: float, weight_decay: float
) -> torch.optim.Optimizer:
    decay_parameters = []
    no_decay_parameters = []
    for name, parameter in model.named_parameters():
        if parameter.ndim <= 1 or name.endswith("bias"):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


@torch.no_grad()
def update_ema(
    ema_state: dict[str, torch.Tensor],
    model: nn.Module,
    decay: float,
) -> None:
    for name, value in model.state_dict().items():
        if value.is_floating_point():
            ema_state[name].mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            ema_state[name].copy_(value)


def main() -> None:
    args = parse_args()
    if args.architecture == "mixer" and (
        not args.no_scaler or args.pca_components > 0
    ):
        raise SystemExit(
            "MLP-Mixer requires raw normalized pixels. Use --no-scaler and do not enable PCA."
        )
    random.seed(args.random_state)
    np.random.seed(args.random_state)
    torch.manual_seed(args.random_state)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.random_state)
        torch.backends.cudnn.benchmark = True
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
        selected_classes=(
            DIVERSE_5_DOG_CLASSES
            if args.class_preset == "diverse5"
            else DIVERSE_DOG_CLASSES
            if args.class_preset == "diverse10"
            else None
        ),
    )
    image_size = infer_image_size(data.X_train.shape[1], args.channels, args.image_size)
    train_loader, val_loader, test_loader = build_loaders(
        data,
        args,
        pin_memory=device.type == "cuda",
        image_size=image_size,
    )

    if args.architecture == "mixer":
        model = MLPMixer(
            image_size=image_size,
            channels=args.channels,
            patch_size=args.patch_size,
            num_classes=data.num_classes,
            hidden_size=args.hidden_size,
            depth=args.depth,
            token_mlp_size=args.token_mlp_size,
            channel_mlp_size=args.channel_mlp_size,
            dropout=args.dropout,
            feature_noise=args.feature_noise,
            feature_drop=args.feature_drop,
            drop_path=args.drop_path,
            layer_scale=args.layer_scale,
        ).to(device)
        print(
            f"Architecture: MLP-Mixer | image: {image_size}x{image_size}x{args.channels} | "
            f"patch: {args.patch_size}x{args.patch_size}"
        )
    else:
        model = LibraryMLP(
            input_size=data.X_train.shape[1],
            num_classes=data.num_classes,
            hidden_size=args.hidden_size,
            depth=args.depth,
            expansion=args.expansion,
            dropout=args.dropout,
            feature_noise=args.feature_noise,
            feature_drop=args.feature_drop,
        ).to(device)
        print("Architecture: residual flatten MLP")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Train: {len(data.X_train)} | Val: {len(data.X_val)} | "
        f"Test: {len(data.X_test)} | Features: {data.X_train.shape[1]} | "
        f"Classes: {data.num_classes} | Parameters: {parameter_count:,}"
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    warmup_epochs = min(max(0, args.warmup_epochs), args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_state = copy.deepcopy(model.state_dict())

    best_top1 = -1.0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        current_lr = learning_rate_for_epoch(
            epoch,
            args.epochs,
            warmup_epochs,
            args.learning_rate,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            args.mixup_alpha,
            ema_state,
            args.ema_decay,
        )
        raw_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(ema_state)
        val_loss, val_top1, val_top3, val_top5 = evaluate(
            model, val_loader, criterion, device, use_amp
        )
        evaluated_state = copy.deepcopy(model.state_dict())
        model.load_state_dict(raw_state)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr: {optimizer.param_groups[0]['lr']:.2e} | "
            f"train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | "
            f"val top-1: {val_top1:.2f}% | val top-3: {val_top3:.2f}% | "
            f"val top-5: {val_top5:.2f}%"
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            best_state = evaluated_state
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint)
    test_loss, test_top1, test_top3, test_top5 = evaluate(
        model, test_loader, criterion, device, use_amp
    )
    print(
        f"Test | loss: {test_loss:.4f} | top-1: {test_top1:.2f}% | "
        f"top-3: {test_top3:.2f}% | top-5: {test_top5:.2f}%"
    )
    print(f"Saved best model to: {args.checkpoint}")


if __name__ == "__main__":
    main()
