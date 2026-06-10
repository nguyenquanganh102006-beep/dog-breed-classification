from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from metrics import classification_metrics
from model.base import BaseClassifier


class ResidualMLP(BaseClassifier):
    """Original hand-written MLP: Linear layers, ReLU, dropout, momentum SGD."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 256,
        depth: int = 5,
        expansion: int = 2,
        dropout: float = 0.3,
        epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 1e-2,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,
        feature_noise: float = 0.0,
        feature_drop: float = 0.0,
        patience: int = 5,
        device: str = "cpu",
    ) -> None:
        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.feature_noise = feature_noise
        self.feature_drop = feature_drop
        self.patience = patience
        self.momentum = 0.9
        self.device = torch.device(device)

        self.layer_sizes = [input_size] + [hidden_size] * depth + [num_classes]
        self.weights = [
            self._make_weight(self.layer_sizes[index], self.layer_sizes[index + 1])
            for index in range(len(self.layer_sizes) - 1)
        ]
        self.biases = [
            torch.zeros(1, self.layer_sizes[index + 1], device=self.device)
            for index in range(len(self.layer_sizes) - 1)
        ]
        self.vel_w = [torch.zeros_like(weight) for weight in self.weights]
        self.vel_b = [torch.zeros_like(bias) for bias in self.biases]

    def _make_weight(self, fan_in: int, fan_out: int) -> torch.Tensor:
        weight = torch.randn(fan_in, fan_out, device=self.device) * math.sqrt(
            2.0 / fan_in
        )
        return weight.requires_grad_(False)

    def _augment(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_noise > 0:
            features = features + torch.randn_like(features) * self.feature_noise
        if self.feature_drop > 0:
            mask = torch.rand_like(features) > self.feature_drop
            features = features * mask / (1.0 - self.feature_drop)
        return features

    def _forward(self, features: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            features = self._augment(features)

        self.activations = [features]
        self.pre_activations = []
        self.dropout_masks = []

        values = features
        for index in range(len(self.weights) - 1):
            z = values @ self.weights[index] + self.biases[index]
            self.pre_activations.append(z)
            values = torch.relu(z)
            if training and self.dropout > 0:
                mask = (torch.rand_like(values) > self.dropout).float()
                values = values * mask / (1.0 - self.dropout)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
            self.activations.append(values)

        logits = values @ self.weights[-1] + self.biases[-1]
        self.pre_activations.append(logits)
        return logits

    def _backward(self, labels: torch.Tensor, logits: torch.Tensor) -> float:
        batch_size = labels.shape[0]
        probabilities = torch.softmax(logits, dim=1)
        targets = torch.zeros_like(probabilities)
        targets.scatter_(1, labels[:, None], 1.0)
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + (
                self.label_smoothing / self.num_classes
            )

        gradient = (probabilities - targets) / batch_size
        grad_weights = []
        grad_biases = []

        for index in range(len(self.weights) - 1, -1, -1):
            grad_w = self.activations[index].t() @ gradient
            grad_b = gradient.sum(dim=0, keepdim=True)
            if self.weight_decay > 0:
                grad_w = grad_w + self.weight_decay * self.weights[index]
            grad_weights.insert(0, grad_w)
            grad_biases.insert(0, grad_b)

            if index > 0:
                gradient = gradient @ self.weights[index].t()
                if self.dropout_masks[index - 1] is not None:
                    gradient = gradient * self.dropout_masks[index - 1]
                gradient = gradient * (self.pre_activations[index - 1] > 0).float()

        total_norm = 0.0
        for grad_w in grad_weights:
            grad_w.clamp_(-1.0, 1.0)
            total_norm += grad_w.norm().item()
        for grad_b in grad_biases:
            grad_b.clamp_(-1.0, 1.0)

        for index in range(len(self.weights)):
            self.vel_w[index] = (
                self.momentum * self.vel_w[index] - self.current_lr * grad_weights[index]
            )
            self.vel_b[index] = (
                self.momentum * self.vel_b[index] - self.current_lr * grad_biases[index]
            )
            self.weights[index] += self.vel_w[index]
            self.biases[index] += self.vel_b[index]

        return total_norm

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        X = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_validation = (
            torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            if X_val is not None
            else X
        )
        validation_labels = y_val if y_val is not None else y_train

        best_top1 = -1.0
        best_state = None
        stale_epochs = 0
        use_minibatch = self.batch_size and self.batch_size > 0

        for epoch in range(1, self.epochs + 1):
            self.current_lr = self.learning_rate * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - 1) / self.epochs)
            )
            total_loss = 0.0

            if use_minibatch:
                indices = torch.randperm(len(X), device=self.device)
                batches = range(0, len(X), self.batch_size)
            else:
                indices = torch.arange(len(X), device=self.device)
                batches = [0]

            for start in batches:
                batch_indices = (
                    indices[start : start + self.batch_size] if use_minibatch else indices
                )
                logits = self._forward(X[batch_indices], training=True)
                loss = F.cross_entropy(
                    logits,
                    y[batch_indices],
                    label_smoothing=self.label_smoothing,
                )
                self._backward(y[batch_indices], logits)
                total_loss += loss.item() * len(batch_indices)

            validation_scores = self._predict_tensor(X_validation)
            metrics = classification_metrics(validation_scores, validation_labels)
            print(
                f"Epoch {epoch:03d}/{self.epochs} | lr: {self.current_lr:.2e} | "
                f"loss: {total_loss / len(X):.4f} | "
                f"val top-1: {metrics['top1']:.2f}% | "
                f"val top-3: {metrics['top3']:.2f}% | "
                f"val top-5: {metrics['top5']:.2f}%"
            )

            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                best_state = self._state_dict()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self._load_state_dict(best_state)

    def _state_dict(self) -> dict[str, list[torch.Tensor]]:
        return {
            "weights": [weight.detach().clone() for weight in self.weights],
            "biases": [bias.detach().clone() for bias in self.biases],
            "vel_w": [velocity.detach().clone() for velocity in self.vel_w],
            "vel_b": [velocity.detach().clone() for velocity in self.vel_b],
        }

    def _load_state_dict(self, state: dict[str, list[torch.Tensor]]) -> None:
        self.weights = [weight.clone() for weight in state["weights"]]
        self.biases = [bias.clone() for bias in state["biases"]]
        self.vel_w = [velocity.clone() for velocity in state["vel_w"]]
        self.vel_b = [velocity.clone() for velocity in state["vel_b"]]

    @torch.inference_mode()
    def _predict_tensor(self, features: torch.Tensor) -> np.ndarray:
        score_batches = []
        for start in range(0, len(features), self.batch_size):
            score_batches.append(
                self._forward(features[start : start + self.batch_size], training=False).cpu()
            )
        return torch.cat(score_batches).numpy()

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        features = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(features)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.drop_prob <= 0 or not self.training:
            return features
        keep_prob = 1.0 - self.drop_prob
        shape = (features.shape[0],) + (1,) * (features.ndim - 1)
        mask = torch.empty(shape, dtype=features.dtype, device=features.device)
        mask.bernoulli_(keep_prob)
        return features * mask / keep_prob


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


class MLPMixerNet(nn.Module):
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
        crop_padding: int,
        hflip_prob: float,
        erase_prob: float,
        erase_scale: float,
        drop_path: float,
        layer_scale: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")

        self.image_size = image_size
        self.channels = channels
        self.patch_size = patch_size
        self.feature_noise = feature_noise
        self.feature_drop = feature_drop
        self.crop_padding = crop_padding
        self.hflip_prob = hflip_prob
        self.erase_prob = erase_prob
        self.erase_scale = erase_scale

        num_patches = (image_size // patch_size) ** 2
        patch_features = channels * patch_size * patch_size
        self.patch_projection = nn.Sequential(
            nn.LayerNorm(patch_features),
            nn.Linear(patch_features, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        self.blocks = nn.Sequential(
            *[
                MixerBlock(
                    num_patches=num_patches,
                    hidden_size=hidden_size,
                    token_mlp_size=token_mlp_size,
                    channel_mlp_size=channel_mlp_size,
                    dropout=dropout,
                    drop_path=drop_path * index / max(1, depth - 1),
                    layer_scale=layer_scale,
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
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

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return features
        if self.feature_noise > 0:
            features = features + torch.randn_like(features) * self.feature_noise
        if self.feature_drop > 0:
            mask = torch.rand_like(features) > self.feature_drop
            features = features * mask / (1.0 - self.feature_drop)
        return features

    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return images

        if self.hflip_prob > 0:
            flip_mask = torch.rand(images.shape[0], device=images.device) < self.hflip_prob
            images[flip_mask] = images[flip_mask].flip(dims=(2,))

        if self.crop_padding > 0:
            images = images.permute(0, 3, 1, 2)
            images = F.pad(
                images,
                (self.crop_padding, self.crop_padding, self.crop_padding, self.crop_padding),
                mode="reflect",
            )
            max_offset = self.crop_padding * 2
            top_offsets = torch.randint(
                0, max_offset + 1, (images.shape[0],), device=images.device
            )
            left_offsets = torch.randint(
                0, max_offset + 1, (images.shape[0],), device=images.device
            )
            cropped = torch.empty(
                images.shape[0],
                self.channels,
                self.image_size,
                self.image_size,
                device=images.device,
                dtype=images.dtype,
            )
            for index, (top, left) in enumerate(zip(top_offsets.tolist(), left_offsets.tolist())):
                cropped[index] = images[
                    index,
                    :,
                    top : top + self.image_size,
                    left : left + self.image_size,
                ]
            images = cropped.permute(0, 2, 3, 1)

        if self.erase_prob > 0 and self.erase_scale > 0:
            erase_mask = torch.rand(images.shape[0], device=images.device) < self.erase_prob
            erase_size = max(1, int(self.image_size * self.erase_scale))
            for index in torch.nonzero(erase_mask, as_tuple=False).flatten().tolist():
                top = torch.randint(
                    0, self.image_size - erase_size + 1, (), device=images.device
                ).item()
                left = torch.randint(
                    0, self.image_size - erase_size + 1, (), device=images.device
                ).item()
                images[index, top : top + erase_size, left : left + erase_size, :] = 0.0

        return images

    def spatial_augment_features(self, features: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return features
        images = features.reshape(-1, self.image_size, self.image_size, self.channels)
        images = self._augment_images(images)
        return images.reshape(features.shape[0], -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self._augment_features(features)
        images = features.reshape(-1, self.image_size, self.image_size, self.channels)
        patches = images.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        patches = patches.contiguous().reshape(
            images.shape[0], -1, self.channels * self.patch_size * self.patch_size
        )
        patches = self.patch_projection(patches) + self.position_embedding
        patches = self.blocks(patches)
        patches = self.norm(patches)
        features = self.token_pool(patches.transpose(1, 2)).squeeze(-1)
        return self.head(features)


class MLPMixerClassifier(BaseClassifier):
    """MLP-Mixer classifier trained with the same simple SGD loop."""

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        image_size: int = 64,
        channels: int = 3,
        patch_size: int = 4,
        hidden_size: int = 256,
        depth: int = 4,
        expansion: int = 2,
        token_mlp_size: int = 128,
        channel_mlp_size: int = 0,
        dropout: float = 0.15,
        epochs: int = 100,
        batch_size: int = 512,
        learning_rate: float = 1e-2,
        weight_decay: float = 0.0,
        optimizer_name: str = "sgd",
        label_smoothing: float = 0.0,
        feature_noise: float = 0.0,
        feature_drop: float = 0.0,
        crop_padding: int = 4,
        hflip_prob: float = 0.5,
        erase_prob: float = 0.1,
        erase_scale: float = 0.15,
        mixup_alpha: float = 0.1,
        ema_decay: float = 0.99,
        warmup_epochs: int = 10,
        grad_clip: float = 1.0,
        drop_path: float = 0.05,
        layer_scale: float = 0.1,
        patience: int = 5,
        device: str = "cpu",
    ) -> None:
        expected_size = image_size * image_size * channels
        if input_size != expected_size:
            raise ValueError(
                f"MLP-Mixer needs flat image features with size {expected_size}, "
                f"got {input_size}. Check --image-size, --channels, scaler, and PCA."
            )

        self.input_size = input_size
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.label_smoothing = label_smoothing
        self.mixup_alpha = mixup_alpha
        self.ema_decay = ema_decay
        self.warmup_epochs = warmup_epochs
        self.grad_clip = grad_clip
        self.patience = patience
        self.device = torch.device(device)
        mixer_channel_mlp_size = (
            channel_mlp_size if channel_mlp_size > 0 else hidden_size * expansion
        )

        self.network = MLPMixerNet(
            image_size=image_size,
            channels=channels,
            patch_size=patch_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            depth=depth,
            token_mlp_size=token_mlp_size,
            channel_mlp_size=mixer_channel_mlp_size,
            dropout=dropout,
            feature_noise=feature_noise,
            feature_drop=feature_drop,
            crop_padding=crop_padding,
            hflip_prob=hflip_prob,
            erase_prob=erase_prob,
            erase_scale=erase_scale,
            drop_path=drop_path,
            layer_scale=layer_scale,
        ).to(self.device)
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self._adamw_parameter_groups(weight_decay),
                lr=learning_rate,
            )
        elif optimizer_name == "sgd":
            self.optimizer = torch.optim.SGD(
                self.network.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        self.ema_state = self._copy_state() if ema_decay > 0 else None

    def _adamw_parameter_groups(self, weight_decay: float) -> list[dict[str, object]]:
        decay_params = []
        no_decay_params = []
        for name, parameter in self.network.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                parameter.ndim <= 1
                or name.endswith(".bias")
                or "norm" in name.lower()
                or "position_embedding" in name
                or "token_scale" in name
                or "channel_scale" in name
            ):
                no_decay_params.append(parameter)
            else:
                decay_params.append(parameter)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def _copy_state(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().clone()
            for key, value in self.network.state_dict().items()
        }

    @torch.no_grad()
    def _update_ema(self) -> None:
        if self.ema_state is None:
            return
        for key, value in self.network.state_dict().items():
            self.ema_state[key].mul_(self.ema_decay).add_(
                value.detach(),
                alpha=1.0 - self.ema_decay,
            )

    def _load_network_state(self, state: dict[str, torch.Tensor]) -> None:
        self.network.load_state_dict(
            {key: value.to(self.device) for key, value in state.items()}
        )

    def _mixup_batch(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        if self.mixup_alpha <= 0 or len(features) < 2:
            return features, labels, labels, 1.0
        distribution = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = float(distribution.sample().item())
        permutation = torch.randperm(len(features), device=features.device)
        mixed_features = lam * features + (1.0 - lam) * features[permutation]
        return mixed_features, labels, labels[permutation], lam

    def _lr_for_epoch(self, epoch: int) -> float:
        if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
            return self.learning_rate * epoch / self.warmup_epochs
        cosine_epochs = max(1, self.epochs - self.warmup_epochs)
        cosine_epoch = max(0, epoch - self.warmup_epochs - 1)
        return self.learning_rate * 0.5 * (
            1.0 + math.cos(math.pi * cosine_epoch / cosine_epochs)
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        X = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_validation = (
            torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
            if X_val is not None
            else X
        )
        validation_labels = y_val if y_val is not None else y_train

        best_top1 = -1.0
        best_state = None
        stale_epochs = 0

        for epoch in range(1, self.epochs + 1):
            self.current_lr = self._lr_for_epoch(epoch)
            for group in self.optimizer.param_groups:
                group["lr"] = self.current_lr

            self.network.train()
            total_loss = 0.0
            indices = torch.randperm(len(X), device=self.device)
            for start in range(0, len(X), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                batch_features = self.network.spatial_augment_features(X[batch_indices])
                batch_features, labels_a, labels_b, lam = self._mixup_batch(
                    batch_features,
                    y[batch_indices],
                )
                logits = self.network(batch_features)
                loss_a = F.cross_entropy(
                    logits,
                    labels_a,
                    label_smoothing=self.label_smoothing,
                )
                loss_b = F.cross_entropy(
                    logits,
                    labels_b,
                    label_smoothing=self.label_smoothing,
                )
                loss = lam * loss_a + (1.0 - lam) * loss_b
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(),
                        self.grad_clip,
                    )
                self.optimizer.step()
                self._update_ema()
                total_loss += loss.item() * len(batch_indices)

            validation_scores = self._predict_tensor(X_validation, use_ema=True)
            metrics = classification_metrics(validation_scores, validation_labels)
            print(
                f"Epoch {epoch:03d}/{self.epochs} | lr: {self.current_lr:.2e} | "
                f"loss: {total_loss / len(X):.4f} | "
                f"val top-1: {metrics['top1']:.2f}% | "
                f"val top-3: {metrics['top3']:.2f}% | "
                f"val top-5: {metrics['top5']:.2f}%"
            )

            if metrics["top1"] > best_top1:
                best_top1 = metrics["top1"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in (
                        self.ema_state if self.ema_state is not None else self.network.state_dict()
                    ).items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        if best_state is not None:
            self._load_network_state(best_state)
            self.ema_state = self._copy_state() if self.ema_decay > 0 else None

    @torch.inference_mode()
    def _predict_tensor(self, features: torch.Tensor, use_ema: bool = True) -> np.ndarray:
        original_state = None
        if use_ema and self.ema_state is not None:
            original_state = self._copy_state()
            self._load_network_state(self.ema_state)

        self.network.eval()
        score_batches = []
        for start in range(0, len(features), self.batch_size):
            logits = self.network(features[start : start + self.batch_size])
            score_batches.append(logits.cpu())

        if original_state is not None:
            self._load_network_state(original_state)

        return torch.cat(score_batches).numpy()

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        features = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return self._predict_tensor(features)
