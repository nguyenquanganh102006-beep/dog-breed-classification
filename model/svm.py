from __future__ import annotations

import math

import numpy as np
import torch

from model.base import BaseClassifier


class LinearSVM(BaseClassifier):
    """Multiclass linear SVM trained with mini-batch AdamW."""

    def __init__(
        self,
        num_classes: int,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        c: float = 1.0,
        batch_size: int = 512,
        device: str = "cpu",
    ) -> None:
        self.num_classes = num_classes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.c = c
        self.batch_size = batch_size
        self.device = torch.device(device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        X = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        self.weights = torch.zeros(
            X.shape[1], self.num_classes, device=self.device, requires_grad=True
        )
        self.bias = torch.zeros(self.num_classes, device=self.device, requires_grad=True)
        parameters = [self.weights, self.bias]
        first_moments = [torch.zeros_like(parameter) for parameter in parameters]
        second_moments = [torch.zeros_like(parameter) for parameter in parameters]
        step = 0

        for epoch in range(1, self.epochs + 1):
            current_lr = self.learning_rate * 0.5 * (
                1.0 + math.cos(math.pi * (epoch - 1) / self.epochs)
            )
            permutation = torch.randperm(len(X), device=self.device)
            total_loss = 0.0
            for start in range(0, len(X), self.batch_size):
                indices = permutation[start : start + self.batch_size]
                scores = X[indices] @ self.weights + self.bias
                correct = scores.gather(1, y[indices, None])
                margins = torch.clamp(scores - correct + 1.0, min=0.0)
                margins.scatter_(1, y[indices, None], 0.0)
                hinge_loss = margins.sum(dim=1).mean()
                regularization = 0.5 * torch.sum(self.weights * self.weights)
                loss = regularization + self.c * hinge_loss

                loss.backward()
                step += 1
                self._adamw_step(
                    parameters,
                    first_moments,
                    second_moments,
                    step,
                    current_lr,
                )
                total_loss += loss.item() * len(indices)
            if epoch == 1 or epoch % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch:03d}/{self.epochs} | SVM loss: {total_loss / len(X):.4f}")

        self.weights = self.weights.detach()
        self.bias = self.bias.detach()

    def _adamw_step(
        self,
        parameters: list[torch.Tensor],
        first_moments: list[torch.Tensor],
        second_moments: list[torch.Tensor],
        step: int,
        learning_rate: float,
    ) -> None:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        with torch.no_grad():
            for parameter, first, second in zip(
                parameters, first_moments, second_moments
            ):
                gradient = parameter.grad
                first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
                second.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                corrected_first = first / (1.0 - beta1**step)
                corrected_second = second / (1.0 - beta2**step)
                parameter.addcdiv_(
                    corrected_first,
                    corrected_second.sqrt().add_(epsilon),
                    value=-learning_rate,
                )
                parameter.grad = None

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        return (X_tensor @ self.weights + self.bias).cpu().numpy()
