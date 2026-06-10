from __future__ import annotations

import numpy as np
import torch

from model.base import BaseClassifier


class KNN(BaseClassifier):
    def __init__(self, k: int = 5, batch_size: int = 256, device: str = "cpu") -> None:
        self.k = k
        self.batch_size = batch_size
        self.device = torch.device(device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        if not 1 <= self.k <= len(X_train):
            raise ValueError(f"k must be between 1 and {len(X_train)}")
        self.X_train = torch.as_tensor(X_train, dtype=torch.float32, device=self.device)
        self.y_train = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        self.num_classes = int(self.y_train.max().item()) + 1

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        score_batches = []
        for start in range(0, len(X_tensor), self.batch_size):
            distances = torch.cdist(X_tensor[start : start + self.batch_size], self.X_train)
            indices = distances.topk(self.k, largest=False).indices
            labels = self.y_train[indices]
            scores = torch.zeros(
                (len(labels), self.num_classes), dtype=torch.float32, device=self.device
            )
            scores.scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
            score_batches.append(scores.cpu())
        return torch.cat(score_batches).numpy()
