from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model.base import BaseClassifier


@dataclass
class Node:
    probabilities: np.ndarray
    feature: int | None = None
    threshold: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None


class DecisionTree(BaseClassifier):
    def __init__(
        self,
        num_classes: int,
        max_depth: int = 10,
        min_samples_split: int = 4,
        max_thresholds: int = 20,
    ) -> None:
        self.num_classes = num_classes
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        self.root = self._build_tree(X_train, y_train, depth=0)

    def _probabilities(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y, minlength=self.num_classes).astype(np.float64)
        return counts / counts.sum()

    @staticmethod
    def _gini(y: np.ndarray) -> float:
        probabilities = np.bincount(y).astype(np.float64) / len(y)
        return float(1.0 - np.sum(probabilities**2))

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None]:
        parent_impurity = self._gini(y)
        best_gain = 0.0
        best_feature = None
        best_threshold = None

        for feature in range(X.shape[1]):
            values = X[:, feature]
            quantiles = np.linspace(0, 1, self.max_thresholds + 2)[1:-1]
            thresholds = np.unique(np.quantile(values, quantiles))
            for threshold in thresholds:
                left_mask = values < threshold
                left_count = int(left_mask.sum())
                if left_count == 0 or left_count == len(y):
                    continue
                right_mask = ~left_mask
                weighted_impurity = (
                    left_count * self._gini(y[left_mask])
                    + right_mask.sum() * self._gini(y[right_mask])
                ) / len(y)
                gain = parent_impurity - weighted_impurity
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = float(threshold)
        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(probabilities=self._probabilities(y))
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or np.unique(y).size == 1
        ):
            return node

        feature, threshold = self._best_split(X, y)
        if feature is None or threshold is None:
            return node

        left_mask = X[:, feature] < threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _predict_one(self, sample: np.ndarray) -> np.ndarray:
        node = self.root
        while node.feature is not None:
            node = node.left if sample[node.feature] < node.threshold else node.right
        return node.probabilities

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        return np.stack([self._predict_one(sample) for sample in X])
