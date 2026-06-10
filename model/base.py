from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseClassifier(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Fit the classifier."""

    @abstractmethod
    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Return one score per class for every sample."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_scores(X).argmax(axis=1)
