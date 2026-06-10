from __future__ import annotations

import numpy as np


def top_k_accuracy(scores: np.ndarray, labels: np.ndarray, k: int) -> float:
    k = min(k, scores.shape[1])
    top_k = np.argpartition(scores, -k, axis=1)[:, -k:]
    return float(np.any(top_k == labels[:, None], axis=1).mean() * 100.0)


def classification_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "top1": top_k_accuracy(scores, labels, 1),
        "top3": top_k_accuracy(scores, labels, 3),
        "top5": top_k_accuracy(scores, labels, 5),
    }
