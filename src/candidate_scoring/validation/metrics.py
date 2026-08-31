"""Small statistics, implemented here so the package does not depend on scipy/sklearn."""

from __future__ import annotations

import numpy as np


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties averaged."""
    positives, negatives = labels.sum(), (~labels).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _rank(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    # Average ranks within ties so AUC stays correct on discrete scores.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for i, count in enumerate(counts):
        if count > 1:
            mask = inverse == i
            ranks[mask] = ranks[mask].mean()
    return ranks


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 4000, lr: float = 0.2):
    """Plain gradient-descent logistic regression, used only as a cross-check.

    Deliberately not sklearn: this is a diagnostic inside the harness, not a shipped
    model, and pulling in a training framework would invite exactly the confusion
    ADR-0002 exists to prevent.
    """
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)
    x = np.hstack([np.ones((len(x), 1)), x])
    weights = np.zeros(x.shape[1])
    target = y.astype(float)
    for _ in range(steps):
        predictions = 1.0 / (1.0 + np.exp(-x @ weights))
        weights -= lr * (x.T @ (predictions - target)) / len(x)
    return weights[0], weights[1:]
