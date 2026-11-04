"""Dependency-light binary anomaly metrics with exact tie-aware ROC/AP."""

from __future__ import annotations

from typing import Any

import numpy as np


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Return ROC-AUC, average precision and thresholded classification metrics.

    AUC and AP are undefined when there is only one label class.
    """
    truth = np.asarray(labels).ravel()
    values = np.asarray(scores, dtype=np.float64).ravel()
    if truth.size == 0 or truth.shape != values.shape:
        raise ValueError("labels and scores must be equally sized and nonempty")
    if not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    if not np.isin(truth, [0, 1, False, True]).all():
        raise ValueError("labels must be binary")

    truth = truth.astype(bool)
    predicted = values >= threshold
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    tn = int(np.count_nonzero(~predicted & ~truth))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    positives = int(truth.sum())
    negatives = int(truth.size - positives)
    auc: float | None = None
    ap: float | None = None
    if positives and negatives:
        order = np.argsort(-values, kind="stable")
        ranked_values = values[order]
        ranked_truth = truth[order]
        # Evaluate at distinct score boundaries: ties must enter simultaneously.
        starts = np.r_[0, 1 + np.flatnonzero(ranked_values[1:] != ranked_values[:-1])]
        group_tp = np.add.reduceat(ranked_truth.astype(np.int64), starts)
        group_size = np.diff(np.r_[starts, truth.size])
        cum_tp = np.cumsum(group_tp, dtype=np.int64)
        cum_fp = np.cumsum(group_size - group_tp, dtype=np.int64)
        tpr = np.r_[0.0, cum_tp / positives]
        fpr = np.r_[0.0, cum_fp / negatives]
        auc = float(np.sum((tpr[1:] + tpr[:-1]) * np.diff(fpr) / 2))
        ap = float(np.sum((group_tp / positives) * (cum_tp / (cum_tp + cum_fp))))
    return {
        "n": int(truth.size),
        "positive": positives,
        "negative": negatives,
        "roc_auc": auc,
        "average_precision": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
