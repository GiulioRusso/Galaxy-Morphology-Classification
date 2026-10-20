"""Metrics: macro-F1 (selects the model) and per-class F1 on all 10 classes.
Accuracy is not reported: with imbalanced classes it is dominated by the large ones."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from net.data.dataset import CLASSES


def compute_metrics(y, probs) -> dict[str, float]:
    """Labels + class probabilities (N, 10) -> {macro_f1, f1_0 .. f1_9}."""
    pred, labels = np.asarray(probs).argmax(1), range(len(CLASSES))
    out = {"macro_f1": f1_score(y, pred, average="macro", labels=labels, zero_division=0)}
    out.update({f"f1_{k}": v for k, v in enumerate(f1_score(y, pred, average=None, labels=labels, zero_division=0))})
    return {k: float(v) for k, v in out.items()}
