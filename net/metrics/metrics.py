"""Classification metrics from labels + class probabilities; keys become history.csv / metrics.csv columns."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from net.data.dataset import CLASSES, SUPERCLASS, SUPERCLASSES


def ece(y, probs, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def compute_metrics(y, probs) -> dict[str, float]:
    """macro-F1 (primary), balanced / top-1 accuracy, cross-entropy, ECE, 4-superclass scores, per-class F1."""
    y, probs = np.asarray(y), np.asarray(probs, dtype=np.float64)
    pred = probs.argmax(1)
    super_map = np.array(SUPERCLASS)
    super_probs = np.stack([probs[:, super_map == s].sum(1) for s in range(len(SUPERCLASSES))], 1)
    ys, ps = super_map[y], super_probs.argmax(1)
    out = dict(
        macro_f1=f1_score(y, pred, average="macro", labels=range(len(CLASSES)), zero_division=0),
        balanced_acc=balanced_accuracy_score(y, pred),
        acc=accuracy_score(y, pred),
        nll=float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, None)).mean()),
        ece=ece(y, probs),
        super4_macro_f1=f1_score(ys, ps, average="macro", labels=range(len(SUPERCLASSES)), zero_division=0),
        super4_acc=accuracy_score(ys, ps),
    )
    per_class = f1_score(y, pred, average=None, labels=range(len(CLASSES)), zero_division=0)
    out.update({f"f1_{k}": v for k, v in enumerate(per_class)})
    return {k: float(v) for k, v in out.items()}
