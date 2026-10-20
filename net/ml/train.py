"""Handcrafted features + gradient boosting: train(cfg).

Writes experiments/<timestamp>_<name>/: parameters.yaml, logs/{train.log, log_epoch.csv} (one row per boosting
iteration: train/val loss and val F1 from staged predictions), plots/, checkpoints.nosync/model.joblib,
and the test run in inference_*/.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

from net.data.dataset import CLASSES, read_split
from net.metrics.metrics import compute_metrics
from net.ml.features import features
from net.training.checkpoint import CKPT_DIR
from net.training.monitoring import CSVLog, plot_confusion, plot_history
from net.utils.config import relative_to_root
from net.utils.experiment import new_experiment, save_parameters
from net.utils.logger import logger
from net.utils.seed import seed_everything


def full_probs(clf, p):
    """Columns for classes absent from a tiny debug train set stay 0."""
    out = np.zeros((len(p), len(CLASSES)))
    out[:, clf.classes_] = p
    return out


def train(cfg: dict) -> None:
    exp = new_experiment(cfg)
    seed_everything(cfg["seed"])
    tr, va = read_split(cfg, "train"), read_split(cfg, "val")
    X_tr, X_va = features(cfg, tr.image_id.to_numpy()), features(cfg, va.image_id.to_numpy())
    logger.info(f"train={len(tr)} val={len(va)} features={X_tr.shape[1]}")

    # early stopping (if enabled) uses an internal split of the training set; our val split stays untouched
    clf = HistGradientBoostingClassifier(random_state=cfg["seed"], **cfg["model"]).fit(X_tr, tr.label)
    history = CSVLog(exp / "logs" / "log_epoch.csv")
    staged = zip(clf.staged_predict_proba(X_tr), clf.staged_predict_proba(X_va))
    labels = range(len(CLASSES))
    for it, (p_tr, p_va) in enumerate(staged, 1):
        m_va = compute_metrics(va.label, full_probs(clf, p_va))
        history.write({"iteration": it, "train_loss": log_loss(tr.label, full_probs(clf, p_tr), labels=labels),
                       "val_loss": log_loss(va.label, full_probs(clf, p_va), labels=labels),
                       **{f"val_{k}": v for k, v in m_va.items()}})
    history.close()
    plot_history(exp / "logs" / "log_epoch.csv", exp / "plots")
    plot_confusion(va.label.to_numpy(), full_probs(clf, p_va), exp / "plots" / "val_confusion_matrix.png",
                   f"Validation confusion matrix (iteration {it})")
    logger.info(f"done: {clf.n_iter_} iterations, val macro_f1={m_va['macro_f1']:.4f}")

    (exp / CKPT_DIR).mkdir()
    joblib.dump(clf, exp / CKPT_DIR / "model.joblib")
    cfg["inference"]["checkpoint"] = relative_to_root(exp / CKPT_DIR / "model.joblib")
    save_parameters(cfg, exp / "parameters.yaml")
    if cfg["inference"]["after_train"]:
        from net.ml.inference import inference
        inference(cfg)
