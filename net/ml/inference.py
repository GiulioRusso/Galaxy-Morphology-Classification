"""B0 inference + evaluation: inference(cfg) -> <experiment>/inference_<timestamp>/ (same files as torch runs)."""

from __future__ import annotations

import joblib

from net.data.dataset import read_split
from net.ml.features import features
from net.ml.train import full_probs
from net.training.inference import save_test
from net.utils.config import resolve
from net.utils.experiment import new_experiment


def inference(cfg: dict) -> dict:
    inf = cfg["inference"]
    if not inf["checkpoint"]:
        raise SystemExit("inference.checkpoint is empty: pass a run's parameters.yaml or set inference.checkpoint=...")
    exp = new_experiment(cfg, parent=resolve(inf["checkpoint"]).parent, prefix="inference_")
    clf = joblib.load(resolve(inf["checkpoint"]))
    rows = read_split(cfg, inf["split"])
    probs = full_probs(clf, clf.predict_proba(features(cfg, rows.image_id.to_numpy())))
    return save_test(exp, cfg, rows.image_id.to_numpy(), rows.label.to_numpy(), probs)
