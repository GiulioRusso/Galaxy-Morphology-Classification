"""Torch inference + evaluation: inference(cfg)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from net.data.dataloader import build_loader
from net.data.dataset import CLASSES
from net.metrics.metrics import compute_metrics
from net.training.monitoring import plot_confusion, plot_examples
from net.utils.config import resolve
from net.utils.logger import logger

PROB_COLS = [f"p{k}" for k in range(len(CLASSES))]


def save_test(exp: Path, cfg: dict, ids, y, probs) -> dict:
    """predictions.csv (image_id, label, p0..p9), logs/log_test.csv (macro-F1, per-class F1) and plots/."""
    split = cfg["inference"]["split"]
    out = pd.DataFrame(probs, columns=PROB_COLS)
    out.insert(0, "label", y)
    out.insert(0, "image_id", ids)
    out.to_csv(exp / "predictions.csv", index=False, float_format="%.5f")
    m = compute_metrics(y, probs)
    pd.DataFrame([{"split": split, "n": len(y), **m}]).to_csv(exp / "logs" / "log_test.csv", index=False,
                                                              float_format="%.5f")
    plots = exp / "plots"
    plot_confusion(y, probs, plots / "confusion_matrix.png", f"Confusion matrix (row-normalised, {split})")
    plot_examples(ids, y, probs, np.load(resolve(cfg["data"]["images"]), mmap_mode="r"), plots / "examples.png")
    logger.info(f"{split}: macro_f1={m['macro_f1']:.4f}  per-class F1: "
                + " ".join(f"{m[f'f1_{k}']:.3f}" for k in range(len(CLASSES))))
    return m


def inference(cfg: dict) -> dict:
    """Predict `inference.split` with `inference.checkpoint`.

    Writes <experiment>/inference_<timestamp>/: parameters.yaml, logs/{test.log, log_test.csv}, predictions.csv, plots/.
    """
    import torch

    from net.model.model import Net
    from net.training.builders import get_device, limit_mps_memory
    from net.training.checkpoint import experiment_of, load_checkpoint
    from net.training.routines import run_epoch
    from net.utils.experiment import new_experiment
    from net.utils.seed import seed_everything

    inf = cfg["inference"]
    if not inf["checkpoint"]:
        raise SystemExit("inference.checkpoint is empty: pass a run's parameters.yaml or set inference.checkpoint=...")
    exp = new_experiment(cfg, parent=experiment_of(inf["checkpoint"]), prefix="inference_", log="test.log")
    seed_everything(cfg["seed"], cfg["deterministic"])
    device = get_device(cfg["device"])
    limit_mps_memory(device, cfg["mps_memory_fraction"])

    model = Net(**{**cfg["model"], "pretrained": False}).to(device)
    model.load_state_dict(load_checkpoint(inf["checkpoint"], device)["model"])
    loader = build_loader(cfg, inf["split"])
    amp = cfg["train"]["amp"] and device.type in ("mps", "cuda")
    _, probs, y = run_epoch(model, loader, None, device, amp=amp)
    del model
    if device.type == "mps":
        torch.mps.empty_cache()
    return save_test(exp, cfg, loader.dataset.ids, y, probs)
