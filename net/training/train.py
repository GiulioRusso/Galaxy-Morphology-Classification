"""Torch training loop: train(cfg)."""

from __future__ import annotations

import time

import numpy as np
import torch

from net.data.dataloader import build_loader
from net.data.dataset import CLASSES
from net.metrics.metrics import compute_metrics
from net.training.builders import build_all, free_memory, get_device, limit_mps_memory
from net.training.checkpoint import CKPT_DIR, load_checkpoint, save_checkpoint
from net.training.monitoring import CSVLog, plot_confusion, plot_history, plot_steps
from net.training.routines import run_epoch
from net.utils.config import relative_to_root
from net.utils.experiment import new_experiment, save_parameters
from net.utils.logger import logger
from net.utils.seed import seed_everything


def train(cfg: dict) -> None:
    """Writes experiments/<timestamp>_<name>/: parameters.yaml, logs/{train.log, log_epoch.csv, log_steps.csv},
    plots/ (rewritten every epoch), checkpoints.nosync/{best,last}.pt and, with inference.after_train,
    the test run in inference_<timestamp>/.
    """
    exp = new_experiment(cfg)
    seed_everything(cfg["seed"], cfg["deterministic"])
    device = get_device(cfg["device"])
    limit_mps_memory(device, cfg["mps_memory_fraction"])
    t = cfg["train"]
    amp = t["amp"] and device.type in ("mps", "cuda")

    train_loader, val_loader = build_loader(cfg, "train", persistent=True), build_loader(cfg, "val", persistent=True)
    counts = np.bincount(train_loader.dataset.labels, minlength=len(CLASSES))
    model, loss_fn, optimizer, scheduler = build_all(cfg, device, counts)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    sign = 1 if t["mode"] == "min" else -1  # internally always minimise sign * monitor
    first_epoch, best, stale = 1, float("inf"), 0

    if t["resume"]:
        ckpt = load_checkpoint(t["resume"], device)
        model.load_state_dict(ckpt["model"])
        if t["resume_optimizer"]:  # continue the old run: optimizer, scheduler, epoch counter, best score
            optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt["scheduler"] is not None:
                scheduler.load_state_dict(ckpt["scheduler"])
            first_epoch, best = ckpt["epoch"] + 1, ckpt["best"]
        logger.info(f"resumed from {t['resume']} ({'full state' if t['resume_optimizer'] else 'weights only'}), "
                    f"starting at epoch {first_epoch}")

    ckpt_dir, plots = exp / CKPT_DIR, exp / "plots"
    ckpt_dir.mkdir()
    history, steps = CSVLog(exp / "logs" / "log_epoch.csv"), CSVLog(exp / "logs" / "log_steps.csv")
    logger.info(f"device={device} amp={amp} train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
                f"class counts={counts.tolist()}")
    for epoch in range(first_epoch, t["epochs"] + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer, scaler, amp, steps, epoch)
        val_loss, vp, vy = run_epoch(model, val_loader, loss_fn, device, amp=amp)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               **{f"val_{k}": v for k, v in compute_metrics(vy, vp).items()},
               "lr": lr, "seconds": time.perf_counter() - start}

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(row[t["monitor"]])
        elif scheduler is not None:
            scheduler.step()

        improved = sign * row[t["monitor"]] < best
        if improved:
            best, stale = sign * row[t["monitor"]], 0
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch, best)
            plot_confusion(vy, vp, plots / "val_confusion_matrix.png", f"Validation confusion matrix (epoch {epoch})")
        else:
            stale += 1
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch, best)
        history.write(row)
        plot_history(exp / "logs" / "log_epoch.csv", plots)
        plot_steps(exp / "logs" / "log_steps.csv", plots)
        logger.info(f"epoch {epoch:03d}/{t['epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                    f"val_macro_f1={row['val_macro_f1']:.4f}  lr={lr:.3g}  ({row['seconds']:.0f}s)"
                    + ("  *best*" if improved else ""))
        if stale >= t["patience"]:
            logger.info(f"early stop: no {t['monitor']} improvement for {stale} epochs")
            break
    history.close(), steps.close()

    # point inference at this run's best checkpoint, so parameters.yaml drives both scripts
    cfg["inference"]["checkpoint"] = relative_to_root(ckpt_dir / "best.pt")
    save_parameters(cfg, exp / "parameters.yaml")
    logger.info(f"done: best {t['monitor']}={sign * best:.4f}")
    del train_loader, val_loader, model, optimizer  # stop persistent workers, free GPU memory before testing
    free_memory(device)
    if cfg["inference"]["after_train"]:
        from net.training.inference import inference
        inference(cfg)
    else:
        logger.info(f"inference: uv run scripts/inference.py {relative_to_root(exp / 'parameters.yaml')}")
