"""Save / load training checkpoints (best.pt, last.pt; used for resume and inference).

Checkpoints live in <experiment>/checkpoints.nosync/: inside the experiment folder, but the .nosync suffix keeps
them out of iCloud Drive and out of git (*.nosync is ignored).
"""

from __future__ import annotations

from pathlib import Path

import torch

from net.utils.config import resolve

CKPT_DIR = "checkpoints.nosync"


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best: float) -> None:
    torch.save({
        "epoch": epoch,
        "best": best,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
    }, path)


def load_checkpoint(path, device) -> dict:
    """A checkpoint saved above, or a bare state_dict (e.g. external pretrained weights) as {"model": ...}."""
    ckpt = torch.load(resolve(path), map_location=device)
    return ckpt if "model" in ckpt else {"model": ckpt}


def experiment_of(checkpoint) -> Path:
    """Experiment folder a checkpoint belongs to (where its inference_* folders go)."""
    parent = resolve(checkpoint).parent
    return parent.parent if parent.name == CKPT_DIR else parent
