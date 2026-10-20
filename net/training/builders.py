"""Device, model, loss, optimizer and scheduler from the config."""

from __future__ import annotations

import torch
from torch import nn

from net.loss import loss as custom_losses
from net.model.model import Net


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build(modules: tuple, spec: dict | None, *args):
    """`{name: X, **kwargs}` → X(*args, **kwargs), X looked up in `modules` in order. None → None."""
    if not spec:
        return None
    kwargs = dict(spec)
    name = kwargs.pop("name")
    for module in modules:
        if hasattr(module, name):
            return getattr(module, name)(*args, **kwargs)
    raise ValueError(f"{name!r} not found in {[m.__name__ for m in modules]}")


def build_all(cfg: dict, device: torch.device, class_counts):
    """→ model, loss_fn, optimizer, scheduler (None when disabled)."""
    model = Net(**cfg["model"]).to(device)
    spec = dict(cfg["loss"])
    if spec.get("weight") == "balanced":  # inverse class frequency, mean 1
        counts = torch.as_tensor(class_counts, dtype=torch.float32).clamp(min=1)
        spec["weight"] = counts.sum() / (len(counts) * counts)
    loss_fn = build((nn, custom_losses), spec).to(device)
    optimizer = build((torch.optim,), cfg["optimizer"], model.parameters())
    scheduler = build((torch.optim.lr_scheduler,), cfg["scheduler"], optimizer)
    return model, loss_fn, optimizer, scheduler


def limit_mps_memory(device: torch.device, fraction: float) -> None:
    """CPU and GPU share RAM on Apple silicon. PyTorch's default MPS limit (1.7x the recommended working set) can
    exceed physical memory, and macOS then freezes instead of raising OOM. Capping it (0.4 ~ 4.3 GB on a 16 GB Mac)
    makes overuse raise an error instead."""
    if device.type == "mps":
        torch.mps.set_per_process_memory_fraction(fraction)


def free_memory(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
