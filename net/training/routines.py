"""One pass over a DataLoader, for training or evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, amp=False, steps=None, epoch=0):
    """Trains when optimizer is given, otherwise evaluates. Training writes one row per step to `steps` (CSVLog).

    :return: (mean loss, or nan without loss_fn; class probabilities (N, C) and labels (N,) when evaluating).
    """
    training = optimizer is not None
    model.train(training)
    total, n, probs, labels = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        bar = tqdm(loader, desc="train" if training else "eval", leave=False, dynamic_ncols=True, disable=None)
        for i, (x, y) in enumerate(bar):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
            out = out.float()
            loss = loss_fn(out, y) if loss_fn is not None else torch.tensor(float("nan"))
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                steps.write({"epoch": epoch, "step": (epoch - 1) * len(loader) + i + 1,
                             "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]})
            else:
                probs.append(F.softmax(out, 1).cpu())
                labels.append(y.cpu())
            total, n = total + loss.item() * len(x), n + len(x)
    if training:
        return total / n, None, None
    return total / n, torch.cat(probs).numpy(), torch.cat(labels).numpy()
