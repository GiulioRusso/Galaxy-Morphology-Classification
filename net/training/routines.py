"""One pass over a DataLoader, for training or evaluation."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm


def run_epoch(model, loader, loss_fn, device, optimizer=None, scaler=None, amp=False, steps=None, epoch=0):
    """Trains when optimizer is given, otherwise evaluates. Training writes one row per step to `steps` (CSVLog).

    :return: (mean loss or nan without loss_fn, class probabilities (N, C), labels (N,)) as numpy arrays.
    """
    training = optimizer is not None
    model.train(training)
    total, probs, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for x, y in tqdm(loader, desc="train" if training else "eval", leave=False, dynamic_ncols=True, disable=None):
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
            p = F.softmax(out.detach(), 1).cpu()
            if steps is not None:
                steps.write({"epoch": epoch, "step": (epoch - 1) * len(loader) + len(probs) + 1, "loss": loss.item(),
                             "acc": (p.argmax(1) == y.cpu()).float().mean().item(),
                             "lr": optimizer.param_groups[0]["lr"]})
            probs.append(p)
            labels.append(y.cpu())
            total += loss.item() * len(x)
    return total / sum(len(p) for p in probs), torch.cat(probs).numpy(), torch.cat(labels).numpy()
