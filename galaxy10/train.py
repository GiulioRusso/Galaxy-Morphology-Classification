"""Train one config over its seeds; write per-run outputs.

    python -m galaxy10.train configs/b1_resnet50.yaml            # all seeds -> results/<name>/seed<N>/
    python -m galaxy10.train configs/b1_resnet50.yaml --debug    # 1 epoch, 128 imgs/split -> outputs/debug/

results/<name>/seed<N>/: train_log.csv, test_probs.csv, metrics.csv (committed)
outputs/<name>/seed<N>/best.pt                                    (not committed)
"""
import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from galaxy10 import models, transforms
from galaxy10.data import CLASSES, Galaxy10
from galaxy10.evaluate import PROB_COLS, evaluate_run

DEFAULTS = dict(h5="data/Galaxy10_DECals.h5", splits="splits", pretrained=True, loss="ce", sampler=False,
                epochs=30, batch_size=64, lr=3e-4, weight_decay=0.05, patience=5, amp=True, num_workers=4,
                limit=None, device=None, seeds=[0, 1, 2])


def pick_device(name=None):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def focal_loss(logits, y, gamma=2.0):
    logp = F.log_softmax(logits.float(), 1).gather(1, y[:, None]).squeeze(1)
    return (-(1 - logp.exp()) ** gamma * logp).mean()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = [F.softmax(model(x.to(device)).float(), 1).cpu() for x, _ in loader]
    return torch.cat(probs).numpy()


def run(cfg, seed, run_dir, ckpt_dir):
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)
    device = pick_device(cfg["device"])
    amp = cfg["amp"] and device.type in ("mps", "cuda")
    split_csv = Path(cfg["splits"]) / f"galaxy10_seed{seed}.csv"
    ds = {s: Galaxy10(split_csv, s, cfg["h5"], transforms.build(s == "train"), cfg["limit"])
          for s in ("train", "val", "test")}

    y = ds["train"].labels
    counts = np.maximum(np.bincount(y, minlength=len(CLASSES)), 1)
    weight = torch.tensor(len(y) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    loss_fn = {"ce": lambda o, t: F.cross_entropy(o.float(), t),
               "weighted_ce": lambda o, t: F.cross_entropy(o.float(), t, weight=weight),
               "focal": focal_loss}[cfg["loss"]]
    sampler = WeightedRandomSampler(1.0 / counts[y], len(y)) if cfg["sampler"] else None
    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
              persistent_workers=cfg["num_workers"] > 0)
    train_dl = DataLoader(ds["train"], shuffle=sampler is None, sampler=sampler, drop_last=True, **kw)
    val_dl, test_dl = (DataLoader(ds[s], **kw) for s in ("val", "test"))

    model = models.build(cfg["model"], len(CLASSES), cfg["pretrained"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log, best, stale = [], -1.0, 0
    for epoch in range(cfg["epochs"]):
        t0, model_loss = time.time(), []
        model.train()
        for x, t in train_dl:
            x, t = x.to(device), t.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(x), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model_loss.append(loss.item())
        sched.step()
        val_f1 = f1_score(ds["val"].labels, predict(model, val_dl, device).argmax(1), average="macro")
        log.append(dict(epoch=epoch, train_loss=np.mean(model_loss), val_macro_f1=val_f1,
                        lr=sched.get_last_lr()[0], seconds=time.time() - t0))
        print(f"[seed {seed}] epoch {epoch}: loss={log[-1]['train_loss']:.4f} val_macro_f1={val_f1:.4f} "
              f"({log[-1]['seconds']:.0f}s)", flush=True)
        if val_f1 > best:
            best, stale = val_f1, 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    pd.DataFrame(log).to_csv(run_dir / "train_log.csv", index=False, float_format="%.5f")

    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device))
    probs = predict(model, test_dl, device)
    out = pd.DataFrame(probs, columns=PROB_COLS)
    out.insert(0, "label", ds["test"].labels)
    out.insert(0, "image_id", ds["test"].ids)
    out.to_csv(run_dir / "test_probs.csv", index=False, float_format="%.5f")
    return evaluate_run(run_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--debug", action="store_true", help="1 epoch on 128 images per split, outputs/debug/")
    a = p.parse_args()
    cfg = {**DEFAULTS, **yaml.safe_load(open(a.config))}
    results, outputs = Path("results"), Path("outputs")
    if a.debug:
        cfg.update(epochs=1, limit=128)
        results = outputs = Path("outputs/debug")
    for seed in a.seeds or cfg["seeds"]:
        run_dir, ckpt_dir = results / cfg["name"] / f"seed{seed}", outputs / cfg["name"] / f"seed{seed}"
        if cfg["model"] == "handcrafted":
            from galaxy10 import features
            m = features.run(cfg, seed, run_dir)
        else:
            m = run(cfg, seed, run_dir, ckpt_dir)
        print(f"{cfg['name']} seed {seed}: test macro_f1={m['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
