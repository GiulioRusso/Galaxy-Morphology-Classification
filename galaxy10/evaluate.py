"""Metrics from a run's test_probs.csv -> metrics.csv (+ optional figures).

    python -m galaxy10.evaluate configs/b1_resnet50_wce.yaml [--figures]

--figures writes, per seed: confusion_matrix.png, reliability.png, curves.png (train/val loss, val macro-F1),
examples.png (per class: most confident correct and most confident wrong test images).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from galaxy10.data import CLASSES, SUPERCLASS, SUPERCLASSES

PROB_COLS = [f"p{k}" for k in range(len(CLASSES))]


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


def metrics(y, probs):
    pred = probs.argmax(1)
    super_map = np.array(SUPERCLASS)
    super_probs = np.stack([probs[:, super_map == s].sum(1) for s in range(len(SUPERCLASSES))], 1)
    ys, ps = super_map[y], super_probs.argmax(1)
    out = dict(
        macro_f1=f1_score(y, pred, average="macro"),
        balanced_acc=balanced_accuracy_score(y, pred),
        acc=accuracy_score(y, pred),
        ece=ece(y, probs),
        super4_macro_f1=f1_score(ys, ps, average="macro"),
        super4_acc=accuracy_score(ys, ps),
    )
    per_class = f1_score(y, pred, average=None, labels=range(len(CLASSES)))
    out.update({f"f1_{k}": v for k, v in enumerate(per_class)})
    return out


def load_probs(path):
    df = pd.read_csv(path)
    return df.label.to_numpy(), df[PROB_COLS].to_numpy()


def figures(run_dir, images_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y, probs = load_probs(run_dir / "test_probs.csv")
    out_dir = run_dir

    cm = confusion_matrix(y, probs.argmax(1), labels=range(len(CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.5 else "#333333")
    ax.set_xticks(range(len(CLASSES)), [f"{k}" for k in range(len(CLASSES))])
    ax.set_yticks(range(len(CLASSES)), [f"{k} {c}" for k, c in enumerate(CLASSES)], fontsize=8)
    ax.set_xlabel("predicted class")
    ax.set_title("Confusion matrix (row-normalised, test)")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, 16)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, 14)
    acc = [correct[idx == b].mean() if (idx == b).any() else np.nan for b in range(15)]
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", label="perfect calibration")
    ax.bar(edges[:-1], acc, width=1 / 15, align="edge", color="#3b6fb6", edgecolor="white", lw=2,
           label="accuracy per bin")
    ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1),
           title=f"Reliability diagram (ECE={ece(y, probs):.3f})")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "reliability.png", dpi=120)
    plt.close(fig)

    log = pd.read_csv(run_dir / "train_log.csv")
    if len(log) > 1:  # B0 (gradient boosting) has no epochs
        best = log.val_macro_f1.idxmax()
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
        a1.plot(log.epoch, log.train_loss, color="#3b6fb6", lw=2, label="train (training objective)")
        if "val_loss" in log:
            a1.plot(log.epoch, log.val_loss, color="#d9822b", lw=2, label="val (cross-entropy)")
        a1.set(xlabel="epoch", ylabel="loss", title="Loss")
        a2.plot(log.epoch, log.val_macro_f1, color="#3b6fb6", lw=2, label="val macro-F1")
        if "val_acc" in log:
            a2.plot(log.epoch, log.val_acc, color="#999999", lw=2, label="val accuracy")
        a2.axvline(log.epoch[best], color="#333333", lw=1, ls="--", label=f"best epoch ({log.epoch[best]})")
        a2.set(xlabel="epoch", ylim=(0, 1), title="Validation")
        for a in (a1, a2):
            a.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            a.spines[["top", "right"]].set_visible(False)
            a.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=120)
        plt.close(fig)

    if images_path and Path(images_path).exists():
        images = np.load(images_path, mmap_mode="r")
        ids, pred, conf = pd.read_csv(run_dir / "test_probs.csv").image_id.to_numpy(), probs.argmax(1), probs.max(1)
        k = 3
        fig, axes = plt.subplots(len(CLASSES), 2 * k, figsize=(2 * k * 1.3, len(CLASSES) * 1.45))
        for c, row in enumerate(axes):
            good = np.flatnonzero((y == c) & (pred == c))
            bad = np.flatnonzero((y == c) & (pred != c))
            picks = list(good[np.argsort(-conf[good])][:k]) + [None] * (k - min(k, len(good)))
            picks += list(bad[np.argsort(-conf[bad])][:k]) + [None] * (k - min(k, len(bad)))
            for j, (ax, i) in enumerate(zip(row, picks)):
                ax.set_xticks([]), ax.set_yticks([])
                if i is None:
                    ax.set_frame_on(False)
                    continue
                ax.imshow(images[ids[i]])
                ax.set_title(f"{'✓' if j < k else f'→ {pred[i]}'} {conf[i]:.2f}", fontsize=7)
                for sp in ax.spines.values():
                    sp.set_color("#3b6fb6" if j < k else "#c0392b")
                    sp.set_linewidth(2)
            row[0].set_ylabel(f"{c} {CLASSES[c]}", fontsize=7, rotation=0, ha="right", va="center")
        fig.suptitle("Test images per class: most confident correct (blue) | most confident wrong (red, → predicted)",
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / "examples.png", dpi=110)
        plt.close(fig)


def evaluate_run(run_dir, with_figures=False, images_path=None):
    y, probs = load_probs(run_dir / "test_probs.csv")
    m = metrics(y, probs)
    log = pd.read_csv(run_dir / "train_log.csv")
    m["val_macro_f1"] = log.val_macro_f1.max()
    pd.DataFrame([m]).to_csv(run_dir / "metrics.csv", index=False, float_format="%.4f")
    if with_figures:
        figures(run_dir, images_path)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--figures", action="store_true")
    p.add_argument("--results", default="results")
    p.add_argument("--images", default="data/Galaxy10_DECals.npy", help="for examples.png")
    a = p.parse_args()
    name = yaml.safe_load(open(a.config))["name"]
    for run_dir in sorted((Path(a.results) / name).glob("seed*")):
        m = evaluate_run(run_dir, a.figures, a.images)
        print(f"{run_dir}: macro_f1={m['macro_f1']:.4f} acc={m['acc']:.4f} ece={m['ece']:.4f}")


if __name__ == "__main__":
    main()
