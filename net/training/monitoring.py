"""CSV logs and plots of an experiment. Plots go to <experiment>/plots/ and are overwritten on every call.

Training: logs/log_epoch.csv (one row per epoch), logs/log_steps.csv (one row per optimisation step).
Test:     logs/log_test.csv, predictions.csv + plots/confusion_matrix.png, examples.png.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

from net.data.dataset import CLASSES  # noqa: E402

TRAIN, VAL = "#3b6fb6", "#d9822b"


class CSVLog:
    """Append-only CSV; header from the first row's keys. Kept open, flushed every row."""

    def __init__(self, path: Path) -> None:
        self.path, self.file, self.writer = path, None, None

    def write(self, row: dict) -> None:
        if self.writer is None:
            self.file = open(self.path, "a", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            if self.file.tell() == 0:
                self.writer.writeheader()
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        if self.file:
            self.file.close()


def _save(fig, path: Path) -> None:
    path.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _panel(ax, xlabel, title):
    ax.set(xlabel=xlabel, title=title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


def plot_history(csv_path: Path, plots: Path) -> None:
    """loss.png (train + val), macro_f1.png (val), per_class_f1.png (val), lr.png; best epoch marked."""
    df = pd.read_csv(csv_path)
    x = df.columns[0]
    best = df[x][df.val_macro_f1.idxmax()]
    for name, cols in (("loss", ["train_loss", "val_loss"]), ("macro_f1", ["val_macro_f1"]), ("lr", ["lr"])):
        if not set(cols) <= set(df):
            continue
        fig, ax = plt.subplots(figsize=(6, 3.8))
        for c in cols:
            ax.plot(df[x], df[c], marker=".", lw=2, color=VAL if c.startswith("val_") else TRAIN,
                    label=c.split("_")[0] if c != "lr" else c)
        ax.axvline(best, color="#333333", lw=1, ls="--", label=f"best val macro-F1 ({x} {best})")
        _panel(ax, x, {"loss": "Loss", "macro_f1": "Validation macro-F1", "lr": "Learning rate"}[name])
        ax.legend(frameon=False)
        _save(fig, plots / f"{name}.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.get_cmap("tab10").colors
    for k, name in enumerate(CLASSES):
        ax.plot(df[x], df[f"val_f1_{k}"], lw=1.5, color=colors[k], label=f"{k} {name}")
    ax.axvline(best, color="#333333", lw=1, ls="--")
    _panel(ax, x, "Validation F1 per class")
    ax.set(ylabel="F1", ylim=(0, 1))
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    _save(fig, plots / "per_class_f1.png")


def plot_steps(csv_path: Path, plots: Path, window: int = 50) -> None:
    """Per-step training loss (raw + moving average) -> plots/loss_steps.png."""
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.plot(df.step, df.loss, color=TRAIN, lw=0.5, alpha=0.3, label="per step")
    ax.plot(df.step, df.loss.rolling(window, min_periods=1).mean(), color=TRAIN, lw=2, label=f"mean of {window}")
    for s in df.groupby("epoch").step.min().iloc[1:]:
        ax.axvline(s, color="#cccccc", lw=0.5)
    _panel(ax, "step (grey lines: epochs)", "Training loss per step")
    ax.legend(frameon=False)
    _save(fig, plots / "loss_steps.png")


def plot_confusion(y, probs, path: Path, title: str) -> None:
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
    ax.set(xlabel="predicted class", title=title)
    _save(fig, path)


def plot_examples(ids, y, probs, images, path: Path, k: int = 3) -> None:
    """Per class: the k most confident correct (blue) and the k most confident wrong (red, -> predicted) images."""
    pred, conf = probs.argmax(1), probs.max(1)
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
                sp.set_color(TRAIN if j < k else "#c0392b")
                sp.set_linewidth(2)
        row[0].set_ylabel(f"{c} {CLASSES[c]}", fontsize=7, rotation=0, ha="right", va="center")
    fig.suptitle("Test images per class: most confident correct (blue) | most confident wrong (red, → predicted)",
                 fontsize=9)
    _save(fig, path)
