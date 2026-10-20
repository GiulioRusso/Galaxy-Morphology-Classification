"""Aggregate experiments/*/ (mean ± std over seeds) into README.md between markers.

    uv run scripts/make_results_table.py

Per (name, seed) the latest training run with a test run is used: test metrics from its latest
inference_*/metrics.csv, validation macro-F1 = best val_macro_f1 in its history.csv.

Markers: DATASET (class counts from splits/), HEADLINE, RESULTS, PERCLASS.
The headline/per-class model is the one with the best mean
*validation* macro-F1, so test numbers never drive model selection.
"""
import re
from pathlib import Path

import pandas as pd
import yaml

from net.data.dataset import CLASSES

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
COLS = [("macro_f1", "Test macro-F1"), ("balanced_acc", "Balanced acc."), ("acc", "Top-1 acc."),
        ("ece", "ECE ↓"), ("super4_macro_f1", "4-superclass macro-F1"), ("val_macro_f1", "Val macro-F1")]


def load(experiments=ROOT / "experiments"):
    runs = {}
    for run in sorted(experiments.glob("*/parameters.yaml")):  # timestamped names: later runs overwrite earlier
        tests = sorted(run.parent.glob("inference_*/metrics.csv"))
        cfg = yaml.safe_load(run.read_text())
        if tests and cfg["inference"]["split"] == "test" and not cfg["data"]["limit"]:  # skip debug runs
            val = pd.read_csv(run.parent / "history.csv").val_macro_f1.max()
            runs[cfg["name"], cfg["seed"]] = pd.read_csv(tests[-1]).assign(
                name=cfg["name"], seed=cfg["seed"], title=cfg.get("title", cfg["name"]), val_macro_f1=val)
    return pd.concat(runs.values(), ignore_index=True) if runs else None


def title(df, name):
    return df.loc[df.name == name, "title"].iloc[-1]


def pm(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def render(df):
    if df is None:
        empty = "_No results yet: train with `scripts/main.py`, then run `scripts/make_results_table.py`._"
        return empty, empty, empty
    groups = df.groupby("name")
    order = sorted(groups.groups, key=lambda n: title(df, n))
    lines = ["| Model | Seeds | " + " | ".join(h for _, h in COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
    for n in order:
        g = groups.get_group(n)
        lines.append(f"| {title(df, n)} | {len(g)} | " + " | ".join(pm(g[c]) for c, _ in COLS) + " |")
    best = groups.val_macro_f1.mean().idxmax()
    g = groups.get_group(best)
    headline = (f"**Best model (by validation macro-F1): {title(df, best)}. Test macro-F1 {pm(g.macro_f1)} "
                f"on all 10 classes ({len(g)} seeds).**")
    per = ["| Class | F1 |", "|---|---|"] + [f"| {k} {c} | {pm(g[f'f1_{k}'])} |" for k, c in enumerate(CLASSES)]
    return headline, "\n".join(lines), "\n".join(per)


def dataset_table(splits=ROOT / "splits"):
    kept = pd.read_csv(splits / "galaxy10_seed0.csv")
    dropped = pd.read_csv(splits / "dropped.csv")
    n_kept = kept.label.value_counts().reindex(range(len(CLASSES)), fill_value=0)
    n_drop = dropped.label.value_counts().reindex(range(len(CLASSES)), fill_value=0)
    lines = ["| Class | Name | Release | Used |", "|---|---|---|---|"]
    lines += [f"| {k} | {c} | {n_kept[k] + n_drop[k]} | {n_kept[k]} |" for k, c in enumerate(CLASSES)]
    lines.append(f"| | **Total** | **{len(kept) + len(dropped)}** | **{len(kept)}** |")
    reasons = ", ".join(f"{n} {r}" for r, n in dropped.reason.value_counts().items())
    return "\n".join(lines) + f"\n\nDropped ({len(dropped)}): {reasons}. List: `splits/dropped.csv`."


def write(text, marker, body):
    pat = re.compile(rf"(<!-- {marker}:START -->\n).*?(<!-- {marker}:END -->)", re.S)
    assert pat.search(text), f"README is missing the {marker} markers"
    return pat.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main():
    text = README.read_text()
    text = write(text, "DATASET", dataset_table())
    for marker, body in zip(["HEADLINE", "RESULTS", "PERCLASS"], render(load())):
        text = write(text, marker, body)
    README.write_text(text)
    print(f"updated {README.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
