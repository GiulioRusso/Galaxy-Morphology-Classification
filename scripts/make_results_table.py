"""Collect experiments/*/ into README.md between markers.

    uv run scripts/make_results_table.py

Per config name, the latest training run with a test run is used: test F1 from its latest
inference_*/logs/log_test.csv, validation macro-F1 = best val_macro_f1 in its logs/log_epoch.csv.
The headline model is the one with the best *validation* macro-F1, so test numbers never drive model selection.

Markers: DATASET (class counts from splits/), HEADLINE, RESULTS, PERCLASS, PROTOCOL.
Runs on splits/galaxy10_random.csv (standard protocol of published work) only enter PROTOCOL, which pairs each
with the same config on the clean split and adds accuracy (computed from predictions.csv, for this table only).
"""
import re
from pathlib import Path

import pandas as pd
import yaml

from net.data.dataset import CLASSES

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
CLEAN, RANDOM = "splits/galaxy10.csv", "splits/galaxy10_random.csv"


def load(experiments=ROOT / "experiments"):
    runs = {}
    for params in sorted(experiments.glob("*/parameters.yaml")):  # timestamped names: later runs overwrite earlier
        run, cfg = params.parent, yaml.safe_load(params.read_text())
        tests = sorted(run.glob("inference_*/logs/log_test.csv"))
        if tests and cfg["inference"]["split"] == "test" and not cfg["data"]["limit"]:  # skip debug runs
            val = pd.read_csv(run / "logs" / "log_epoch.csv").val_macro_f1.max()
            pred = pd.read_csv(tests[-1].parents[1] / "predictions.csv")
            accuracy = (pred.filter(like="p").to_numpy().argmax(1) == pred.label).mean()
            runs[cfg["name"]] = pd.read_csv(tests[-1]).assign(
                name=cfg["name"], title=cfg.get("title", cfg["name"]), val_macro_f1=val, run=run.name,
                data_split=cfg["data"]["split"], accuracy=accuracy)
    return pd.concat(runs.values(), ignore_index=True).sort_values("title") if runs else None


def render(df):
    empty = "_No results yet: run `launch/train.sh`._"
    df = None if df is None else df[df.data_split == CLEAN]
    if df is None or df.empty:
        return empty, empty, empty
    best = df.loc[df.val_macro_f1.idxmax()]
    headline = (f"**Best model (by validation macro-F1): {best.title}. "
                f"Test macro-F1 {best.macro_f1:.3f} on all 10 classes.**")
    lines = ["| Model | Test macro-F1 | Val macro-F1 |", "|---|---|---|"]
    lines += [f"| {r.title} | {r.macro_f1:.3f} | {r.val_macro_f1:.3f} |" for r in df.itertuples()]
    per = ["| Class | " + " | ".join(df.name) + " |", "|---" * (len(df) + 1) + "|"]
    per += [f"| {k} {c} | " + " | ".join(f"{v:.3f}" for v in df[f"f1_{k}"]) + " |" for k, c in enumerate(CLASSES)]
    return headline, "\n".join(lines), "\n".join(per)


def protocol_table(df):
    """Same config on the standard random split vs the clean split: how much the standard protocol inflates scores."""
    rnd = None if df is None else df[df.data_split == RANDOM]
    if rnd is None or rnd.empty:
        return "_No runs on the standard split yet: see Reproduce._"
    clean = df[df.data_split == CLEAN].set_index("name")
    lines = ["| Model | Standard split: accuracy | Standard: macro-F1 | Clean split: accuracy | Clean: macro-F1 |",
             "|---|---|---|---|---|"]
    for r in rnd.itertuples():
        c = clean.loc[r.name.removesuffix("_random_split")] if r.name.removesuffix("_random_split") in clean.index \
            else None
        tail = f"{c.accuracy:.1%} | {c.macro_f1:.3f}" if c is not None else "– | –"
        lines.append(f"| {r.title} | {r.accuracy:.1%} | {r.macro_f1:.3f} | {tail} |")
    return "\n".join(lines)


def dataset_table(splits=ROOT / "splits"):
    kept = pd.read_csv(splits / "galaxy10.csv")
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
    df = load()
    for marker, body in zip(["HEADLINE", "RESULTS", "PERCLASS", "PROTOCOL"], [*render(df), protocol_table(df)]):
        text = write(text, marker, body)
    README.write_text(text)
    print(f"updated {README.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
