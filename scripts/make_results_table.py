"""Aggregate results/<name>/seed*/metrics.csv (mean ± std over seeds) into README.md between markers.

    python scripts/make_results_table.py

Markers: HEADLINE, RESULTS, PERCLASS. The headline/per-class model is the one with the best mean
*validation* macro-F1, so test numbers never drive model selection.
"""
import re
from pathlib import Path

import pandas as pd
import yaml

from galaxy10.data import CLASSES

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
COLS = [("macro_f1", "Test macro-F1"), ("balanced_acc", "Balanced acc."), ("acc", "Top-1 acc."),
        ("ece", "ECE ↓"), ("super4_macro_f1", "4-superclass macro-F1"), ("val_macro_f1", "Val macro-F1")]


def load(results=ROOT / "results"):
    rows = []
    for f in sorted(results.glob("*/seed*/metrics.csv")):
        rows.append(pd.read_csv(f).assign(name=f.parts[-3], seed=f.parts[-2]))
    return pd.concat(rows, ignore_index=True) if rows else None


def title(name):
    cfg = ROOT / "configs" / f"{name}.yaml"
    return yaml.safe_load(open(cfg)).get("title", name) if cfg.exists() else name


def pm(s):
    return f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"


def render(df):
    if df is None:
        empty = "_No results yet: run `make train CONFIG=...` then `make table`._"
        return empty, empty, empty
    groups = df.groupby("name")
    order = sorted(groups.groups, key=lambda n: title(n))
    lines = ["| Model | Seeds | " + " | ".join(h for _, h in COLS) + " |", "|---" * (len(COLS) + 2) + "|"]
    for n in order:
        g = groups.get_group(n)
        lines.append(f"| {title(n)} | {len(g)} | " + " | ".join(pm(g[c]) for c, _ in COLS) + " |")
    best = groups.val_macro_f1.mean().idxmax()
    g = groups.get_group(best)
    headline = (f"**Best model (by validation macro-F1): {title(best)}. Test macro-F1 {pm(g.macro_f1)} "
                f"on all 10 classes ({len(g)} seeds).**")
    per = ["| Class | F1 |", "|---|---|"] + [f"| {k} {c} | {pm(g[f'f1_{k}'])} |" for k, c in enumerate(CLASSES)]
    return headline, "\n".join(lines), "\n".join(per)


def write(text, marker, body):
    pat = re.compile(rf"(<!-- {marker}:START -->\n).*?(\n<!-- {marker}:END -->)", re.S)
    assert pat.search(text), f"README is missing the {marker} markers"
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main():
    text = README.read_text()
    for marker, body in zip(["HEADLINE", "RESULTS", "PERCLASS"], render(load())):
        text = write(text, marker, body)
    README.write_text(text)
    print(f"updated {README.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
