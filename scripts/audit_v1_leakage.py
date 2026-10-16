"""Audit train/val/test leakage caused by v1 offline augmentation.

DL/dataset/data_augmentation.py walks source images start, start+1, ... of one
class and appends 7 rotated/flipped copies per source to the annotation CSV,
with fixed splits: train x4, validation x1, test x2. So copy row r (in a block
of copies for one class that begins at row first) comes from source row
class_start + (r - first) // 7, where class_start is the first row of that
class in the non-augmented CSV (the script's `start`). Source splits are read
from the augmented CSV itself: its original rows were re-split, so they differ
from the non-augmented CSV, and the augmented CSV is what training used.

Output: one row per augmented copy, plus a summary on stdout.

    python scripts/audit_v1_leakage.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "DL" / "dataset" / "annotations"
OUT = ROOT / "scripts" / "audit_v1_leakage.csv"
COPIES_PER_SOURCE = 7
COPY_SPLITS = ["train"] * 4 + ["validation"] + ["test"] * 2  # from data_augmentation.py


def audit(name):
    base = pd.read_csv(ANN / f"{name}.csv")
    aug = pd.read_csv(ANN / f"{name}-augmented.csv")
    sources, copies = aug.iloc[: len(base)], aug.iloc[len(base):]
    assert sources[["FILENAME", "CLASS"]].equals(base[["FILENAME", "CLASS"]]), f"{name}: rows mismatch"
    rows = []
    for cls, block in copies.groupby("CLASS", sort=False):
        assert (block.index.to_series().diff().dropna() == 1).all(), "copy block not contiguous"
        class_start = sources.index[sources.CLASS == cls][0]
        for r, (fname, _, split) in zip(block.index, block.itertuples(index=False)):
            k = r - block.index[0]
            assert split == COPY_SPLITS[k % COPIES_PER_SOURCE], f"{fname}: unexpected split pattern"
            src = sources.iloc[class_start + k // COPIES_PER_SOURCE]
            assert src.CLASS == cls, f"{fname}: source outside class block"
            rows.append(dict(annotation=name, copy=fname, source=src.FILENAME, cls=cls,
                             copy_split=split, source_split=src.SPLIT))
    return pd.DataFrame(rows)


def main():
    df = pd.concat([audit(n) for n in ["all-10", "all-4", "all-3"]], ignore_index=True)
    df["cross_split"] = df.copy_split != df.source_split
    df.to_csv(OUT, index=False)
    for name, g in df.groupby("annotation", sort=False):
        n_test = (pd.read_csv(ANN / f"{name}-augmented.csv").SPLIT == "test").sum()
        groups = g.groupby("source")
        # a galaxy leaks if its source + copies span more than one split (always true: copies span 3)
        leaking = groups.apply(lambda s: len(set(s.copy_split) | {s.source_split.iat[0]}) > 1).sum()
        test_copies = (g.copy_split == "test").sum()
        print(f"{name}: {len(g)} copies of {groups.ngroups} galaxies; "
              f"{g.cross_split.sum()} source/copy pairs in different splits; "
              f"{leaking} galaxies span several splits; "
              f"{test_copies}/{n_test} test rows are copies of galaxies also in train")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
