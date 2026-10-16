"""Galaxy10 DECaLS: black-image filter, duplicate grouping, leakage-free splits, torch Dataset.

    python -m galaxy10.data            # writes splits/galaxy10_seed{0,1,2}.csv and splits/dropped.csv

image_id is the row index in Galaxy10_DECals.h5 and never changes (no renaming).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from torch.utils.data import Dataset

CLASSES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar Shaped Smooth",
    "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on no Bulge", "Edge-on with Bulge",
]
SUPERCLASSES = ["Disturbed/Merging", "Smooth", "Spiral", "Edge-on"]
SUPERCLASS = [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]  # class -> superclass (v1 4-class mapping)
DUPLICATE_RADIUS_ARCSEC = 2.0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def find_black(images, chunk=512):
    """Indices of images whose pixels are all zero."""
    return np.array([i + j for i in range(0, len(images), chunk)
                     for j, img in enumerate(images[i:i + chunk]) if img.max() == 0], dtype=int)


def duplicate_groups(ra, dec, radius_arcsec=DUPLICATE_RADIUS_ARCSEC):
    """Group id per image: images within radius of each other (transitively) share a group."""
    coords = np.radians(np.stack([dec, ra], 1))
    nbrs = BallTree(coords, metric="haversine").query_radius(coords, np.radians(radius_arcsec / 3600))
    rows = np.repeat(np.arange(len(nbrs)), [len(n) for n in nbrs])
    graph = coo_matrix((np.ones(len(rows)), (rows, np.concatenate(nbrs))), shape=(len(ra), len(ra)))
    return connected_components(graph, directed=False)[1]


def make_split(labels, groups, seed):
    """Stratified 70/15/15 split over duplicate groups (a group's label = its first member's)."""
    g = pd.DataFrame(dict(label=labels, group=groups)).groupby("group").label.first()
    train, rest = train_test_split(g.index, test_size=1 - SPLIT_FRACTIONS[0], stratify=g, random_state=seed)
    val_frac = SPLIT_FRACTIONS[1] / (1 - SPLIT_FRACTIONS[0])
    val, test = train_test_split(rest, train_size=val_frac, stratify=g[rest], random_state=seed)
    split_of = pd.Series("train", index=g.index)
    split_of[val], split_of[test] = "val", "test"
    return split_of[groups].to_numpy()


def prepare(h5_path, splits_dir, seeds):
    with h5py.File(h5_path, "r") as f:
        labels, ra, dec = f["ans"][:].astype(int), f["ra"][:], f["dec"][:]
        black = find_black(f["images"])
    Path(splits_dir).mkdir(exist_ok=True)
    keep = np.setdiff1d(np.arange(len(labels)), black)
    pd.DataFrame(dict(image_id=black, label=labels[black], reason="all-black")).to_csv(
        Path(splits_dir) / "dropped.csv", index=False)
    print(f"dropped {len(black)} all-black images: {black.tolist()}")

    groups = duplicate_groups(ra[keep], dec[keep])
    sizes = np.bincount(groups)
    print(f"{len(keep)} images, {len(sizes)} groups, {(sizes > 1).sum()} duplicate groups "
          f"covering {sizes[sizes > 1].sum()} images (radius {DUPLICATE_RADIUS_ARCSEC}\")")
    for seed in seeds:
        df = pd.DataFrame(dict(image_id=keep, label=labels[keep], group=groups,
                               split=make_split(labels[keep], groups, seed)))
        df.to_csv(Path(splits_dir) / f"galaxy10_seed{seed}.csv", index=False)
        print(f"seed {seed}:", df.split.value_counts().to_dict())


class Galaxy10(Dataset):
    """(image, label) for one split; reads lazily from the h5 so DataLoader workers each open their own handle."""

    def __init__(self, split_csv, split, h5_path, transform, limit=None):
        df = pd.read_csv(split_csv)
        df = df[df.split == split]
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=0)
        self.ids, self.labels = df.image_id.to_numpy(), df.label.to_numpy()
        self.h5_path, self.transform, self.images = h5_path, transform, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.images is None:
            self.images = h5py.File(self.h5_path, "r")["images"]
        return self.transform(self.images[self.ids[i]]), int(self.labels[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/Galaxy10_DECals.h5")
    p.add_argument("--splits", default="splits")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    prepare(a.h5, a.splits, a.seeds)


if __name__ == "__main__":
    main()
