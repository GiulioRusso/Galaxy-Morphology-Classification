"""Galaxy10 DECaLS: image cleaning, overlap grouping, leakage-free splits, torch Dataset.

    python -m galaxy10.data   # data/images.npy + splits/galaxy10_seed{0,1,2}.csv + splits/dropped.csv

image_id is the row index in Galaxy10_DECals.h5 and never changes (no renaming).
The h5 is gzip-chunked across 555 images per chunk, so reading one image decompresses ~100 MB;
images are therefore converted once to an uncompressed uint8 .npy (3.5 GB) that is memory-mapped.

Cleaning (logged to splits/dropped.csv):
- blank images (one constant pixel value) are dropped;
- pixel-identical images are kept once if their labels agree, and all dropped if they conflict
  (in the released file, all 59 identical pairs carry two different labels).
Grouping: cutouts within GROUP_RADIUS_ARCSEC overlap on the sky (a cutout spans 67" or 134") and share pixels
(median correlation 0.94-0.99 measured on the overlap), so they form one group that never spans two splits.
"""
import argparse
import hashlib
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
GROUP_RADIUS_ARCSEC = 60.0
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


def to_npy(h5_images, out):
    """Copy the h5 image array to an uncompressed .npy, one h5 chunk-row at a time."""
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=h5_images.shape)
    step = h5_images.chunks[0] if h5_images.chunks else 512
    for i in range(0, len(arr), step):
        arr[i:i + step] = h5_images[i:i + step]
    arr.flush()


def find_blank(images):
    """Indices of images with a single constant pixel value (no data; 3 in the release, all value 32)."""
    flat = images.reshape(len(images), -1)
    return np.flatnonzero(flat.max(1) == flat.min(1))


def find_identical(images, labels):
    """(index, reason) of identical-pixel images to drop: extra copies, or all copies if labels conflict."""
    digests = pd.Series([hashlib.md5(img.tobytes()).hexdigest() for img in images])
    drop = []
    for _, idx in digests.groupby(digests).groups.items():
        if len(idx) > 1:
            conflict = len(set(labels[idx])) > 1
            drop += [(i, "identical, conflicting labels" if conflict else "identical copy")
                     for i in (idx if conflict else idx[1:])]
    return drop


def overlap_groups(ra, dec, radius_arcsec=GROUP_RADIUS_ARCSEC):
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
    npy = Path(h5_path).with_suffix(".npy")
    with h5py.File(h5_path, "r") as f:
        labels, ra, dec = f["ans"][:].astype(int), f["ra"][:], f["dec"][:]
        if not npy.exists():
            print(f"converting {h5_path} -> {npy}")
            to_npy(f["images"], npy)
    images = np.load(npy, mmap_mode="r")
    dropped = pd.DataFrame([(i, "blank") for i in find_blank(images)] + find_identical(images, labels),
                           columns=["image_id", "reason"]).drop_duplicates("image_id").sort_values("image_id")
    dropped.insert(1, "label", labels[dropped.image_id])
    Path(splits_dir).mkdir(exist_ok=True)
    dropped.to_csv(Path(splits_dir) / "dropped.csv", index=False)
    print("dropped:", dropped.reason.value_counts().to_dict())
    keep = np.setdiff1d(np.arange(len(labels)), dropped.image_id)

    groups = overlap_groups(ra[keep], dec[keep])
    sizes = np.bincount(groups)
    print(f"{len(keep)} images, {len(sizes)} groups, {(sizes > 1).sum()} multi-image groups "
          f"covering {sizes[sizes > 1].sum()} images (radius {GROUP_RADIUS_ARCSEC}\")")
    for seed in seeds:
        df = pd.DataFrame(dict(image_id=keep, label=labels[keep], group=groups,
                               split=make_split(labels[keep], groups, seed)))
        df.to_csv(Path(splits_dir) / f"galaxy10_seed{seed}.csv", index=False)
        print(f"seed {seed}:", df.split.value_counts().to_dict())


class Galaxy10(Dataset):
    """(image, label) for one split, read from the memory-mapped images .npy (see prepare)."""

    def __init__(self, split_csv, split, images_path, transform, limit=None):
        df = pd.read_csv(split_csv)
        df = df[df.split == split]
        if limit:
            df = df.sample(n=min(limit, len(df)), random_state=0)
        self.ids, self.labels = df.image_id.to_numpy(), df.label.to_numpy()
        self.images_path, self.transform, self.images = images_path, transform, None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.images is None:
            self.images = np.load(self.images_path, mmap_mode="r")  # opened lazily, per DataLoader worker
        return self.transform(np.array(self.images[self.ids[i]])), int(self.labels[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", default="data/Galaxy10_DECals.h5")
    p.add_argument("--splits", default="splits")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args()
    prepare(a.h5, a.splits, a.seeds)


if __name__ == "__main__":
    main()
