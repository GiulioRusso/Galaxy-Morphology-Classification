"""Galaxy10 torch dataset: rows of a split CSV -> transformed (image, label)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from net.utils.config import resolve

CLASSES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth", "Cigar Shaped Smooth",
    "Barred Spiral", "Unbarred Tight Spiral", "Unbarred Loose Spiral", "Edge-on no Bulge", "Edge-on with Bulge",
]


def read_split(cfg: dict, split: str) -> pd.DataFrame:
    """Rows of one split (train / val / test); data.limit keeps a fixed random subset, for debugging."""
    df = pd.read_csv(resolve(cfg["data"]["split"]))
    df = df[df.split == split]
    limit = cfg["data"]["limit"]
    return df.sample(n=min(limit, len(df)), random_state=0) if limit else df


class Galaxy10(Dataset):
    """image_id -> (image, label), images read from the memory-mapped .npy (see net/data/preprocessing.py).

    segmaps_path: add the segmentation map (net/data/segmentation.py) as a 4th channel (sky 0, other 0.5, galaxy 1).
    morphology: (len(rows), F) array of morphology measurements; items become (image, measurements, label).
    """

    def __init__(self, rows: pd.DataFrame, images_path, transform, segmaps_path=None, morphology=None) -> None:
        self.ids, self.labels = rows.image_id.to_numpy(), rows.label.to_numpy()
        self.images_path, self.transform, self.images = resolve(images_path), transform, None
        self.segmaps_path, self.segmaps = segmaps_path and resolve(segmaps_path), None
        self.morphology = morphology

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        if self.images is None:  # opened lazily, once per DataLoader worker
            self.images = np.load(self.images_path, mmap_mode="r")
            self.segmaps = self.segmaps_path and np.load(self.segmaps_path, mmap_mode="r")
        x = np.array(self.images[self.ids[i]])
        if self.segmaps is not None:
            x = np.concatenate([x, self.segmaps[self.ids[i]][..., None] * np.uint8(127)], -1)
        x, y = self.transform(x, int(self.labels[i]))
        return (x, y) if self.morphology is None else (x, self.morphology[i], y)
