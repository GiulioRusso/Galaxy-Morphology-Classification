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
    """image_id -> (image, label), images read from the memory-mapped .npy (see net/data/preprocessing.py)."""

    def __init__(self, rows: pd.DataFrame, images_path, transform) -> None:
        self.ids, self.labels = rows.image_id.to_numpy(), rows.label.to_numpy()
        self.images_path, self.transform, self.images = resolve(images_path), transform, None

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        if self.images is None:
            self.images = np.load(self.images_path, mmap_mode="r")  # opened lazily, once per DataLoader worker
        return self.transform(np.array(self.images[self.ids[i]]), int(self.labels[i]))
