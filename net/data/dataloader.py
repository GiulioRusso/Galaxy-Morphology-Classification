"""Build a DataLoader for one split."""

from __future__ import annotations

import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

from net.data import morphology
from net.data.dataset import Galaxy10, read_split
from net.data.transforms.transforms import build_transforms
from net.utils.config import resolve


def build_loader(cfg: dict, split: str, persistent: bool = False) -> DataLoader:
    """train -> shuffled (or class-balanced with data.sampler), `transforms.train`; else `transforms.eval`."""
    d, train = cfg["data"], split == "train"
    transform = build_transforms(cfg["transforms"]["train" if train else "eval"])
    rows = read_split(cfg, split)
    morph = morphology.load(resolve(d["morphology"]), rows.image_id) if d.get("morphology") else None
    dataset = Galaxy10(rows, d["images"], transform, d.get("segmaps"), morph)
    sampler = None
    if train and d["sampler"]:
        counts = np.bincount(dataset.labels)
        sampler = WeightedRandomSampler(1.0 / counts[dataset.labels], len(dataset))
    return DataLoader(dataset, batch_size=d["batch_size"], shuffle=train and sampler is None, sampler=sampler,
                      drop_last=train, num_workers=d["num_workers"],
                      persistent_workers=persistent and d["num_workers"] > 0)
