"""Sample transforms: plain functions `(image, label) -> (image, label)`, listed per stage in the config.

    transforms:
      train: [{name: to_tensor}, {name: d4}, {name: random_crop, size: 224}, {name: color_jitter, ...}, {name: normalize}]
      eval:  [{name: to_tensor}, {name: center_crop, size: 224}, {name: normalize}]

Randomness uses torch's RNG, which DataLoader seeds per worker.
"""

from __future__ import annotations

import sys
from functools import partial

import numpy as np
import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def to_tensor(x, y):
    """uint8 HWC array -> float CHW tensor in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().div(255), y


def d4(x, y):
    """Random element of the dihedral group D4: k*90 deg rotation, then optional flip.

    Galaxy cutouts have no preferred orientation or handedness, so all 8 are label-preserving.
    """
    x = torch.rot90(x, int(torch.randint(4, ())), dims=(-2, -1))
    return (x.flip(-1) if torch.rand(()) < 0.5 else x), y


def random_crop(x, y, size: int = 224):
    i, j = (int(torch.randint(s - size + 1, ())) for s in x.shape[-2:])
    return x[..., i:i + size, j:j + size], y


def center_crop(x, y, size: int = 224):
    return TF.center_crop(x, [size]), y


def color_jitter(x, y, brightness: float = 0.0, contrast: float = 0.0):
    return v2.ColorJitter(brightness=brightness, contrast=contrast)(x), y


def normalize(x, y, mean: list = IMAGENET_MEAN, std: list = IMAGENET_STD):
    return TF.normalize(x, mean, std), y


def _apply(transforms: list, x, y):
    for transform in transforms:
        x, y = transform(x, y)
    return x, y


def build_transforms(specs: list[dict] | None):
    """[{name: fn, **kwargs}, ...] -> one callable (x, y) -> (x, y). Picklable, so num_workers > 0 works."""
    module = sys.modules[__name__]
    fns = [partial(getattr(module, s["name"]), **{k: v for k, v in s.items() if k != "name"}) for s in specs or []]
    return partial(_apply, fns)
