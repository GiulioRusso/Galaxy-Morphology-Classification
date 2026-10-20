import numpy as np
import pandas as pd

from net.data.dataset import Galaxy10


def test_dataset_returns_image_and_label_of_its_id(fake):
    split = pd.read_csv(fake["splits"] / "galaxy10_seed0.csv").query("split == 'test'")
    ds = Galaxy10(split, fake["npy"], transform=lambda x, y: (x, y))
    assert len(ds) == len(split)
    for k in (0, len(ds) - 1):
        img, label = ds[k]
        i = ds.ids[k]
        assert label == fake["labels"][i]
        assert np.array_equal(img, fake["images"][i])
