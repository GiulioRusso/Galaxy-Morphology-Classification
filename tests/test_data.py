import numpy as np
import pandas as pd

from galaxy10.data import Galaxy10


def test_dataset_returns_image_and_label_of_its_id(fake):
    ds = Galaxy10(fake["splits"] / "galaxy10_seed0.csv", "test", fake["h5"], transform=lambda x: x)
    split = pd.read_csv(fake["splits"] / "galaxy10_seed0.csv").query("split == 'test'")
    assert len(ds) == len(split)
    for k in (0, len(ds) - 1):
        img, label = ds[k]
        i = ds.ids[k]
        assert label == fake["labels"][i]
        assert np.array_equal(img, fake["images"][i])
