from pathlib import Path

import pandas as pd

from tests.conftest import BLACK, DUP, TWIN

COMMITTED = Path(__file__).resolve().parents[1] / "splits" / "galaxy10.csv"


def check(df):
    assert df.image_id.is_unique
    assert (df.groupby("group").split.nunique() == 1).all(), "a duplicate group spans several splits"
    assert set(df.split) == {"train", "val", "test"}
    for split, g in df.groupby("split"):
        assert g.label.nunique() == 10, f"{split} misses a class"


def test_committed_split():
    check(pd.read_csv(COMMITTED))


def test_fake_splits(fake):
    df = pd.read_csv(fake["splits"] / "galaxy10.csv")
    check(df)
    assert not df.image_id.isin(BLACK + list(TWIN)).any()
    assert pd.read_csv(fake["splits"] / "dropped.csv").image_id.tolist() == sorted(BLACK + list(TWIN))
    assert df.set_index("image_id").loc[list(DUP), "split"].nunique() == 1
