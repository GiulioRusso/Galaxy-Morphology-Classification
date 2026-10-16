from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import BLACK, DUP

COMMITTED = sorted(Path(__file__).resolve().parents[1].glob("splits/galaxy10_seed*.csv"))


def check(df):
    assert df.image_id.is_unique
    assert (df.groupby("group").split.nunique() == 1).all(), "a duplicate group spans several splits"
    assert set(df.split) == {"train", "val", "test"}
    for split, g in df.groupby("split"):
        assert g.label.nunique() == 10, f"{split} misses a class"


@pytest.mark.parametrize("path", COMMITTED, ids=[p.name for p in COMMITTED])
def test_committed_splits(path):
    check(pd.read_csv(path))


def test_fake_splits(fake):
    df = pd.read_csv(fake["splits"] / "galaxy10_seed0.csv")
    check(df)
    assert not df.image_id.isin(BLACK).any()
    assert pd.read_csv(fake["splits"] / "dropped.csv").image_id.tolist() == BLACK
    assert df.set_index("image_id").loc[list(DUP), "split"].nunique() == 1
