import h5py
import numpy as np
import pytest

from galaxy10.data import prepare

N, BLACK, DUP = 200, [5, 77], (10, 30)  # DUP: same sky position, must land in the same split
TWIN = (40, 41)  # pixel-identical with conflicting labels: both dropped


@pytest.fixture(scope="session")
def fake(tmp_path_factory):
    d = tmp_path_factory.mktemp("galaxy10")
    rng = np.random.default_rng(0)
    images = rng.integers(1, 256, size=(N, 256, 256, 3), dtype=np.uint8)
    images[BLACK] = 32
    labels = np.arange(N) % 10
    ra, dec = rng.uniform(0, 360, N), rng.uniform(-30, 30, N)
    ra[DUP[1]], dec[DUP[1]] = ra[DUP[0]] + 1e-4, dec[DUP[0]]  # 0.36 arcsec apart
    labels[DUP[1]] = labels[DUP[0]]
    images[TWIN[1]] = images[TWIN[0]]  # labels 0 and 1 differ
    h5 = d / "fake.h5"
    with h5py.File(h5, "w") as f:
        f["images"], f["ans"], f["ra"], f["dec"] = images, labels, ra, dec
    prepare(h5, d / "splits", seeds=[0])
    return dict(h5=h5, npy=h5.with_suffix(".npy"), splits=d / "splits", images=images, labels=labels)
