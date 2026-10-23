"""Segmentation, morphology measurements, and a training run with both (4-channel input + fused measurements)."""

import numpy as np
import pandas as pd

from net.data.morphology import NAMES, measure, prepare
from net.data.segmentation import GALAXY, OTHER, SKY, segment
from net.utils.config import ROOT
from tests.test_pipeline import data_overrides, only_run, script


def synthetic(axis_ratio=0.5):
    """Elliptical galaxy at the centre, a star in a corner, noisy sky."""
    yy, xx = np.indices((256, 256))
    galaxy = 150 * np.exp(-(((xx - 128) / 25) ** 2 + ((yy - 128) / (25 * axis_ratio)) ** 2))
    star = 200 * np.exp(-((xx - 40) ** 2 + (yy - 40) ** 2) / 8)
    img = 30 + galaxy + star + np.random.default_rng(0).normal(0, 3, (256, 256))
    return np.repeat(np.clip(img, 0, 255).astype(np.uint8)[..., None], 3, -1)


def test_segment_and_measure():
    seg = segment(synthetic())
    assert seg[128, 128] == GALAXY and seg[40, 40] == OTHER and seg[5, 250] == SKY
    x = measure(synthetic(), seg)
    assert x.shape == (len(NAMES),) and np.isfinite(x).all()
    round_ = measure(synthetic(1.0), segment(synthetic(1.0)))
    a2 = NAMES.index("A2_r1")
    assert x[a2] > 3 * round_[a2]  # elongated galaxy: much stronger m=2 mode than a round one
    assert (segment(np.full((256, 256, 3), 32, np.uint8)) == SKY).all()  # blank image


def test_train_with_segmentation_and_morphology(fake, tmp_path):
    seg, morph = tmp_path / "segmaps.npy", tmp_path / "morphology.npz"
    prepare(fake["npy"], fake["splits"] / "galaxy10.csv", seg, morph, {}, workers=2)
    assert np.load(morph)["X"].shape[1] == len(NAMES) and np.isfinite(np.load(morph)["X"]).all()
    out = tmp_path / "runs"
    script("main")([str(ROOT / "config/zoobot_seg_morph.yaml"), *data_overrides(fake, out, 16), "device=cpu",
                    "model.name=test_resnet", "model.pretrained=false", "train.epochs=1", "data.batch_size=8",
                    "data.num_workers=0", f"data.segmaps={seg}", f"data.morphology={morph}"])
    (test,) = only_run(out).glob("inference_*")
    assert len(pd.read_csv(test / "predictions.csv")) == 16
