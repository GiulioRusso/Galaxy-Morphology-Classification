import pandas as pd

from galaxy10.train import DEFAULTS, run


def test_one_batch_cpu(fake, tmp_path):
    cfg = {**DEFAULTS, "name": "smoke", "model": "test_resnet", "pretrained": False, "device": "cpu",
           "h5": fake["h5"], "splits": fake["splits"], "epochs": 1, "batch_size": 8, "limit": 8,
           "num_workers": 0}
    m = run(cfg, 0, tmp_path / "run", tmp_path / "ckpt")
    assert 0 <= m["macro_f1"] <= 1
    assert len(pd.read_csv(tmp_path / "run" / "test_probs.csv")) == 8
