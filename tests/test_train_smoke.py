import pandas as pd

from galaxy10.train import DEFAULTS, run


def test_one_batch_cpu(fake, tmp_path):
    cfg = {**DEFAULTS, "name": "smoke", "model": "test_resnet", "pretrained": False, "device": "cpu",
           "images": fake["npy"], "splits": fake["splits"], "epochs": 1, "batch_size": 8, "limit": 8,
           "num_workers": 0}
    m = run(cfg, 0, tmp_path / "run", tmp_path / "ckpt")
    assert 0 <= m["macro_f1"] <= 1
    assert len(pd.read_csv(tmp_path / "run" / "test_probs.csv")) == 8


def test_figures(fake, tmp_path):
    from galaxy10.evaluate import evaluate_run

    cfg = {**DEFAULTS, "name": "smoke", "model": "test_resnet", "pretrained": False, "device": "cpu",
           "images": fake["npy"], "splits": fake["splits"], "epochs": 2, "batch_size": 8, "limit": 16,
           "num_workers": 0, "patience": 5}
    run(cfg, 0, tmp_path / "run", tmp_path / "ckpt")
    evaluate_run(tmp_path / "run", with_figures=True, images_path=fake["npy"])
    for f in ("confusion_matrix.png", "reliability.png", "curves.png", "examples.png"):
        assert (tmp_path / "run" / f).stat().st_size > 0
