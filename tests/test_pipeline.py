"""End-to-end smoke tests: scripts/main.py trains + tests into one experiment folder; inference re-runs."""

import runpy

import pandas as pd

from net.utils.config import ROOT


def script(name):
    return runpy.run_path(str(ROOT / "scripts" / f"{name}.py"))["main"]


def only_run(out):
    (run,) = [p for p in out.iterdir() if p.is_dir()]
    return run


def data_overrides(fake, out, limit):
    return [f"output_dir={out}", f"data.images={fake['npy']}", f"data.limit={limit}",
            f"data.split={fake['splits']}/galaxy10.csv"]


def test_dl_train_test_and_inference(fake, tmp_path):
    script("main")([str(ROOT / "config/b1_resnet50_wce.yaml"), *data_overrides(fake, tmp_path, 16), "device=cpu",
                    "model.name=test_resnet", "model.pretrained=false", "train.epochs=2", "data.batch_size=8",
                    "data.num_workers=0"])
    run = only_run(tmp_path)
    assert run.name.endswith("_b1_resnet50_wce")  # suffix = config file name
    files = {str(p.relative_to(run)) for p in run.rglob("*")}
    assert {"parameters.yaml", "logs/train.log", "logs/log_epoch.csv", "logs/log_steps.csv",
            "checkpoints.nosync/best.pt", "checkpoints.nosync/last.pt", "plots/loss.png", "plots/macro_f1.png",
            "plots/loss_steps.png", "plots/per_class_f1.png", "plots/val_confusion_matrix.png"} <= files
    epochs = pd.read_csv(run / "logs" / "log_epoch.csv")
    assert epochs.epoch.tolist() == [1, 2]
    assert not any(c.startswith("train_") and c != "train_loss" for c in epochs)  # F1 is computed on val only
    assert pd.read_csv(run / "logs" / "log_steps.csv").step.tolist() == [1, 2, 3, 4]

    (test,) = run.glob("inference_*")
    assert {"parameters.yaml", "logs/test.log", "logs/log_test.csv", "predictions.csv", "plots/confusion_matrix.png",
            "plots/examples.png"} <= {str(p.relative_to(test)) for p in test.rglob("*")}
    assert len(pd.read_csv(test / "predictions.csv")) == 16
    assert 0 <= pd.read_csv(test / "logs" / "log_test.csv").macro_f1[0] <= 1

    script("inference")([str(run / "parameters.yaml")])  # parameters.yaml drives inference directly
    assert len(list(run.glob("inference_*"))) == 2


def test_ml_train_and_test(fake, tmp_path):
    script("main")([str(ROOT / "config/b0_handcrafted.yaml"), *data_overrides(fake, tmp_path, 12),
                    "data.workers=2", "model.max_iter=5", "model.early_stopping=false"])
    run = only_run(tmp_path)
    assert {"parameters.yaml", "logs/train.log", "logs/log_epoch.csv", "checkpoints.nosync/model.joblib",
            "plots/loss.png"} <= {
        str(p.relative_to(run)) for p in run.rglob("*")}
    (test,) = run.glob("inference_*")
    assert len(pd.read_csv(test / "predictions.csv")) == 12
