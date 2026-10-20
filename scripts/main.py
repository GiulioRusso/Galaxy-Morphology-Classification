"""Train (+ test with inference.after_train).

    uv run scripts/main.py config/b1_resnet50_focal.yaml [key.sub=value ...]
    uv run scripts/main.py config/b1_resnet50_focal.yaml seed=1          # seed also selects the split
    uv run scripts/main.py experiments/<run>/parameters.yaml             # reproduce a run
"""

from __future__ import annotations

from net.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(argv)
    if cfg["framework"] == "ml":
        from net.ml.train import train
    else:
        from net.training.train import train
    train(cfg)


if __name__ == "__main__":
    main()
