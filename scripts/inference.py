"""Inference + evaluation on a split (default: test).

    uv run scripts/inference.py experiments/<run>/parameters.yaml [inference.split=val inference.checkpoint=...]
"""

from __future__ import annotations

from net.utils.config import load_config


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(argv)
    if cfg["framework"] == "ml":
        from net.ml.inference import inference
    else:
        from net.training.inference import inference
    inference(cfg)


if __name__ == "__main__":
    main()
