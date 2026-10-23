"""Segment every galaxy and measure its morphology: data/segmaps.npy + data/morphology.npz.

    uv run scripts/segment.py config/segment.yaml
"""

from __future__ import annotations

from net.data.morphology import prepare
from net.utils.config import load_config, resolve
from net.utils.logger import setup_logger


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(argv)
    setup_logger()
    prepare(resolve(cfg["images"]), resolve(cfg["split"]), resolve(cfg["segmaps"]), resolve(cfg["morphology"]),
            cfg["segmentation"], cfg["workers"])


if __name__ == "__main__":
    main()
