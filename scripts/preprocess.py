"""Clean the release, group overlapping cutouts, write the leakage-free splits.

    uv run scripts/download_data.py
    uv run scripts/preprocess.py config/preprocess.yaml
"""

from __future__ import annotations

from net.data.preprocessing import prepare
from net.utils.config import load_config, resolve
from net.utils.logger import setup_logger


def main(argv: list[str] | None = None) -> None:
    cfg = load_config(argv)
    setup_logger()
    prepare(resolve(cfg["h5"]), resolve(cfg["splits_dir"]), cfg["seeds"], cfg["group_radius_arcsec"])


if __name__ == "__main__":
    main()
