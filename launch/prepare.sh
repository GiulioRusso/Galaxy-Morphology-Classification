#!/bin/zsh
# Download Galaxy10 DECaLS (2.7 GB), build data/Galaxy10_DECals.npy + splits/ (the split is already committed),
# then segment and measure every galaxy (data/segmaps.npy, data/morphology.npz).
#   launch/prepare.sh
set -e
cd "${0:A:h}/.."
uv run scripts/download_data.py
uv run scripts/preprocess.py config/preprocess.yaml
uv run scripts/segment.py config/segment.yaml
