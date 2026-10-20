#!/bin/zsh
# Download Galaxy10 DECaLS (2.7 GB) and build data/Galaxy10_DECals.npy + splits/ (the split is already committed).
#   launch/prepare.sh
set -e
cd "${0:A:h}/.."
uv run scripts/download_data.py
uv run scripts/preprocess.py config/preprocess.yaml
