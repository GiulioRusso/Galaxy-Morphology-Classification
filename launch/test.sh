#!/bin/zsh
# Unit tests + lint (same as CI). Quick end-to-end debug run on real data: launch/test.sh --debug
set -e
cd "${0:A:h}/.."
uv run ruff check
uv run pytest -q
if [[ $1 == --debug ]]; then
  uv run scripts/main.py config/resnet50_wce.yaml data.limit=128 train.epochs=1 output_dir=/tmp/galaxy10_debug
fi
