#!/bin/zsh
# Test a trained run again; writes experiments/<run>/inference_<timestamp>/.
#   launch/inference.sh experiments/<run> [inference.split=val ...]
set -e
cd "${0:A:h}/.."
run=$1; shift
uv run scripts/inference.py "$run/parameters.yaml" "$@"
