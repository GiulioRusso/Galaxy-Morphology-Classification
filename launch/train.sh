#!/bin/zsh
# Train + test one or more configs in sequence, then regenerate the README tables.
#   launch/train.sh config/handcrafted_hgb.yaml config/resnet50_*.yaml   # step 1: baseline + imbalance study
#   (set loss/sampler of convnext_tiny.yaml and zoobot.yaml to the resnet50 winner by Val macro-F1 in the README)
#   launch/train.sh config/convnext_tiny.yaml config/zoobot.yaml        # step 2
# Each run writes everything to experiments/<timestamp>_<config name>/ (follow it: tail -f experiments/*/logs/train.log).
# Mac: plugged in, Low Power Mode off. caffeinate keeps it awake; a run is killed if memory gets low,
# so the Mac never freezes. Stop: Ctrl-C.
cd "${0:A:h}/.."
caffeinate -i -w $$ &   # keep the Mac awake while this script runs
for cfg in "$@"; do
  echo "$(date '+%F %T') start $cfg"
  uv run scripts/main.py "$cfg" & pid=$!
  while kill -0 $pid 2>/dev/null; do
    lvl=$(sysctl -n kern.memorystatus_vm_pressure_level)
    free=$(memory_pressure -Q | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
    if (( lvl >= 4 || free < 12 )); then echo "memory low (pressure $lvl, free $free%): killing $cfg"; pkill -P $pid; kill $pid; fi
    sleep 5
  done
  wait $pid; rc=$?; echo "$(date "+%F %T") exit $rc $cfg"
done
uv run scripts/make_results_table.py
