"""Timestamped experiment folders and the parameters.yaml that reproduces them."""

from __future__ import annotations

import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

from net.utils.config import ROOT, relative_to_root, resolve
from net.utils.logger import logger, setup_logger


def git_state() -> str:
    """Short commit hash, marked (dirty) when there are uncommitted changes."""
    try:
        def run(*a):
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()

        return run("rev-parse", "--short", "HEAD") + (" (dirty)" if run("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def save_parameters(cfg: dict, path: Path) -> None:
    """Write the full resolved config; `uv run <same script> <path>` reproduces the run."""
    script = relative_to_root(sys.argv[0])
    header = (
        f"# reproduce: uv run {script} {relative_to_root(path)}\n"
        f"# command:   {shlex.join(['uv', 'run', script, *sys.argv[1:]])}\n"
        f"# git:       {git_state()}\n"
        f"# python:    {platform.python_version()} on {platform.platform()}\n\n"
    )
    path.write_text(header + yaml.safe_dump(cfg, sort_keys=False))


def new_experiment(cfg: dict, parent: Path | None = None, prefix: str = "", log: str = "train.log") -> Path:
    """Create `<parent>/<prefix><timestamp>[_<name>]/` with parameters.yaml and logs/<log>."""
    parent = parent or resolve(cfg["output_dir"])
    stem = prefix + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not prefix:  # training run: <timestamp>_<name>
        stem += f"_{cfg['name']}" if cfg.get("name") else ""
    exp, i = parent / stem, 1
    while exp.exists():  # two runs in the same second
        exp, i = parent / f"{stem}-{i}", i + 1
    (exp / "logs").mkdir(parents=True)

    setup_logger(exp / "logs" / log)
    save_parameters(cfg, exp / "parameters.yaml")
    logger.info(f"experiment: {relative_to_root(exp)}")
    return exp
