"""Load a YAML config and apply `key.sub=value` CLI overrides."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Relative paths are anchored at the repo root, absolute ones pass through."""
    return ROOT / path


def relative_to_root(path: str | Path) -> str:
    """Repo-relative path string when inside the repo (portable across machines), else absolute."""
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def parse_value(text: str):
    """CLI override value → python value (YAML syntax: 3, 0.1, true, null, [1, 2], {a: 1})."""
    value = yaml.safe_load(text)
    if isinstance(value, str):  # YAML 1.1 reads "1e-4" as a string
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(argv: list[str] | None = None) -> dict:
    """`<config.yaml> [key.sub=value ...]` → nested dict, CLI overrides applied.

    Overriding a key that is not in the YAML is an error, so typos fail loudly.
    """
    argv = sys.argv[1:] if argv is None else argv
    if not argv or "=" in argv[0]:
        sys.exit("usage: <command> <config.yaml> [key.sub=value ...]")

    cfg = yaml.safe_load(Path(argv[0]).read_text())
    for item in argv[1:]:
        key, sep, text = item.partition("=")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.get(p) if isinstance(node, dict) else None
        if not sep or not isinstance(node, dict) or leaf not in node:
            sys.exit(f"[config] unknown key {key!r}: add it to the YAML first")
        node[leaf] = parse_value(text)
    if "name" in cfg and not cfg["name"]:  # experiment folder suffix: the config file name, e.g. b1_resnet50_focal
        cfg["name"] = Path(argv[0]).stem
    return cfg
