"""Config loading with single-level `extends` inheritance."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, resolving `extends:` against the config directory."""
    path = Path(path)
    if not path.exists():
        path = CONFIG_DIR / path.name
    with path.open() as fh:
        cfg = yaml.safe_load(fh) or {}

    parent_name = cfg.pop("extends", None)
    if parent_name:
        cfg = _deep_merge(load_config(CONFIG_DIR / parent_name), cfg)
    return cfg
