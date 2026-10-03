"""YAML configuration: layered over configs/default.yaml, hashed for provenance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load default.yaml, then overlay `path` if given. Fails loudly on missing files."""
    if not DEFAULT_CONFIG.exists():
        raise FileNotFoundError(f"default config missing: {DEFAULT_CONFIG}")
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text()) or {}
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        cfg = _merge(cfg, yaml.safe_load(p.read_text()) or {})
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
