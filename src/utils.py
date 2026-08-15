from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def load_config(path=None) -> dict:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str) -> Path:
    return REPO_ROOT / relative


def save_json(obj: dict, relative_path: str) -> Path:
    out = resolve_path(relative_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    return out
