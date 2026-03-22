from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


@dataclass
class RunPaths:
    runs_dir: Path
    run_dir: Path
    ckpt_path: Path


def make_run_paths(runs_dir: str | Path, run_name: str = "latest") -> RunPaths:
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_name
    ckpt_path = run_dir / "model.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(runs_dir=runs_dir, run_dir=run_dir, ckpt_path=ckpt_path)


