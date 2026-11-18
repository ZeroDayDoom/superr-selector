"""Configuration loading.

All analytical constants are stored in ``configs/config.yaml``. Code must read
values from here instead of hard-coding them so that every decision is
reproducible and separately versioned from logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    raw_data_dir: Path
    processed_dir: Path
    outputs_dir: Path
    models_dir: Path
    figures_dir: Path
    exports_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class Config:
    raw: Dict[str, Any]
    paths: Paths
    seed: int
    cleaning: Dict[str, Any]
    pressure: Dict[str, Any]
    models: Dict[str, Any]
    ratings: Dict[str, Any]
    dashboard: Dict[str, Any]


def _build_paths(cfg: Dict[str, Any]) -> Paths:
    p = cfg["paths"]
    raw = Path(p["raw_data_dir"])
    if not raw.is_absolute():
        raw = (PROJECT_ROOT / raw).resolve()
    else:
        raw = raw.resolve()

    def out(name: str) -> Path:
        d = Path(p[name])
        if not d.is_absolute():
            d = (PROJECT_ROOT / d).resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d

    return Paths(
        raw_data_dir=raw,
        processed_dir=out("processed_dir"),
        outputs_dir=out("outputs_dir"),
        models_dir=out("models_dir"),
        figures_dir=out("figures_dir"),
        exports_dir=out("exports_dir"),
        reports_dir=out("reports_dir"),
    )


@lru_cache(maxsize=8)
def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(
        raw=raw,
        paths=_build_paths(raw),
        seed=int(raw["seed"]),
        cleaning=raw["cleaning"],
        pressure=raw["pressure"],
        models=raw["models"],
        ratings=raw["ratings"],
        dashboard=raw["dashboard"],
    )
