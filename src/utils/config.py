"""YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "base_config.yaml"
    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f)
