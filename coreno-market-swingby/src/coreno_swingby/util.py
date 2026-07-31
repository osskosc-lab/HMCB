from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np


def save_json(data: dict, path: Path) -> None:
    def clean(value):
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        return value

    path.write_text(
        json.dumps(clean(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def config_snapshot(*configs) -> dict:
    return {type(config).__name__: asdict(config) for config in configs}
