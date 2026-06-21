from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ktt_ocr_eval.core.schemas import MethodConfig


def load_methods(path: Path) -> list[MethodConfig]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    methods_payload = payload.get("methods", [])
    if not isinstance(methods_payload, list):
        raise ValueError(f"{path} must contain a list at key 'methods'")
    return [MethodConfig.model_validate(item) for item in methods_payload]


def load_eval_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload
