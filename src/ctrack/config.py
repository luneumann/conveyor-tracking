"""YAML configuration loading with defaults (PRD 5.5)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "camera": {"type": "webcam", "device": 0, "width": 1280, "height": 720, "exposure_offset_ms": 30},
    "detector": {"type": "hand", "min_confidence": 0.6, "model_path": "models/hand_landmarker.task"},
    "tracker": {"coast_ms": 300, "reacquire_radius_px": 80},
    "predictor": {
        "process_noise": 50.0,
        "process_noise_theta": 2.0,
        "measurement_noise": 4.0,
        "measurement_noise_theta": 0.05,
        "horizon_ms": 100,
    },
    "transform": {"type": "identity"},
    "output": {"type": "udp", "host": "127.0.0.1", "port": 5005, "rate_hz": 0, "include_predicted": True},
    "metrics": {"csv": "logs/run.csv", "match_tolerance_ms": 25, "window": 300},
    "visualizer": {"enabled": True},
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _apply(cfg: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    # A different module type must not inherit the previous type's kwargs (e.g. webcam 'device').
    cfg = copy.deepcopy(cfg)
    for section, block in user.items():
        if isinstance(block, dict) and "type" in block and block["type"] != cfg.get(section, {}).get("type"):
            cfg[section] = {}
    return deep_merge(cfg, user)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            cfg = _apply(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _apply(cfg, overrides)
    return cfg


def parse_override(expr: str) -> dict[str, Any]:
    """Turn 'predictor.horizon_ms=150' into {'predictor': {'horizon_ms': 150}}."""
    key, _, raw = expr.partition("=")
    if not key or not _:
        raise ValueError(f"Override must look like section.key=value, got '{expr}'")
    value = yaml.safe_load(raw)
    out: dict[str, Any] = {}
    node = out
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
    return out
