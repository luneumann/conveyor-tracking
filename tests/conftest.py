from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from ctrack.types import Detection

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))


def det(x: float, y: float, theta: float = 0.0, confidence: float = 1.0) -> Detection:
    return Detection(x, y, theta, confidence)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
