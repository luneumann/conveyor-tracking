from __future__ import annotations

import os
import sys
from pathlib import Path

# Tests run on the CPU path: no 30 s background compilations, no native threads alive at interpreter exit.
# tests/test_accelerator.py enables it explicitly (CTRACK_TEST_ACCEL=1).
os.environ.setdefault("CTRACK_NO_ACCEL", "1")

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
