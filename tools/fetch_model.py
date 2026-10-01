#!/usr/bin/env python3
"""Download the MediaPipe hand landmarker model to models/hand_landmarker.task (ADR-003)."""

from __future__ import annotations

import sys
from pathlib import Path

from ctrack.models import HAND_MODEL_URL, fetch_hand_model

DEST = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def main() -> int:
    if DEST.exists():
        print(f"already present: {DEST}")
        return 0
    print(f"downloading {HAND_MODEL_URL}")
    fetch_hand_model(DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
