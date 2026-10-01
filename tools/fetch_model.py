#!/usr/bin/env python3
"""Download the MediaPipe hand landmarker model to models/hand_landmarker.task (ADR-003)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
DEST = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def main() -> int:
    if DEST.exists():
        print(f"already present: {DEST}")
        return 0
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    tmp.rename(DEST)
    print(f"saved {DEST} ({DEST.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
