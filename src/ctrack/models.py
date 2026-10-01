"""Model files that are not part of the pip packages."""

from __future__ import annotations

import urllib.request
from pathlib import Path

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/"
                  "float16/latest/hand_landmarker.task")


def fetch_hand_model(dest: Path) -> Path:
    """Download the MediaPipe hand landmarker model to `dest` (atomic: .part file, then rename)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
    tmp.rename(dest)
    return dest
