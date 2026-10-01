#!/usr/bin/env python3
"""Smoke test: load the MediaPipe hand model and run it on blank frames (no camera needed)."""

import time

import numpy as np

from ctrack.detector import HandDetector
from ctrack.types import Frame

d = HandDetector()
t0 = time.time()
results = [d.detect(Frame(np.zeros((720, 1280, 3), np.uint8), 1000.0 + i * 0.033, i)) for i in range(30)]
ms = (time.time() - t0) / 30 * 1000
d.close()
print(f"DETECTOR OK: {ms:.1f} ms/frame, detections on blank image: {sum(r is not None for r in results)}")
