#!/usr/bin/env python3
"""Measure the real exposure offset of a webcam (P1-3).

The screen toggles a large patch black/white at known times; the camera films the screen.
For each toggle, the first frame whose brightness crosses the midpoint is found, and
offset = (time.time() after grab) - (time of toggle). The median is the value for
`camera.exposure_offset_ms`.

Setup: the camera must see the screen. A laptop webcam cannot see its own display — use a mirror,
or an external webcam pointed at the laptop screen. Includes display latency (~1 refresh),
so the result is a slight over-estimate; that is the conservative direction.

    python tools/latency_probe.py --device 0 --toggles 20
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import cv2
import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--toggles", type=int, default=20)
    p.add_argument("--period", type=float, default=0.6, help="seconds between toggles")
    p.add_argument("--roi", type=float, default=0.3, help="central image fraction used for brightness")
    args = p.parse_args(argv)

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print("cannot open camera", file=sys.stderr)
        return 1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("probe", cv2.WINDOW_NORMAL)
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    print("Point the camera at the 'probe' window (fill most of the camera view). Press any key to start, Q to abort.")
    while True:
        ok, img = cap.read()
        if ok:
            cv2.imshow("camera", img)
        cv2.imshow("probe", np.full((600, 800), 255, np.uint8))
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return 1
        if k != 255:
            break

    def brightness(img: np.ndarray) -> float:
        h, w = img.shape[:2]
        r = args.roi / 2
        roi = img[int(h * (0.5 - r)):int(h * (0.5 + r)), int(w * (0.5 - r)):int(w * (0.5 + r))]
        return float(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).mean())

    # Calibrate black/white levels.
    levels = []
    for value in (0, 255):
        cv2.imshow("probe", np.full((600, 800), value, np.uint8))
        cv2.waitKey(1)
        t_end = time.time() + 1.0
        samples = []
        while time.time() < t_end:
            ok, img = cap.read()
            cv2.waitKey(1)
            if ok:
                samples.append(brightness(img))
        levels.append(statistics.median(samples[len(samples) // 2:]))
    mid = (levels[0] + levels[1]) / 2
    print(f"black {levels[0]:.0f}, white {levels[1]:.0f}, threshold {mid:.0f}")
    if levels[1] - levels[0] < 30:
        print("contrast too low — camera does not see the probe window", file=sys.stderr)
        return 1

    offsets = []
    white = True
    for i in range(args.toggles):
        white = not white
        cv2.imshow("probe", np.full((600, 800), 255 if white else 0, np.uint8))
        cv2.waitKey(1)  # pushes the frame to the display
        t_toggle = time.time()
        t_deadline = t_toggle + args.period
        found = None
        while time.time() < t_deadline:
            if not cap.grab():
                continue
            t_grab = time.time()
            ok, img = cap.retrieve()
            cv2.waitKey(1)
            if ok and found is None and (brightness(img) > mid) == white:
                found = t_grab - t_toggle
        if found is not None:
            offsets.append(found * 1000.0)
            print(f"toggle {i + 1:2d}: {offsets[-1]:6.1f} ms")
        else:
            print(f"toggle {i + 1:2d}: not detected")

    cap.release()
    cv2.destroyAllWindows()
    if len(offsets) < 3:
        print("too few detections", file=sys.stderr)
        return 1
    med = statistics.median(offsets)
    print(f"\nexposure_offset_ms ≈ {med:.0f}  (median of {len(offsets)}, "
          f"min {min(offsets):.0f}, max {max(offsets):.0f})")
    print(f"→ set in config:  camera: {{ exposure_offset_ms: {med:.0f} }}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
