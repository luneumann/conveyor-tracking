#!/usr/bin/env python3
"""Teach-in for the shape_match detector: cut a reference template out of a camera frame or image.

The template is the cropped region; its centre becomes the pose origin and its orientation in the
image becomes theta = 0. Rotate the part to the pose you want to call "zero" before teaching.

    python tools/teach.py --camera 0 --out templates/part.png            # live: SPACE to freeze, drag ROI
    python tools/teach.py --image photo.png --out templates/part.png     # from file, drag ROI
    python tools/teach.py --image photo.png --roi 410,220,200,100 --out templates/part.png   # no GUI

Optional: --mask-auto writes <out>_mask.png (Otsu on the crop, largest blob, filled) so background
pixels inside the ROI do not take part in matching. Check it visually; pass it as detector.mask.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ctrack.teach import auto_mask, save_template  # noqa: F401  (auto_mask re-exported)


def grab_frame(device: int) -> np.ndarray:
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise SystemExit(f"cannot open camera {device} (macOS: allow camera access for your terminal app)")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("SPACE = freeze frame, Q = abort")
    frame = None
    while True:
        ok, img = cap.read()
        if not ok:
            continue
        cv2.imshow("teach", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            frame = img
            break
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if frame is None:
        raise SystemExit("aborted")
    return frame


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", type=int, help="camera index")
    src.add_argument("--image", type=Path, help="image file")
    p.add_argument("--roi", help="x,y,w,h in pixels (skips the GUI selection)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mask-auto", action="store_true")
    args = p.parse_args(argv)

    if args.image is not None:
        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"cannot read {args.image}")
    else:
        img = grab_frame(args.camera)

    if args.roi:
        x, y, w, h = (int(v) for v in args.roi.split(","))
    else:
        print("drag a rectangle around the part, ENTER to confirm, C to cancel")
        x, y, w, h = (int(v) for v in cv2.selectROI("teach", img, showCrosshair=True))
        cv2.destroyAllWindows()
    try:
        out, mask_path, coverage = save_template(img, (x, y, w, h), args.out, mask_auto=args.mask_auto)
    except ValueError as e:
        raise SystemExit(str(e))
    print(f"template {w}x{h} -> {out}   (pose origin = template centre)")
    if mask_path is not None:
        print(f"mask covers {coverage * 100:.0f}% of the template -> {mask_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
