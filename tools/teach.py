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


def auto_mask(crop_gray: np.ndarray) -> np.ndarray:
    """Largest Otsu blob (either polarity: the one that does not touch most of the border), filled."""
    _, bw = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:  # background is the bright class
        bw = 255 - bw
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(crop_gray)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return mask


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
    if w < 8 or h < 8:
        raise SystemExit("ROI too small / cancelled")
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        raise SystemExit(f"ROI {x},{y},{w},{h} is outside the {img.shape[1]}x{img.shape[0]} image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), gray)
    print(f"template {w}x{h} -> {args.out}   (pose origin = template centre)")
    if args.mask_auto:
        mask_path = args.out.with_name(args.out.stem + "_mask.png")
        mask = auto_mask(gray)
        cv2.imwrite(str(mask_path), mask)
        print(f"mask covers {mask.mean() / 255 * 100:.0f}% of the template -> {mask_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
