"""Template teach-in for the shape_match detector (used by tools/teach.py and the web GUI)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def auto_mask(crop_gray: np.ndarray) -> np.ndarray:
    """Largest Otsu blob (the class that is not the background), filled."""
    _, bw = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([bw[0], bw[-1], bw[:, 0], bw[:, -1]])
    if border.mean() > 127:  # background is the bright class
        bw = 255 - bw
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(crop_gray)
    if contours:
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return mask


def save_template(image: np.ndarray, roi: tuple[int, int, int, int], out: Path,
                  mask_auto: bool = False) -> tuple[Path, Path | None, float | None]:
    """Cut `roi` (x, y, w, h) out of a BGR/gray image, write it to `out` (PNG) and optionally a mask.

    Returns (template path, mask path or None, mask coverage 0..1 or None). The template centre becomes
    the pose origin and the part's orientation in the image becomes theta = 0.
    """
    x, y, w, h = roi
    if w < 8 or h < 8:
        raise ValueError("ROI too small")
    if x < 0 or y < 0 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError(f"ROI {x},{y},{w},{h} is outside the {image.shape[1]}x{image.shape[0]} image")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop = gray[y:y + h, x:x + w]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), crop)
    if not mask_auto:
        return out, None, None
    mask_path = out.with_name(out.stem + "_mask.png")
    mask = auto_mask(crop)
    cv2.imwrite(str(mask_path), mask)
    return out, mask_path, float(mask.mean() / 255)
