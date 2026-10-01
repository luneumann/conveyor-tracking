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


VISION_MODELS = {  # file name -> (url, approx. size in MB)
    "mobile_sam_image_encoder.onnx": ("https://huggingface.co/Acly/MobileSAM/resolve/main/mobile_sam_image_encoder.onnx", 28.2),
    "sam_mask_decoder_single.onnx": ("https://huggingface.co/Acly/MobileSAM/resolve/main/sam_mask_decoder_single.onnx", 16.5),
    "dinov2_small.onnx": ("https://huggingface.co/onnx-community/dinov2-small/resolve/main/onnx/model.onnx", 88.5),
}


def fetch_vision_models(models_dir: Path) -> list[str]:
    """Download the MobileSAM + DINOv2-small ONNX files that are missing; returns the names fetched."""
    fetched = []
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, (url, _) in VISION_MODELS.items():
        dest = models_dir / name
        if dest.exists():
            continue
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        fetched.append(name)
    return fetched
