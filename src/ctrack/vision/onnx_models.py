"""ONNX wrappers: MobileSAM (one-click segmentation) and DINOv2-small (patch features)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SAM_ENCODER = "mobile_sam_image_encoder.onnx"
SAM_DECODER = "sam_mask_decoder_single.onnx"
DINO = "dinov2_small.onnx"
PATCH = 14
EMBED_DIM = 384
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def vision_models_present(models_dir: Path) -> bool:
    return all((models_dir / f).exists() for f in (SAM_ENCODER, SAM_DECODER, DINO))


def _session(path: Path):
    import onnxruntime as ort

    if not path.exists():
        raise FileNotFoundError(f"Modell fehlt: {path}. In der Oberfläche 'Modelle laden' drücken.")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


class SamSegmenter:
    """Point-prompt segmentation. Call set_image() once per image (~0.4 s), then segment() per click (~25 ms)."""

    def __init__(self, models_dir: Path) -> None:
        self._enc = _session(models_dir / SAM_ENCODER)
        self._dec = _session(models_dir / SAM_DECODER)
        self._emb: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    def set_image(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        s = 1024.0 / max(h, w)
        nh, nw = round(h * s), round(w * s)
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        # The exported encoder expects the image already scaled to 1024 on its long side and padded
        # (it normalizes internally). Feeding the original size silently gives garbage embeddings.
        canvas = np.zeros((1024, 1024, 3), np.float32)
        canvas[:nh, :nw] = rgb
        self._emb = self._enc.run(None, {"input_image": canvas})[0]
        self._shape = (h, w)

    def segment(self, points: list[tuple[float, float]], labels: list[int]) -> np.ndarray:
        """Boolean mask for positive (1) / negative (0) clicks in original image pixel coordinates."""
        if self._emb is None or self._shape is None:
            raise RuntimeError("set_image() first")
        if not points:
            raise ValueError("at least one click is required")
        h, w = self._shape
        s = 1024.0 / max(h, w)
        coords = np.concatenate([np.asarray(points, np.float32) * s, np.zeros((1, 2), np.float32)])[None]
        lab = np.asarray(list(labels) + [-1], np.float32)[None]  # trailing padding point, as in SAM's ONNX export
        masks, _, _ = self._dec.run(None, {
            "image_embeddings": self._emb, "point_coords": coords, "point_labels": lab,
            "mask_input": np.zeros((1, 1, 256, 256), np.float32), "has_mask_input": np.zeros(1, np.float32),
            "orig_im_size": np.array([h, w], np.float32)})
        return masks[0, 0] > 0


class DinoFeatures:
    """Frozen DINOv2-small patch embeddings (384-d per 14x14 px patch), any input size divisible by 14."""

    def __init__(self, models_dir: Path) -> None:
        self._sess = _session(models_dir / DINO)
        self._input = self._sess.get_inputs()[0].name

    def extract(self, bgr: np.ndarray) -> np.ndarray:
        """(gh, gw, 384) L2-normalized patch features of an image whose sides are multiples of 14."""
        h, w = bgr.shape[:2]
        if h % PATCH or w % PATCH:
            raise ValueError(f"image size {w}x{h} must be a multiple of {PATCH}")
        x = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        out = self._sess.run(None, {self._input: np.ascontiguousarray(x.transpose(2, 0, 1)[None])})[0]
        feats = out[0, 1:].reshape(h // PATCH, w // PATCH, EMBED_DIM)  # drop the CLS token
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-6)
