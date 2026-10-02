"""ONNX wrappers: MobileSAM (one-click segmentation) and DINOv2-small (patch features)."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

SAM_ENCODER = "mobile_sam_image_encoder.onnx"
SAM_DECODER = "sam_mask_decoder_single.onnx"
DINO = "dinov2_small.onnx"
DINO_INT8 = "dinov2_small_int8.onnx"   # dynamically quantized copy (~1.7x faster, same accuracy in our tests)
PATCH = 14
EMBED_DIM = 384
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def vision_models_present(models_dir: Path) -> bool:
    return all((models_dir / f).exists() for f in (SAM_ENCODER, SAM_DECODER, DINO))


def _session(path: Path, threads: int | None = None):
    import onnxruntime as ort

    if not path.exists():
        raise FileNotFoundError(f"Modell fehlt: {path}. In der Oberfläche 'Modelle laden' drücken.")
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


# ---- Apple Neural Engine / GPU via ONNX Runtime's CoreML provider -----------------------------------------------------
# The CoreML provider needs STATIC shapes. ONNX Runtime can pin the symbolic dims at load time (free-dimension
# overrides), no `onnx` package needed. Compiling a shape takes ~30 s the first time (then ~6 s from the on-disk cache),
# so DinoFeatures keeps running on the CPU until the accelerated session for that shape is ready. Measured on an M3:
# 168 px 28 -> 6 ms, 224 px 47 -> 10 ms, 448x252 157 -> 27 ms, output cosine 1.0000 vs. CPU.
_ACCEL_LOCK = threading.Lock()
_ACCEL: dict[tuple[str, int, int], object] = {}      # Session | "pending" | "failed"
_ACCEL_ERRORS: dict[tuple[str, int, int], str] = {}


def coreml_available() -> bool:
    if sys.platform != "darwin" or os.environ.get("CTRACK_NO_ACCEL"):
        return False
    import onnxruntime as ort

    return "CoreMLExecutionProvider" in ort.get_available_providers()


def _compile_accelerated(key: tuple[str, int, int], path: Path, cache_root: Path) -> None:
    import onnxruntime as ort

    _, h, w = key
    try:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        for name, value in (("batch_size", 1), ("num_channels", 3), ("height", h), ("width", w)):
            opts.add_free_dimension_override_by_name(name, value)
        # One cache directory PER shape: the cache key ignores the overrides, so sharing one loads a model
        # compiled for another size and fails at run time with a shape mismatch.
        cache = cache_root / f"{h}x{w}"
        cache.mkdir(parents=True, exist_ok=True)
        provider = ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL",
                                                "RequireStaticInputShapes": "1", "ModelCacheDirectory": str(cache)})
        sess = ort.InferenceSession(str(path), opts, providers=[provider, "CPUExecutionProvider"])
        out = sess.run(None, {sess.get_inputs()[0].name: np.zeros((1, 3, h, w), np.float32)})[0]   # warm-up + sanity
        if out.shape[1:] != ((h // PATCH) * (w // PATCH) + 1, EMBED_DIM):
            raise RuntimeError(f"unexpected output shape {out.shape}")
        result: object = sess
    except Exception as e:  # keep running on the CPU; the reason is shown by accelerator_status()
        _ACCEL_ERRORS[key] = f"{type(e).__name__}: {str(e)[:160]}"
        result = "failed"
    with _ACCEL_LOCK:
        _ACCEL[key] = result


def accelerated_session(path: Path, h: int, w: int, cache_root: Path, block: bool = False):
    """The ready accelerated session for this shape, or None (and start compiling it in the background).

    block=True compiles right here instead: loading a CoreML model holds the Python GIL for seconds (6-14 s measured),
    which stalls the whole pipeline when it happens mid-run, so callers prepare shapes up front (DinoFeatures.warm_up).
    """
    key = (str(path), h, w)
    with _ACCEL_LOCK:
        entry = _ACCEL.get(key)
        first = entry is None
        if first:
            _ACCEL[key] = "pending"
    if first:
        if block:
            _compile_accelerated(key, path, cache_root)
        else:
            threading.Thread(target=_compile_accelerated, args=(key, path, cache_root), daemon=True,
                             name=f"ctrack-coreml-{h}x{w}").start()
            return None
    with _ACCEL_LOCK:
        entry = _ACCEL[key]
    return None if isinstance(entry, str) else entry


def accelerator_status() -> dict:
    """For the GUI: whether the accelerator exists and which shapes are ready / compiling / failed."""
    with _ACCEL_LOCK:
        items = list(_ACCEL.items())
    return {"available": coreml_available(),
            "ready": sorted(f"{h}x{w}" for (_, h, w), v in items if not isinstance(v, str)),
            "pending": sorted(f"{h}x{w}" for (_, h, w), v in items if v == "pending"),
            "failed": {f"{k[1]}x{k[2]}": _ACCEL_ERRORS.get(k, "") for k, v in items if v == "failed"}}


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
    """Frozen DINOv2-small patch embeddings (384-d per 14x14 px patch), any input size divisible by 14.

    By default the fp32 model is used where the Apple accelerator exists (it only takes fp32; the quantized INT8 copy
    stays CPU-only), otherwise the INT8 copy when present, else fp32.
    """

    def __init__(self, models_dir: Path, filename: str | None = None, threads: int | None = 4,
                 accelerator: bool | None = None) -> None:
        # A learned head only fits the backbone it was trained on, so callers pass the recorded name.
        fast = coreml_available() if accelerator is None else accelerator and coreml_available()
        if filename is None:
            filename = DINO if fast or not (models_dir / DINO_INT8).exists() else DINO_INT8
        self.filename = filename
        self.models_dir = models_dir
        # Quantized INT8 has ops CoreML cannot run; the accelerator only makes sense for the fp32 file.
        self._accelerate = fast and filename == DINO
        # 4 threads = the performance cores of an M-series chip; 8 is slower because the efficiency cores stall.
        self._sess = _session(models_dir / filename, threads)
        self._input = self._sess.get_inputs()[0].name

    def warm_up(self, shapes: list[tuple[int, int]]) -> None:
        """Compile the accelerated sessions for these (h, w) now (blocking) instead of stalling the pipeline later."""
        if self._accelerate:
            for h, w in shapes:
                accelerated_session(self.models_dir / self.filename, h, w, self.models_dir / "coreml_cache", block=True)

    def extract(self, bgr: np.ndarray) -> np.ndarray:
        """(gh, gw, 384) L2-normalized patch features of an image whose sides are multiples of 14."""
        h, w = bgr.shape[:2]
        if h % PATCH or w % PATCH:
            raise ValueError(f"image size {w}x{h} must be a multiple of {PATCH}")
        x = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        batch = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        sess = self._sess
        if self._accelerate:
            fast = accelerated_session(self.models_dir / self.filename, h, w, self.models_dir / "coreml_cache")
            if fast is not None:
                sess = fast
        out = sess.run(None, {self._input: batch})[0]
        feats = out[0, 1:].reshape(h // PATCH, w // PATCH, EMBED_DIM)  # drop the CLS token
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-6)
