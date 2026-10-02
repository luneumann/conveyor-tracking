import os
import time

import numpy as np
import pytest

import ctrack.vision.onnx_models as om
from conftest import ROOT
from ctrack.vision.onnx_models import DINO, DINO_INT8, DinoFeatures, accelerator_status, coreml_available

MODELS = ROOT / "models"
needs_vision = pytest.mark.skipif(not (MODELS / DINO).exists(), reason="DINOv2 ONNX model not downloaded")


def test_accelerator_is_off_in_tests_by_default():
    assert os.environ.get("CTRACK_NO_ACCEL") == "1" and not coreml_available()
    st = accelerator_status()
    assert st["available"] is False and set(st) == {"available", "ready", "pending", "failed"}


@needs_vision
def test_default_backbone_without_accelerator_prefers_int8_when_present():
    d = DinoFeatures(MODELS)
    assert d.filename == (DINO_INT8 if (MODELS / DINO_INT8).exists() else DINO)
    assert DinoFeatures(MODELS, DINO).filename == DINO          # a model trained on fp32 gets fp32 back


@needs_vision
def test_extract_falls_back_to_cpu_while_accelerated_session_is_not_ready(monkeypatch):
    monkeypatch.delenv("CTRACK_NO_ACCEL")
    monkeypatch.setattr(om, "accelerated_session", lambda *a, **k: None)         # compile still pending
    d = DinoFeatures(MODELS, DINO, accelerator=True)
    assert d._accelerate
    f = d.extract(np.zeros((168, 168, 3), np.uint8))
    assert f.shape == (12, 12, 384) and np.isfinite(f).all()


def test_int8_backbone_never_uses_the_accelerator(monkeypatch):
    if not (MODELS / DINO_INT8).exists():
        pytest.skip("no INT8 copy")
    monkeypatch.delenv("CTRACK_NO_ACCEL")
    assert not DinoFeatures(MODELS, DINO_INT8, accelerator=True)._accelerate


@needs_vision
@pytest.mark.skipif(os.environ.get("CTRACK_TEST_ACCEL") != "1", reason="set CTRACK_TEST_ACCEL=1 (needs ~40 s for the first compile)")
def test_accelerated_features_match_the_cpu_path(monkeypatch, tmp_path):
    monkeypatch.delenv("CTRACK_NO_ACCEL")
    if not coreml_available():
        pytest.skip("CoreML provider not available")
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (168, 168, 3), dtype=np.uint8)
    cpu = DinoFeatures(MODELS, DINO, accelerator=False).extract(img)
    acc = DinoFeatures(MODELS, DINO, accelerator=True)
    first = acc.extract(img)                                                      # CPU fallback, starts the compile
    assert np.allclose(first, cpu, atol=1e-4)
    t_end = time.time() + 150
    while "168x168" not in accelerator_status()["ready"] and time.time() < t_end:
        assert not accelerator_status()["failed"], accelerator_status()["failed"]
        time.sleep(1)
    assert "168x168" in accelerator_status()["ready"]
    fast = acc.extract(img)
    cos = (fast * cpu).sum(-1)
    assert cos.mean() > 0.999 and cos.min() > 0.99
    t = time.perf_counter()
    for _ in range(20):
        acc.extract(img)
    assert (time.perf_counter() - t) / 20 < 0.015                                 # CPU needs ~28 ms incl. preprocessing
