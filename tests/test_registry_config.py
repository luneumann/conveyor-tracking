from pathlib import Path

import pytest

from conftest import ROOT
from ctrack.config import DEFAULTS, load_config, parse_override
from ctrack.detector.base import Detector
from ctrack.pipeline import Pipeline
from ctrack.registry import CAMERAS, DETECTORS, PUBLISHERS, TRANSFORMS, Registry, load_builtins
from ctrack.types import Detection, TrackState


def test_builtins_registered():
    load_builtins()
    assert {"webcam", "video_file", "synthetic"} <= set(CAMERAS.names())
    assert {"hand", "blob"} <= set(DETECTORS.names())
    assert "identity" in TRANSFORMS.names()
    assert {"udp", "none"} <= set(PUBLISHERS.names())


def test_unknown_type_lists_available():
    reg: Registry = Registry("thing")
    reg.register("a")(dict)
    with pytest.raises(KeyError, match="Available: a"):
        reg.create({"type": "b"})


def test_duplicate_registration_rejected():
    reg: Registry = Registry("thing")
    reg.register("a")(dict)
    with pytest.raises(ValueError):
        reg.register("a")(list)


def test_create_passes_kwargs():
    load_builtins()
    pub = PUBLISHERS.create({"type": "udp", "host": "127.0.0.1", "port": 9999, "rate_hz": 10})
    assert pub.addr == ("127.0.0.1", 9999) and pub.min_interval == pytest.approx(0.1)
    pub.close()


@pytest.mark.parametrize("name", ["default.yaml", "synthetic.yaml", "replay.yaml"])
def test_shipped_configs_load(name):
    cfg = load_config(ROOT / "config" / name)
    assert set(DEFAULTS) <= set(cfg)


def test_type_switch_drops_default_kwargs(tmp_path: Path):
    p = tmp_path / "c.yaml"
    p.write_text("camera: { type: synthetic, fps: 10 }\n")
    cfg = load_config(p)
    assert cfg["camera"] == {"type": "synthetic", "fps": 10}  # no webcam 'device' leaking in


def test_parse_override():
    assert parse_override("predictor.horizon_ms=150") == {"predictor": {"horizon_ms": 150}}
    assert parse_override("camera.realtime=false") == {"camera": {"realtime": False}}
    with pytest.raises(ValueError):
        parse_override("nonsense")


def test_new_detector_needs_only_class_and_registry_entry():
    """P0-10: plugging in a new detector touches neither Tracker, Predictor nor Publisher."""

    @DETECTORS.register("test_fixed")
    class FixedDetector(Detector):
        def __init__(self, x: float = 50.0) -> None:
            self.x = x

        def detect(self, frame):
            return Detection(self.x, 60.0, 0.0, 1.0)

    cfg = load_config(overrides={
        "camera": {"type": "synthetic", "max_frames": 3, "realtime": False, "width": 64, "height": 48},
        "detector": {"type": "test_fixed", "x": 42.0},
        "output": {"type": "none"},
    })
    pipe = Pipeline.from_config(cfg)
    pipe.tracker.request_lock()
    step = pipe.step()
    assert step.state is TrackState.TRACKING and step.pose.x == pytest.approx(42.0)
