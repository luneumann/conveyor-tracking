"""Background engine behind the web GUI: owns the pipeline thread, settings, teach-in and templates.

All state shared with HTTP handler threads is guarded by `self._lock`. The pipeline itself (tracker,
filter, camera) is only touched from the engine thread; handlers talk to it through settings that the
loop re-reads every frame and through the command queue (lock / reset).
"""

from __future__ import annotations

import json
import math
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..camera.video_file import Recorder
from ..config import load_config
from ..metrics import MetricsLogger
from ..models import fetch_hand_model
from ..pipeline import Pipeline
from ..publisher.base import NullPublisher
from ..publisher.udp import UdpPublisher
from ..registry import load_builtins
from ..teach import save_template
from ..types import TrackState
from ..visualizer import OverlayRenderer

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
PRESETS = {  # motion preset -> (process_noise, measurement_noise)
    "band": (50.0, 1.0),    # steady belt motion: smooth, trusts the constant-velocity model
    "hand": (300000.0, 4.0),  # hand-held / jerky: follows quickly (see SETUP.md, filter tuning)
}
BUILTIN_TEMPLATES = {"synthetic_part"}


@dataclass
class Settings:
    source: str = "camera"          # camera | demo | replay
    device: int = 0
    recording: str = ""             # file name in recordings/ (source == replay)
    detector: str = "template"      # hand | template
    template: str = "synthetic_part"
    preset: str = "band"            # band | hand
    min_score: float = 0.85
    horizon_ms: float = 100.0
    coast_ms: float = 300.0
    auto_lock: bool = False
    output_enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5005
    record: bool = False

    def update(self, data: dict[str, Any]) -> set[str]:
        """Apply validated changes; returns the names of the fields that actually changed."""
        changed: set[str] = set()
        types = {f.name: f.type for f in fields(self)}
        for key, value in data.items():
            if key not in types:
                continue
            kind = {"int": int, "float": float, "str": str, "bool": bool}[str(types[key])]
            if kind is bool:
                value = bool(value)
            else:
                value = kind(value)
            if key == "source" and value not in ("camera", "demo", "replay"):
                raise ValueError("source must be camera, demo or replay")
            if key == "detector" and value not in ("hand", "template"):
                raise ValueError("detector must be hand or template")
            if key == "preset" and value not in PRESETS:
                raise ValueError("unknown preset")
            if key == "min_score":
                value = min(max(value, 0.3), 0.99)
            if key == "horizon_ms":
                value = min(max(value, 0.0), 1000.0)
            if key == "coast_ms":
                value = min(max(value, 50.0), 5000.0)
            if key == "port" and not 1 <= value <= 65535:
                raise ValueError("port out of range")
            if key == "device" and not 0 <= value <= 9:
                raise ValueError("device out of range")
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed.add(key)
        return changed


#: Settings that need a camera/detector restart when changed while running.
RESTART_KEYS = {"source", "device", "recording", "detector", "template"}


def _num(v: float) -> float | None:
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


class Engine:
    def __init__(self, root: Path) -> None:
        load_builtins()
        self.root = root
        self.templates_dir = root / "templates"
        self.logs_dir = root / "logs"
        self.recordings_dir = root / "recordings"
        self.settings_path = root / "logs" / "gui_settings.json"
        self._lock = threading.RLock()
        self._commands: queue.Queue[str] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_cv = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_no = 0
        self._last_raw: np.ndarray | None = None
        self._teach_frame: np.ndarray | None = None
        self._tpl_cache: tuple[tuple, list[dict[str, Any]]] | None = None
        self.settings = self._load_settings()
        self.status: dict[str, Any] = self._idle_status()

    # ---- settings ---------------------------------------------------------------------------
    def _load_settings(self) -> Settings:
        s = Settings()
        try:
            data = json.loads(self.settings_path.read_text())
            data.pop("record", None)
            data.pop("auto_lock", None)
            data.pop("output_enabled", None)  # never start sending / recording implicitly
            s.update(data)
        except (OSError, ValueError, TypeError):
            pass
        if s.detector == "template" and not self._template_path(s.template).exists():
            s.template = "synthetic_part"
        return s

    def _save_settings(self) -> None:
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(asdict(self.settings), indent=1))
        except OSError:
            pass

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            changed = self.settings.update(data)
            if "detector" in changed and "preset" not in data:
                # Sensible default motion model per detector; the user can override it.
                self.settings.update({"preset": "hand" if self.settings.detector == "hand" else "band"})
            running = self._running()
        self._save_settings()
        if running and changed & RESTART_KEYS:
            self.start()  # restart with the new source / detector
        return self.snapshot()

    # ---- session control --------------------------------------------------------------------
    def _running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        self.stop()
        with self._lock:
            s = Settings(**asdict(self.settings))
            self.status = self._idle_status()
            self.status.update(running=True, starting=True)
        self._stop.clear()
        while not self._commands.empty():
            self._commands.get_nowait()
        self._thread = threading.Thread(target=self._loop, args=(s,), daemon=True, name="ctrack-engine")
        self._thread.start()

    def stop(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=8)
            self._thread = None
        with self._lock:
            self.status["running"] = False
            self.status["starting"] = False
        with self._frame_cv:
            self._jpeg = None
            self._frame_cv.notify_all()

    def command(self, name: str) -> None:
        if name not in ("lock", "reset"):
            raise ValueError("unknown command")
        self._commands.put(name)

    # ---- pipeline thread --------------------------------------------------------------------
    def _build_config(self, s: Settings) -> dict[str, Any]:
        overrides: dict[str, Any] = {}
        if s.source == "camera":
            overrides["camera"] = {"type": "webcam", "device": s.device, "width": 1280, "height": 720,
                                   "exposure_offset_ms": 30}
        elif s.source == "demo":
            overrides["camera"] = {"type": "synthetic", "width": 1280, "height": 720, "fps": 30,
                                   "velocity": [250.0, 15.0], "omega": 0.3, "start": [100.0, 300.0, 0.0],
                                   "image_noise": 6.0, "jitter_s": 0.002, "realtime": True}
        else:
            path = self._recording_path(s.recording)
            if path is None:
                raise ValueError("Keine Aufnahme ausgewählt")
            overrides["camera"] = {"type": "video_file", "path": str(path), "realtime": True, "loop": True}

        if s.detector == "hand":
            overrides["detector"] = {"type": "hand", "min_confidence": 0.6,
                                     "model_path": str(self.root / "models" / "hand_landmarker.task")}
        else:
            tpl = self._template_path(s.template)
            if not tpl.exists():
                raise ValueError(f"Referenzbild '{s.template}' nicht gefunden – bitte zuerst einlernen")
            det: dict[str, Any] = {"type": "shape_match", "template": str(tpl), "min_score": s.min_score,
                                   "downscale": 8 if s.source == "demo" else 4}
            mask = tpl.with_name(tpl.stem + "_mask.png")
            if mask.exists():
                det["mask"] = str(mask)
            overrides["detector"] = det

        q, r = PRESETS[s.preset]
        overrides["predictor"] = {"process_noise": q, "measurement_noise": r, "horizon_ms": s.horizon_ms}
        overrides["tracker"] = {"coast_ms": s.coast_ms}
        overrides["output"] = {"type": "none"}
        return load_config(None, overrides)

    def _loop(self, s: Settings) -> None:
        pipeline = metrics = recorder = publisher = None
        null_publisher = NullPublisher()
        applied: dict[str, Any] = {}
        sent = 0
        try:
            cfg = self._build_config(s)
            pipeline = Pipeline.from_config(cfg)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            csv_path = self.logs_dir / f"gui_{stamp}.csv"
            metrics = MetricsLogger(csv=str(csv_path), match_tolerance_ms=25, window=300)
            renderer = OverlayRenderer(horizon_ms=s.horizon_ms, reacquire_radius_px=cfg["tracker"]["reacquire_radius_px"], hud=False)
            tracker = pipeline.tracker
            with self._lock:
                self.status.update(starting=False, csv=csv_path.name, session=stamp)
            frames = 0
            while not self._stop.is_set():
                live = Settings(**asdict(self.settings))
                self._sync_live(live, applied, pipeline, renderer)
                if live.output_enabled != applied.get("output_enabled") or (live.host, live.port) != applied.get("addr"):
                    if publisher is not None:
                        publisher.close()
                    publisher = UdpPublisher(live.host, live.port) if live.output_enabled else None
                    pipeline.publisher = publisher if publisher is not None else null_publisher
                    applied.update(output_enabled=live.output_enabled, addr=(live.host, live.port))
                if live.record != applied.get("record"):
                    if recorder is not None:
                        recorder.close()
                        recorder = None
                    if live.record:
                        recorder = Recorder(self.recordings_dir / f"gui_{time.strftime('%Y%m%d_%H%M%S')}")
                    applied["record"] = live.record
                while True:
                    try:
                        cmd = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    tracker.request_lock() if cmd == "lock" else tracker.reset()
                if live.auto_lock:
                    tracker.auto_lock_step(time.time(), 1.0, immediate=not pipeline.camera.is_live)

                step = pipeline.step()
                if step is None:
                    break
                frames += 1
                if publisher is not None and step.published:
                    sent += 1
                if recorder is not None:
                    recorder.write(step.frame)
                metrics.log(step)
                stats = metrics.stats()
                t = step.frame.t_exposure
                img = renderer.render(step, stats, tracker.reference_pose(t), tracker.reacquire_radius_at(t))
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
                with self._frame_cv:
                    self._last_raw = step.frame.image
                    if ok:
                        self._jpeg = buf.tobytes()
                        self._frame_no += 1
                    self._frame_cv.notify_all()
                self._publish_status(step, stats, tracker, frames, sent, publisher is not None, recorder is not None)
        except Exception as e:  # camera permission, missing template, ... -> shown in the GUI
            with self._lock:
                self.status["error"] = f"{type(e).__name__}: {e}"
        finally:
            summary = None
            if metrics is not None:
                summary = {k: _num(v) for k, v in metrics.stats().items()}
                summary["frames"] = metrics.rows_written
                metrics.close()
            for closer in (recorder, publisher, pipeline):
                if closer is not None:
                    closer.close()
            with self._lock:
                self.status.update(running=False, starting=False)
                if summary is not None:
                    self.status["last_run"] = {**summary, "csv": self.status.get("csv")}
            with self._frame_cv:
                self._jpeg = None
                self._frame_cv.notify_all()

    def _sync_live(self, live: Settings, applied: dict[str, Any], pipeline: Pipeline, renderer: OverlayRenderer) -> None:
        """Apply settings that can change without restarting the camera."""
        tracker, predictor = pipeline.tracker, pipeline.tracker.predictor
        if live.horizon_ms != applied.get("horizon_ms"):
            predictor.configure(horizon_ms=live.horizon_ms)
            pipeline.horizon_s = predictor.horizon_s
            renderer.horizon_ms = live.horizon_ms
            applied["horizon_ms"] = live.horizon_ms
        if live.coast_ms != applied.get("coast_ms"):
            tracker.coast_s = live.coast_ms / 1000.0
            applied["coast_ms"] = live.coast_ms
        if live.preset != applied.get("preset"):
            q, r = PRESETS[live.preset]
            predictor.configure(process_noise=q, measurement_noise=r)
            applied["preset"] = live.preset
        if live.min_score != applied.get("min_score"):
            if hasattr(pipeline.detector, "min_score"):
                pipeline.detector.min_score = live.min_score
            applied["min_score"] = live.min_score

    def _publish_status(self, step, stats, tracker, frames: int, sent: int, sending: bool, recording: bool) -> None:
        pose = step.pose
        vel = step.velocity
        h, w = step.frame.image.shape[:2]
        snap = {
            "running": True, "starting": False, "frame_size": [w, h], "frames": frames,
            "state": step.state.value, "lock_pending": tracker.lock_requested,
            "detected": step.detection is not None,
            "score": _num(step.detection.confidence) if step.detection else None,
            "fps": _num(stats["fps"]), "latency_p50": _num(stats["latency_p50"]),
            "latency_p95": _num(stats["latency_p95"]), "pred_err_p95": _num(stats["pred_err_p95"]),
            "pred_err_pct": _num(stats["pred_err_p95_pct"]),
            "pose": None if pose is None else {"x": pose.x, "y": pose.y, "theta_deg": math.degrees(pose.theta)},
            "velocity": None if vel is None else {"vx": vel.vx, "vy": vel.vy, "speed": math.hypot(vel.vx, vel.vy)},
            "sending": sending, "packets_sent": sent, "recording": recording,
        }
        with self._lock:
            self.status.update(snap)

    @staticmethod
    def _idle_status() -> dict[str, Any]:
        return {"running": False, "starting": False, "error": None, "state": "IDLE", "lock_pending": False,
                "detected": False, "score": None, "fps": None, "latency_p50": None, "latency_p95": None,
                "pred_err_p95": None, "pred_err_pct": None, "pose": None, "velocity": None, "frame_size": None,
                "frames": 0, "sending": False, "packets_sent": 0, "recording": False, "csv": None,
                "session": None, "last_run": None}

    def download_hand_model(self) -> None:
        fetch_hand_model(self.root / "models" / "hand_landmarker.task")

    # ---- live video -------------------------------------------------------------------------
    def wait_frame(self, last_no: int, timeout: float = 1.0) -> tuple[int, bytes | None, bool]:
        """Block until a new frame exists. Returns (frame number, jpeg or None, session still active)."""
        with self._frame_cv:
            self._frame_cv.wait_for(lambda: self._frame_no != last_no or not self.status["running"], timeout)
            return self._frame_no, self._jpeg, bool(self.status["running"])

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {**self.status, "settings": asdict(self.settings), "presets": list(PRESETS),
                    "templates": self.list_templates(), "recordings": self.list_recordings(),
                    "teach_ready": self._teach_frame is not None,
                    "hand_model": (self.root / "models" / "hand_landmarker.task").exists()}

    # ---- teach-in ---------------------------------------------------------------------------
    def freeze_frame(self) -> bool:
        with self._frame_cv:
            if self._last_raw is None or not self._running():
                return False
            self._teach_frame = self._last_raw.copy()
        return True

    def load_teach_image(self, data: bytes) -> None:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Bild konnte nicht gelesen werden")
        if max(img.shape[:2]) > 4000:
            raise ValueError("Bild zu groß (max. 4000 px)")
        self._teach_frame = img

    def teach_jpeg(self) -> bytes | None:
        img = self._teach_frame
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return buf.tobytes() if ok else None

    def save_template(self, name: str, roi: list[float], mask_auto: bool) -> dict[str, Any]:
        if not NAME_RE.match(name):
            raise ValueError("Name: 1–40 Zeichen, nur Buchstaben, Ziffern, _ und -")
        if name in BUILTIN_TEMPLATES:
            raise ValueError(f"'{name}' ist das mitgelieferte Demo-Teil und kann nicht überschrieben werden")
        if self._teach_frame is None:
            raise ValueError("Kein eingefrorenes Bild")
        x, y, w, h = (int(round(v)) for v in roi)
        out, mask, coverage = save_template(self._teach_frame, (x, y, w, h), self.templates_dir / f"{name}.png", mask_auto)
        self._teach_frame = None
        self.update_settings({"detector": "template", "template": name})
        return {"name": name, "width": w, "height": h, "mask_coverage": coverage}

    # ---- templates / recordings -------------------------------------------------------------
    def _template_path(self, name: str) -> Path:
        return self.templates_dir / f"{name}.png"

    def list_templates(self) -> list[dict[str, Any]]:
        files = sorted(self.templates_dir.glob("*.png"))
        sig = tuple((p.name, p.stat().st_mtime_ns) for p in files)
        if self._tpl_cache is not None and self._tpl_cache[0] == sig:
            return self._tpl_cache[1]
        out = []
        for p in files:
            if p.stem.endswith("_mask"):
                continue
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            out.append({"name": p.stem, "width": int(img.shape[1]), "height": int(img.shape[0]),
                        "has_mask": p.with_name(p.stem + "_mask.png").exists(),
                        "builtin": p.stem in BUILTIN_TEMPLATES, "version": p.stat().st_mtime_ns})
        self._tpl_cache = (sig, out)
        return out

    def template_png(self, name: str, mask: bool = False) -> bytes | None:
        if not NAME_RE.match(name):
            return None
        p = self._template_path(name)
        if mask:
            p = p.with_name(p.stem + "_mask.png")
        return p.read_bytes() if p.exists() else None

    def delete_template(self, name: str) -> None:
        if not NAME_RE.match(name) or name in BUILTIN_TEMPLATES:
            raise ValueError("Dieses Referenzbild kann nicht gelöscht werden")
        p = self._template_path(name)
        p.unlink(missing_ok=True)
        p.with_name(p.stem + "_mask.png").unlink(missing_ok=True)
        if self.settings.template == name:
            self.update_settings({"template": "synthetic_part"})

    def list_recordings(self) -> list[str]:
        return sorted(p.name for p in self.recordings_dir.glob("*.mp4")) if self.recordings_dir.exists() else []

    def _recording_path(self, name: str) -> Path | None:
        if not name or "/" in name or "\\" in name or not name.endswith(".mp4"):
            return None
        p = self.recordings_dir / name
        return p if p.exists() else None

    def log_path(self, name: str) -> Path | None:
        if not re.match(r"^[A-Za-z0-9_.-]+\.csv$", name):
            return None
        p = self.logs_dir / name
        return p if p.exists() else None

