"""Background engine behind the web GUI: owns the pipeline thread, settings, teach-in and templates.

All state shared with HTTP handler threads is guarded by `self._lock`. The pipeline itself (tracker,
filter, camera) is only touched from the engine thread; handlers talk to it through settings that the
loop re-reads every frame and through the command queue (lock / reset).
"""

from __future__ import annotations

import collections
import csv
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
from ..models import fetch_hand_model, fetch_vision_models
from ..pipeline import Pipeline
from ..publisher.base import NullPublisher
from ..publisher.udp import UdpPublisher
from ..registry import load_builtins
from ..objectmodel import ObjectModel, train_object_model
from ..teach import save_template
from ..vision.onnx_models import DinoFeatures, SamSegmenter, accelerator_status, vision_models_present
from ..types import Frame, TrackState
from ..visualizer import OverlayRenderer

NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
PRESETS = {  # motion preset -> (process_noise, measurement_noise)
    "band": (50.0, 1.0),    # steady belt motion: smooth, trusts the constant-velocity model
    "hand": (300000.0, 4.0),  # hand-held / jerky: follows quickly (see SETUP.md, filter tuning)
}
BUILTIN_TEMPLATES = {"synthetic_part"}
SPEEDS = {"fast": (140, 0, 336), "balanced": (168, 0, 448), "precise": (168, 336, 448)}   # learned: (tracking crop, refine crop or 0, global width)


@dataclass
class Settings:
    source: str = "camera"          # camera | demo | replay
    device: int = 0
    recording: str = ""             # file name in recordings/ (source == replay)
    detector: str = "template"      # hand | template
    template: str = "synthetic_part"
    learned: str = ""               # name of a taught object (detector == learned)
    obj_score: float = 0.6
    speed: str = "balanced"         # fast | balanced | precise (learned detector)
    preset: str = "band"            # band | hand
    min_score: float = 0.85
    horizon_ms: float = 100.0
    coast_ms: float = 300.0
    auto_lock: bool = False
    output_enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 5005
    record: bool = False
    auto_scene: bool = True         # save the last seconds automatically when a locked object is lost

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
            if key == "detector" and value not in ("hand", "template", "learned"):
                raise ValueError("detector must be hand, template or learned")
            if key == "speed" and value not in SPEEDS:
                raise ValueError("unknown speed")
            if key == "preset" and value not in PRESETS:
                raise ValueError("unknown preset")
            if key == "min_score":
                value = min(max(value, 0.3), 0.99)
            if key == "obj_score":
                value = min(max(value, 0.2), 0.95)
            if key == "learned" and value and not NAME_RE.match(value):
                raise ValueError("invalid object name")
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
RESTART_KEYS = {"source", "device", "recording", "detector", "template", "learned"}


RING_S = 15.0           # seconds kept for "Szene sichern"
AUTO_SCENE_GAP_S = 30.0  # at most one automatic clip per this many seconds
MAX_AUTO_SCENES = 10


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
        self.objects_dir = root / "models" / "objects"
        self._vision_lock = threading.Lock()
        self._sam: SamSegmenter | None = None
        self._dino: DinoFeatures | None = None
        self._sam_frame_id: int | None = None
        self._learn_mask: np.ndarray | None = None
        self._learn_samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._learn_thumbs: list[bytes] = []
        self._learn_empty: list[np.ndarray] = []      # camera frames without the object (real negatives)
        self._empty_capturing = False
        self._learn = {"phase": "idle", "progress": 0.0, "error": None}
        self._obj_cache: tuple[tuple, list[dict[str, Any]]] | None = None
        self.settings = self._load_settings()
        self.status: dict[str, Any] = self._idle_status()

    # ---- settings ---------------------------------------------------------------------------
    def _load_settings(self) -> Settings:
        s = Settings()
        try:
            data = json.loads(self.settings_path.read_text(encoding="utf-8"))
            data.pop("record", None)
            data.pop("auto_lock", None)
            data.pop("output_enabled", None)  # never start sending / recording implicitly
            s.update(data)
        except (OSError, ValueError, TypeError):
            pass
        if s.detector == "template" and not self._template_path(s.template).exists():
            s.template = "synthetic_part"
        if s.detector == "learned" and not (self.objects_dir / f"{s.learned}.npz").exists():
            s.detector, s.learned = "template", ""
        return s

    def _save_settings(self) -> None:
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(asdict(self.settings), indent=1), encoding="utf-8")
        except OSError:
            pass

    def update_settings(self, data: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            changed = self.settings.update(data)
            if "detector" in changed and "preset" not in data:
                # Sensible default motion model per detector; the user can override it.
                self.settings.update({"preset": "band" if self.settings.detector == "template" else "hand"})
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
        if name not in ("lock", "reset", "save_scene"):
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
                                   "velocity": [250.0, 0.0], "omega": 0.3, "start": [100.0, 360.0, 0.0],
                                   "image_noise": 6.0, "jitter_s": 0.002, "realtime": True}
        else:
            path = self._recording_path(s.recording)
            if path is None:
                raise ValueError("Keine Aufnahme ausgewählt")
            overrides["camera"] = {"type": "video_file", "path": str(path), "realtime": True, "loop": True}

        if s.detector == "hand":
            overrides["detector"] = {"type": "hand", "min_confidence": 0.6,
                                     "model_path": str(self.root / "models" / "hand_landmarker.task")}
        elif s.detector == "learned" and not s.learned:
            overrides["detector"] = {"type": "none"}   # nothing taught yet: live preview so the first photo can be taken
        elif s.detector == "learned":
            path = self.objects_dir / f"{s.learned}.npz"
            if not path.exists():
                raise ValueError(f"Gelerntes Objekt '{s.learned}' nicht gefunden – bitte zuerst anlernen")
            if not vision_models_present(self.root / "models"):
                raise ValueError("Die Bild-Modelle fehlen – in der Oberfläche 'Modelle laden' drücken")
            overrides["detector"] = {"type": "learned", "model_path": str(path), "models_dir": str(self.root / "models"),
                                     "min_score": s.obj_score, "view_size": SPEEDS[s.speed][0],
                                     "refine_size": SPEEDS[s.speed][1], "global_width": SPEEDS[s.speed][2]}
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
        # Heavy learned detector on a live camera: run it in its own thread so camera/output keep their rate.
        overrides["pipeline"] = {"async_detect": s.detector == "learned" and bool(s.learned) and s.source != "replay"}
        return load_config(None, overrides)

    def _loop(self, s: Settings) -> None:
        pipeline = metrics = recorder = publisher = None
        null_publisher = NullPublisher()
        applied: dict[str, Any] = {}
        sent = 0
        ring: collections.deque = collections.deque()   # last RING_S seconds of raw frames (JPEG) for "Szene sichern"
        prev_state = TrackState.SEARCHING
        last_auto = 0.0
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
                    if cmd == "save_scene":
                        self._save_scene(ring, "manuell")
                    else:
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
                if live.source != "replay":
                    self._ring_add(ring, step)
                    if live.auto_scene and prev_state in (TrackState.TRACKING, TrackState.COASTING) \
                            and step.state is TrackState.LOST and time.time() - last_auto > AUTO_SCENE_GAP_S:
                        last_auto = time.time()
                        self._save_scene(ring, "verloren")
                prev_state = step.state
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
        if live.speed != applied.get("speed") and hasattr(pipeline.detector, "view_size"):
            pipeline.detector.configure(*SPEEDS[live.speed])
        applied["speed"] = live.speed
        score = live.obj_score if live.detector == "learned" else live.min_score
        if score != applied.get("score"):
            if hasattr(pipeline.detector, "min_score"):
                pipeline.detector.min_score = score
            applied["score"] = score

    # ---- scene buffer ("black box") ---------------------------------------------------------
    @staticmethod
    def _ring_add(ring: collections.deque, step) -> None:
        f = step.frame
        ok, buf = cv2.imencode(".jpg", f.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            return
        pose, det = step.pose, step.detection
        ring.append((f.frame_id, f.t_exposure, buf.tobytes(), step.state.value,
                     det.confidence if det else "", pose.x if pose else "", pose.y if pose else "",
                     math.degrees(pose.theta) if pose else ""))
        while ring and f.t_exposure - ring[0][1] > RING_S:
            ring.popleft()

    def _save_scene(self, ring: collections.deque, reason: str) -> None:
        """Write the buffered seconds as <recordings>/szene_*.mp4 (+ .csv timestamps, .log.csv states) in the background."""
        items = list(ring)
        if len(items) < 5:
            return
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = self.recordings_dir / f"szene_{'auto_' if reason == 'verloren' else ''}{stamp}"
        info = {"reason": reason, **{k: v for k, v in asdict(self.settings).items() if k in
                ("source", "detector", "learned", "speed", "obj_score", "min_score", "template")}}

        def work() -> None:
            rec = Recorder(base)
            try:
                with open(base.with_suffix(".log.csv"), "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["frame_id", "t_exposure", "state", "score", "x", "y", "theta_deg"])
                    for fid, t, jpg, *meta in items:
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        rec.write(Frame(img, t, fid))
                        w.writerow([fid, f"{t:.6f}", *meta])
            finally:
                rec.close()
            base.with_suffix(".json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
            with self._lock:
                self.status["last_scene"] = base.name
            self._prune_auto_scenes()

        threading.Thread(target=work, daemon=True, name="ctrack-save-scene").start()

    def _prune_auto_scenes(self) -> None:
        """Keep the newest MAX_AUTO_SCENES automatic clips (only files this feature created)."""
        clips = sorted(self.recordings_dir.glob("szene_auto_*.mp4"))
        for old in clips[:-MAX_AUTO_SCENES]:
            for ext in (".mp4", ".csv", ".log.csv", ".json"):
                old.with_suffix("").with_name(old.stem + ext).unlink(missing_ok=True)

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
            "latency_p95": _num(stats["latency_p95"]), "perception_p95": _num(stats["perception_p95"]),
            "pred_err_p95": _num(stats["pred_err_p95"]),
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
                "detected": False, "score": None, "fps": None, "latency_p50": None, "latency_p95": None, "perception_p95": None,
                "pred_err_p95": None, "pred_err_pct": None, "pose": None, "velocity": None, "frame_size": None,
                "frames": 0, "sending": False, "packets_sent": 0, "recording": False, "csv": None, "last_scene": None,
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
                    "learned": self.list_learned(), "vision_models": vision_models_present(self.root / "models"),
                    "accel": accelerator_status(),
                    "learn": {**self._learn, "count": len(self._learn_samples), "has_frame": self._teach_frame is not None,
                              "empty": len(self._learn_empty), "empty_capturing": self._empty_capturing,
                              "has_mask": self._learn_mask is not None},
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

    # ---- learned objects: capture -> click -> add -> train ------------------------------------
    def download_vision_models(self) -> None:
        fetch_vision_models(self.root / "models")

    def _vision(self) -> tuple[SamSegmenter, DinoFeatures]:
        if not vision_models_present(self.root / "models"):
            raise ValueError("Die Bild-Modelle fehlen – bitte 'Modelle laden' drücken")
        if self._sam is None:
            self._sam = SamSegmenter(self.root / "models")
        if self._dino is None:
            self._dino = DinoFeatures(self.root / "models")
        return self._sam, self._dino

    def learn_new_photo(self) -> None:
        """A new photo was frozen / uploaded: forget clicks of the previous one."""
        self._learn_mask = None
        self._sam_frame_id = None

    def learn_segment(self, points: list[list[float]], labels: list[int]) -> float:
        """Run SAM for the current photo with the given clicks; returns the mask's share of the image."""
        frame = self._teach_frame
        if frame is None:
            raise ValueError("Kein Foto")
        if not points or len(points) != len(labels) or len(points) > 20:
            raise ValueError("Ungültige Klicks")
        h, w = frame.shape[:2]
        pts = [(min(max(float(x), 0), w - 1), min(max(float(y), 0), h - 1)) for x, y in points]
        with self._vision_lock:
            sam, _ = self._vision()
            if self._sam_frame_id != id(frame):
                sam.set_image(frame)
                self._sam_frame_id = id(frame)
            self._learn_mask = sam.segment(pts, [1 if v else 0 for v in labels])
        return float(self._learn_mask.mean())

    def learn_mask_png(self) -> bytes | None:
        m = self._learn_mask
        if m is None:
            return None
        rgba = np.zeros((*m.shape, 4), np.uint8)
        rgba[m] = (200, 190, 40, 120)                       # BGRA: teal fill
        edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        rgba[edge] = (200, 190, 40, 255)
        return cv2.imencode(".png", rgba)[1].tobytes()

    def _thumb(self, img: np.ndarray, mask: np.ndarray, size: int = 160) -> bytes:
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        pad = int(0.15 * max(w, h))
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        crop = img[y0:y + h + pad, x0:x + w + pad].copy()
        cm = mask[y0:y + h + pad, x0:x + w + pad]
        crop[~cm] = (crop[~cm] * 0.35).astype(np.uint8)       # dim everything outside the mask
        s = size / max(crop.shape[:2])
        return cv2.imencode(".jpg", cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA))[1].tobytes()

    def learn_add(self) -> int:
        frame, mask = self._teach_frame, self._learn_mask
        if frame is None or mask is None:
            raise ValueError("Zuerst auf das Objekt klicken")
        if mask.mean() < 0.002:
            raise ValueError("Die Maske ist zu klein")
        if len(self._learn_samples) >= 12:
            raise ValueError("Maximal 12 Fotos")
        self._learn_samples.append((frame.copy(), mask.copy()))
        self._learn_thumbs.append(self._thumb(frame, mask))
        self._teach_frame, self._learn_mask, self._sam_frame_id = None, None, None
        return len(self._learn_samples)

    def learn_remove(self, index: int) -> None:
        if 0 <= index < len(self._learn_samples):
            del self._learn_samples[index], self._learn_thumbs[index]

    def learn_empty_start(self, n: int = 12, seconds: float = 3.0) -> None:
        """Grab n live frames over `seconds` (point the camera at the scene WITHOUT the object)."""
        if not self._running() or self._last_raw is None:
            raise ValueError("Kamera läuft nicht – zuerst starten")
        if self._empty_capturing:
            raise ValueError("Aufnahme läuft bereits")
        n = int(min(max(n, 3), 24))
        self._empty_capturing = True

        def work() -> None:
            try:
                for _ in range(n):
                    time.sleep(seconds / n)
                    with self._frame_cv:
                        frame = self._last_raw
                    if frame is not None:
                        self._learn_empty.append(frame.copy())
                del self._learn_empty[:-36]          # keep the newest 36
            finally:
                self._empty_capturing = False

        threading.Thread(target=work, daemon=True, name="ctrack-empty").start()

    def learn_empty_clear(self) -> None:
        self._learn_empty.clear()

    def learn_reset(self) -> None:
        self._learn_samples.clear()
        self._learn_thumbs.clear()
        self._learn_empty.clear()
        self._teach_frame, self._learn_mask, self._sam_frame_id = None, None, None
        self._learn.update(phase="idle", progress=0.0, error=None)

    def learn_sample_jpg(self, i: int) -> bytes | None:
        return self._learn_thumbs[i] if 0 <= i < len(self._learn_thumbs) else None

    def learn_train(self, name: str) -> None:
        if not NAME_RE.match(name):
            raise ValueError("Name: 1–40 Zeichen, nur Buchstaben, Ziffern, _ und -")
        if len(self._learn_samples) < 2:
            raise ValueError("Mindestens 2 Fotos (empfohlen: 5 mit verschiedenem Abstand und leichter Drehung)")
        if self._learn["phase"] == "training":
            raise ValueError("Training läuft bereits")
        samples = list(self._learn_samples)
        empties = list(self._learn_empty)
        thumb = self._learn_thumbs[0]
        self._learn.update(phase="training", progress=0.0, error=None)

        def work() -> None:
            try:
                with self._vision_lock:
                    _, dino = self._vision()
                    model = train_object_model(samples, dino, name, empty_scenes=empties,
                                               progress=lambda f: self._learn.update(progress=float(f)))
                model.save(self.objects_dir / f"{name}.npz")
                (self.objects_dir / f"{name}.jpg").write_bytes(thumb)
                self._learn_samples.clear()
                self._learn_thumbs.clear()
                self._learn_empty.clear()
                self._learn.update(phase="done", progress=1.0)
                self.update_settings({"detector": "learned", "learned": name})
            except Exception as e:  # shown in the GUI
                self._learn.update(phase="error", error=f"{type(e).__name__}: {e}")

        threading.Thread(target=work, daemon=True, name="ctrack-train").start()

    def list_learned(self) -> list[dict[str, Any]]:
        files = sorted(self.objects_dir.glob("*.npz")) if self.objects_dir.exists() else []
        sig = tuple((p.name, p.stat().st_mtime_ns) for p in files)
        if self._obj_cache is not None and self._obj_cache[0] == sig:
            return self._obj_cache[1]
        out = []
        for p in files:
            try:
                m = ObjectModel.load(p)
            except Exception:
                continue
            out.append({"name": p.stem, "images": m.n_images, "created": m.created, "version": p.stat().st_mtime_ns})
        self._obj_cache = (sig, out)
        return out

    def learned_jpg(self, name: str) -> bytes | None:
        p = self.objects_dir / f"{name}.jpg"
        return p.read_bytes() if NAME_RE.match(name) and p.exists() else None

    def delete_learned(self, name: str) -> None:
        if not NAME_RE.match(name):
            raise ValueError("invalid name")
        (self.objects_dir / f"{name}.npz").unlink(missing_ok=True)
        (self.objects_dir / f"{name}.jpg").unlink(missing_ok=True)
        if self.settings.learned == name:
            self.update_settings({"detector": "template", "learned": ""})

    # ---- templates / recordings -------------------------------------------------------------
    def _template_path(self, name: str) -> Path:
        return self.templates_dir / f"{name}.png"

    def list_templates(self) -> list[dict[str, Any]]:
        files, mtimes = [], {}
        for p in sorted(self.templates_dir.glob("*.png")):
            try:
                mtimes[p] = p.stat().st_mtime_ns
            except FileNotFoundError:      # deleted between listing and stat (e.g. by the user in Finder)
                continue
            files.append(p)
        sig = tuple((p.name, mtimes[p]) for p in files)
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
                        "builtin": p.stem in BUILTIN_TEMPLATES, "version": mtimes[p]})
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

