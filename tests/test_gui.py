import http.client
import json
import math
import shutil
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np
import pytest

from conftest import ROOT
from ctrack.camera.synthetic import render_object
from ctrack.gui.engine import Engine, Settings
from ctrack.gui.server import Handler
from ctrack.types import Pose

HDR = {"X-Requested-With": "ctrack", "Content-Type": "application/json"}


@pytest.fixture()
def gui(tmp_path):
    (tmp_path / "templates").mkdir()
    shutil.copy(ROOT / "templates" / "synthetic_part.png", tmp_path / "templates")
    engine = Engine(tmp_path)
    Handler.engine = engine
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    Handler.port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    class Client:
        port = Handler.port
        root = tmp_path

        def conn(self):
            return http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)

        def req(self, method, path, body=None, headers=None, conn=None):
            c = conn or self.conn()
            data = body if isinstance(body, bytes) else (json.dumps(body).encode() if body is not None else None)
            c.request(method, path, data, headers if headers is not None else (HDR if method == "POST" else {}))
            r = c.getresponse()
            raw = r.read()
            ctype = r.getheader("Content-Type", "")
            return r.status, (json.loads(raw) if "json" in ctype else raw)

        def post(self, path, body=None):
            return self.req("POST", path, body if body is not None else {})

        def status(self):
            return self.req("GET", "/api/status")[1]

        def wait(self, pred, timeout=20.0):
            t_end = time.time() + timeout
            while time.time() < t_end:
                st = self.status()
                if pred(st):
                    return st
                time.sleep(0.15)
            raise AssertionError(f"timeout; last status: { {k: st.get(k) for k in ('running','state','frames','error')} }")

    yield Client()
    engine.stop()
    server.shutdown()
    server.server_close()


def _png(pose=Pose(300, 200, 0.0)) -> bytes:
    img = np.full((400, 640, 3), 35, np.uint8)
    render_object(img, pose, 160, 70)
    return cv2.imencode(".png", img)[1].tobytes()


def test_index_page_served(gui):
    code, body = gui.req("GET", "/")
    assert code == 200 and b"ctrack" in body and b"Camtrack" in body
    assert b'id="loader"' in body and "Linse putzen".encode() in body       # start-up feedback while models load


def test_rejects_cross_site_requests(gui):
    assert gui.req("POST", "/api/session", {"action": "start"}, headers={"Content-Type": "application/json"})[0] == 403
    assert gui.req("GET", "/api/status", headers={"Host": "evil.example"})[0] == 403  # DNS rebinding
    assert not gui.status()["running"]


@pytest.mark.parametrize("path", ["/logs/..%2f..%2fetc%2fpasswd", "/api/templates/..%2fsecret.png", "/logs/x.txt", "/nope"])
def test_path_traversal_is_not_served(gui, path):
    assert gui.req("GET", path)[0] == 404


def test_settings_roundtrip_and_validation(gui):
    code, d = gui.post("/api/settings", {"source": "demo", "horizon_ms": 150, "min_score": 5})
    assert code == 200 and d["settings"]["source"] == "demo" and d["settings"]["horizon_ms"] == 150
    assert d["settings"]["min_score"] == 0.99  # clamped
    assert gui.post("/api/settings", {"source": "teleport"})[0] == 400
    assert gui.post("/api/settings", {"port": 70000})[0] == 400
    assert Settings().update({"unknown": 1}) == set()
    saved = json.loads((gui.root / "logs" / "gui_settings.json").read_text())
    assert saved["source"] == "demo"


def test_settings_never_restore_sending_or_recording(gui):
    gui.post("/api/settings", {"output_enabled": True, "record": True, "auto_lock": True})
    fresh = Engine(gui.root)
    assert not fresh.settings.output_enabled and not fresh.settings.record and not fresh.settings.auto_lock


def test_post_body_is_consumed_on_keep_alive_connection(gui):
    """Regression: an unread POST body corrupted the next request on the same connection (HTTP 501)."""
    c = gui.conn()
    assert gui.req("POST", "/api/settings", {"horizon_ms": 120}, conn=c)[0] == 200
    assert gui.req("GET", "/api/status", conn=c)[0] == 200
    assert gui.req("POST", "/api/teach/upload", _png(), headers={"X-Requested-With": "ctrack"}, conn=c)[0] == 200
    assert gui.req("GET", "/api/teach/frame.jpg", conn=c)[0] == 200


def test_demo_session_track_stream_and_stop(gui):
    gui.post("/api/settings", {"source": "demo", "detector": "template", "template": "synthetic_part"})
    assert gui.post("/api/session", {"action": "start"})[0] == 200
    gui.wait(lambda s: s["running"] and s["frames"] > 3)
    gui.post("/api/command", {"name": "lock"})
    st = gui.wait(lambda s: s["state"] == "TRACKING")
    assert st["pose"] is not None and st["fps"] > 5
    assert 150 < st["velocity"]["speed"] < 350  # the demo belt moves at ~250 px/s

    c = gui.conn()  # live MJPEG stream delivers JPEG frames
    c.request("GET", "/stream.mjpg")
    r = c.getresponse()
    assert r.status == 200 and "multipart/x-mixed-replace" in r.getheader("Content-Type")
    head = r.read(600)
    assert b"--frame" in head and b"image/jpeg" in head
    c.close()

    code, d = gui.post("/api/session", {"action": "stop"})
    assert code == 200 and not d["running"]
    run = d["last_run"]
    assert run["frames"] > 5 and run["csv"].startswith("gui_")
    code, csv_bytes = gui.req("GET", "/logs/" + run["csv"])
    assert code == 200 and csv_bytes.startswith(b"frame_id,")


def test_udp_output_and_recording_toggles(gui):
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(5)
    gui.post("/api/settings", {"source": "demo", "port": rx.getsockname()[1], "output_enabled": True})
    gui.post("/api/session", {"action": "start"})
    msg = json.loads(rx.recv(65535))
    assert msg["v"] == 1 and msg["state"] in ("SEARCHING", "TRACKING") and "t_exposure" in msg
    st = gui.wait(lambda s: s["sending"] and s["packets_sent"] > 2)
    gui.post("/api/settings", {"output_enabled": False})
    gui.wait(lambda s: not s["sending"])
    gui.post("/api/session", {"action": "stop"})
    rx.close()


def test_camera_error_is_reported_not_raised(gui):
    gui.post("/api/settings", {"source": "replay", "recording": ""})
    gui.post("/api/session", {"action": "start"})
    st = gui.wait(lambda s: s["error"] is not None and not s["running"])
    assert "Aufnahme" in st["error"]


def test_teach_from_upload_select_and_delete(gui):
    assert gui.post("/api/teach/freeze")[0] == 409  # no session running
    assert gui.post("/api/teach/upload", b"not an image")[0] == 400
    code, _ = gui.req("POST", "/api/teach/upload", _png(), headers={"X-Requested-With": "ctrack"})
    assert code == 200
    code, jpg = gui.req("GET", "/api/teach/frame.jpg")
    assert code == 200 and jpg[:2] == b"\xff\xd8"

    assert gui.post("/api/teach/save", {"name": "../evil", "roi": [195, 145, 211, 111]})[0] == 400
    assert gui.post("/api/teach/save", {"name": "synthetic_part", "roi": [195, 145, 211, 111]})[0] == 400  # built-in
    assert gui.post("/api/teach/save", {"name": "ok", "roi": [600, 300, 200, 200]})[0] == 400  # outside image
    code, d = gui.post("/api/teach/save", {"name": "mein_teil", "roi": [195, 145, 211, 111], "mask_auto": True})
    assert code == 200 and d["width"] == 211 and 0.4 < d["mask_coverage"] < 0.6

    st = gui.status()
    assert st["settings"]["template"] == "mein_teil" and st["settings"]["detector"] == "template"
    names = {t["name"]: t for t in st["templates"]}
    assert names["mein_teil"]["has_mask"] and not names["mein_teil"]["builtin"] and names["synthetic_part"]["builtin"]
    assert gui.req("GET", "/api/templates/mein_teil.png")[1][:4] == b"\x89PNG"
    assert gui.req("GET", "/api/templates/mein_teil_mask.png")[0] == 200

    assert gui.post("/api/templates/delete", {"name": "synthetic_part"})[0] == 400
    code, d = gui.post("/api/templates/delete", {"name": "mein_teil"})
    assert code == 200 and d["settings"]["template"] == "synthetic_part"
    assert not (gui.root / "templates" / "mein_teil.png").exists()
    assert not (gui.root / "templates" / "mein_teil_mask.png").exists()


def test_taught_template_detects_in_session(gui):
    """Teach on an uploaded image, then run a session with a replay-less demo: detector uses the new template."""
    gui.req("POST", "/api/teach/upload", _png(), headers={"X-Requested-With": "ctrack"})
    gui.post("/api/teach/save", {"name": "t1", "roi": [195, 145, 211, 111], "mask_auto": True})
    gui.post("/api/settings", {"source": "demo"})
    gui.post("/api/session", {"action": "start"})
    st = gui.wait(lambda s: s["running"] and s["detected"], timeout=25)
    assert st["score"] > 0.85
    gui.post("/api/session", {"action": "stop"})


# ---------------------------------------------------------------------------------------------------------
# Learned objects: photo -> one click (SAM) -> add -> train (DINOv2 head) -> detect
VISION = (ROOT / "models" / "mobile_sam_image_encoder.onnx").exists() and (ROOT / "models" / "dinov2_small.onnx").exists()
needs_vision = pytest.mark.skipif(not VISION, reason="MobileSAM / DINOv2 ONNX models not downloaded")


@pytest.fixture()
def gui_vision(gui):
    (gui.root / "models").mkdir(exist_ok=True)
    for f in ("mobile_sam_image_encoder.onnx", "sam_mask_decoder_single.onnx", "dinov2_small.onnx"):
        (gui.root / "models" / f).symlink_to(ROOT / "models" / f)
    return gui


def _photo(scale: float, theta: float, seed: int) -> bytes:
    """Textured background with the synthetic part at a given zoom / rotation (centre of the image)."""
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.normal(70, 30, (400, 640, 1)).astype(np.float32) * np.ones((1, 1, 3), np.float32), (0, 0), 4)
    img = np.clip(bg, 0, 255).astype(np.uint8)
    render_object(img, Pose(320, 200, theta), 160 * scale, 70 * scale)
    return cv2.imencode(".png", img)[1].tobytes()


@needs_vision
def test_learn_flow_click_segment_add_train_and_detect(gui_vision):
    g = gui_vision
    st = g.status()
    assert st["vision_models"] and st["learn"]["count"] == 0
    for i, (sc, th) in enumerate([(1.0, 0.0), (0.7, 0.35), (1.3, -0.3)]):
        assert g.req("POST", "/api/teach/upload", _photo(sc, th, i), headers={"X-Requested-With": "ctrack"})[0] == 200
        assert g.req("GET", "/api/learn/mask.png")[0] == 404          # new photo: no mask yet
        # One click often gives only part of the object; a second click on the missing part completes it
        # (this is exactly what the GUI invites the user to do).
        c, s_ = math.cos(th), math.sin(th)
        second = [320 + c * 0.3 * 160 * sc, 200 + s_ * 0.3 * 160 * sc]
        code, d = g.post("/api/learn/segment", {"points": [[320, 200], second], "labels": [1, 1]})
        expected = 160 * 70 * sc * sc / (640 * 400)
        assert code == 200 and 0.8 * expected < d["area"] < 1.2 * expected  # the whole part, not more
        code, png = g.req("GET", "/api/learn/mask.png")
        assert code == 200 and png[:4] == b"\x89PNG"
        code, d = g.post("/api/learn/add")
        assert code == 200 and d["count"] == i + 1
        assert g.req("GET", f"/api/learn/sample/{i}.jpg")[1][:2] == b"\xff\xd8"
    assert g.post("/api/learn/add")[0] == 400                          # nothing selected any more
    assert g.post("/api/learn/train", {"name": "../x"})[0] == 400

    assert g.post("/api/learn/train", {"name": "synth"})[0] == 200
    st = g.wait(lambda s: s["learn"]["phase"] in ("done", "error"), timeout=60)
    assert st["learn"]["phase"] == "done", st["learn"]
    assert st["settings"]["detector"] == "learned" and st["settings"]["learned"] == "synth"
    assert [o["name"] for o in st["learned"]] == ["synth"] and st["learned"][0]["images"] == 3
    assert g.req("GET", "/api/learned/synth.jpg")[1][:2] == b"\xff\xd8"
    assert st["learn"]["count"] == 0                                   # samples consumed

    g.post("/api/settings", {"source": "demo"})
    g.post("/api/session", {"action": "start"})
    st = g.wait(lambda s: s["running"] and s["detected"], timeout=40)
    assert st["score"] >= 0.6
    g.post("/api/command", {"name": "lock"})
    st = g.wait(lambda s: s["state"] == "TRACKING", timeout=20)
    assert st["pose"] is not None
    g.post("/api/session", {"action": "stop"})

    code, d = g.post("/api/learned/delete", {"name": "synth"})
    assert code == 200 and d["learned"] == [] and d["settings"]["detector"] == "template"
    assert not (g.root / "models" / "objects" / "synth.npz").exists()


@needs_vision
def test_learn_validation(gui_vision):
    g = gui_vision
    assert g.post("/api/learn/segment", {"points": [[1, 1]], "labels": [1]})[0] == 400      # no photo
    g.req("POST", "/api/teach/upload", _photo(1.0, 0.0, 0), headers={"X-Requested-With": "ctrack"})
    assert g.post("/api/learn/segment", {"points": [], "labels": []})[0] == 400
    assert g.post("/api/learn/segment", {"points": [[1, 1]], "labels": [1, 0]})[0] == 400
    assert g.post("/api/learn/train", {"name": "ok"})[0] == 400                              # no samples
    assert g.post("/api/learned/delete", {"name": "../evil"})[0] == 400
    assert g.req("GET", "/api/learned/..%2fsecret.jpg")[0] == 404
    assert g.post("/api/learn/reset")[0] == 200


def test_learn_without_models_is_a_clear_error(gui):
    gui.req("POST", "/api/teach/upload", _photo(1.0, 0.0, 0), headers={"X-Requested-With": "ctrack"})
    code, d = gui.post("/api/learn/segment", {"points": [[320, 200]], "labels": [1]})
    assert code == 400 and "Modelle" in d["error"]
    assert not gui.status()["vision_models"]


def test_empty_scene_capture_requires_running_camera(gui):
    code, d = gui.post("/api/learn/empty")
    assert code == 400 and "Kamera" in d["error"]
    assert gui.post("/api/learn/empty/clear")[0] == 200


def test_empty_scene_capture_collects_frames_and_can_be_cleared(gui):
    gui.post("/api/settings", {"source": "demo", "detector": "template", "template": "synthetic_part"})
    gui.post("/api/session", {"action": "start"})
    gui.wait(lambda s: s["running"] and s["frames"] > 3)
    assert gui.post("/api/learn/empty")[0] == 200
    assert gui.post("/api/learn/empty")[0] == 400                    # already capturing
    st = gui.wait(lambda s: s["learn"]["empty"] >= 3 and not s["learn"]["empty_capturing"], timeout=30)
    assert 3 <= st["learn"]["empty"] <= 20
    code, d = gui.post("/api/learn/empty/clear")
    assert code == 200 and d["learn"]["empty"] == 0
    gui.post("/api/session", {"action": "stop"})


def test_save_scene_writes_the_buffered_seconds(gui):
    gui.post("/api/settings", {"source": "demo", "auto_lock": False})
    gui.post("/api/session", {"action": "start"})
    gui.wait(lambda st: st["running"] and st["frames"] >= 20)
    assert gui.post("/api/command", {"name": "save_scene"})[0] == 200
    st = gui.wait(lambda st: st.get("last_scene"))
    base = gui.root / "recordings" / st["last_scene"]
    for ext in (".mp4", ".csv", ".log.csv", ".json"):
        assert base.with_suffix(ext).exists() or (gui.root / "recordings" / (st["last_scene"] + ext)).exists(), ext
    cap = cv2.VideoCapture(str(base.with_suffix(".mp4")))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) >= 5
    assert json.loads(base.with_suffix(".json").read_text())["reason"] == "manuell"
    assert st["last_scene"] in gui.req("GET", "/api/status")[1].get("last_scene", "")


def test_old_automatic_scenes_are_pruned(gui):
    from ctrack.gui.engine import MAX_AUTO_SCENES
    d = gui.root / "recordings"
    d.mkdir()
    for i in range(MAX_AUTO_SCENES + 3):
        for ext in (".mp4", ".csv", ".log.csv", ".json"):
            (d / f"szene_auto_20260101_0000{i:02d}{ext}").write_text("x")
    (d / "szene_20260101_000000.mp4").write_text("x")          # manual clip must never be removed
    Engine(gui.root)._prune_auto_scenes()
    assert len(list(d.glob("szene_auto_*.mp4"))) == MAX_AUTO_SCENES
    assert not (d / "szene_auto_20260101_000000.json").exists() and (d / "szene_20260101_000000.mp4").exists()


def test_template_list_survives_a_file_vanishing_mid_listing(gui, monkeypatch):
    from pathlib import Path
    real = Path.stat

    def flaky(self, *a, **k):
        if self.name == "ghost.png":
            raise FileNotFoundError(self)
        return real(self, *a, **k)

    (gui.root / "templates" / "ghost.png").write_bytes(b"x")
    monkeypatch.setattr(Path, "stat", flaky)
    names = [t["name"] for t in Engine(gui.root).list_templates()]
    assert "synthetic_part" in names and "ghost" not in names


def test_screen_recordings_in_other_containers_can_be_replayed(gui):
    """Regression: only *.mp4 was listed, so a macOS screen recording (.mov, name with spaces) could not be loaded."""
    d = gui.root / "recordings"
    d.mkdir()
    name = "Bildschirmaufnahme 2026-10-03 um 14.06.09.mov"
    w = cv2.VideoWriter(str(d / name), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (320, 180))
    for i in range(30):
        w.write(np.full((180, 320, 3), 40 + i, np.uint8))
    w.release()
    (d / "notes.txt").write_text("x")
    listed = gui.req("GET", "/api/status")[1]["recordings"]
    assert name in listed and "notes.txt" not in listed
    gui.post("/api/settings", {"source": "replay", "recording": name, "detector": "template", "template": "synthetic_part"})
    gui.post("/api/session", {"action": "start"})
    st = gui.wait(lambda s: s["frames"] >= 10 or s["error"], timeout=30)
    assert st["error"] is None and st["frames"] >= 10
    gui.post("/api/session", {"action": "stop"})
