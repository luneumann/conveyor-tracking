import http.client
import json
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
    assert code == 200 and b"Conveyor Tracking" in body


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
