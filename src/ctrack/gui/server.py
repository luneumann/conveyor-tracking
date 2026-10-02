"""Local web server for the GUI: static page, JSON API and an MJPEG live stream (stdlib only).

Security: binds to 127.0.0.1 only. Because any web page you visit can send requests to localhost, every
request must carry the right Host header (blocks DNS rebinding) and every state-changing request must
carry `X-Requested-With: ctrack` (a custom header forces a CORS preflight, which this server never grants).
"""

from __future__ import annotations

import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .engine import Engine

MAX_BODY = 25 * 1024 * 1024  # uploaded teach images


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    port: int
    protocol_version = "HTTP/1.1"

    # -- plumbing -----------------------------------------------------------------------------
    def log_message(self, fmt: str, *args: object) -> None:  # keep the console quiet
        pass

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _send(self, code: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: object, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _error(self, code: int, message: str) -> None:
        # The request body may be unread (early rejection): do not reuse this connection.
        self.close_connection = True
        self._send(code, json.dumps({"error": message}).encode(), "application/json", {"Connection": "close"})

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise ValueError("Datei zu groß")
        return self.rfile.read(n) if n else b""

    def _json_body(self, raw: bytes) -> dict:
        data = json.loads(raw) if raw else {}
        if not isinstance(data, dict):
            raise ValueError("JSON object expected")
        return data

    # -- GET ----------------------------------------------------------------------------------
    def do_GET(self) -> None:
        if not self._host_ok():
            return self._error(403, "bad host")
        url = urlparse(self.path)
        path = url.path
        try:
            if path in ("/", "/index.html"):
                html = resources.files("ctrack.gui").joinpath("static/index.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if path == "/api/status":
                return self._json(self.engine.snapshot())
            if path == "/stream.mjpg":
                return self._stream()
            if path == "/api/teach/frame.jpg":
                jpg = self.engine.teach_jpeg()
                return self._send(200, jpg, "image/jpeg") if jpg else self._error(404, "no frame")
            if path == "/api/learn/mask.png":
                png = self.engine.learn_mask_png()
                return self._send(200, png, "image/png") if png else self._error(404, "no mask")
            m = re.fullmatch(r"/api/learn/sample/(\d+)\.jpg", path)
            if m:
                jpg = self.engine.learn_sample_jpg(int(m.group(1)))
                return self._send(200, jpg, "image/jpeg") if jpg else self._error(404, "not found")
            m = re.fullmatch(r"/api/learned/([A-Za-z0-9_-]+)\.jpg", path)
            if m:
                jpg = self.engine.learned_jpg(m.group(1))
                return self._send(200, jpg, "image/jpeg") if jpg else self._error(404, "not found")
            m = re.fullmatch(r"/api/templates/([A-Za-z0-9_-]+?)(_mask)?\.png", path)
            if m:
                png = self.engine.template_png(m.group(1), mask=bool(m.group(2)))
                return self._send(200, png, "image/png") if png else self._error(404, "not found")
            m = re.fullmatch(r"/logs/([A-Za-z0-9_.-]+\.csv)", path)
            if m:
                p = self.engine.log_path(m.group(1))
                if p is None:
                    return self._error(404, "not found")
                return self._send(200, p.read_bytes(), "text/csv",
                                  {"Content-Disposition": f'attachment; filename="{p.name}"'})
            return self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        last = -1
        try:
            while True:
                last, jpg, active = self.engine.wait_frame(last, timeout=1.0)
                if jpg is None:
                    if active:
                        continue  # camera still starting: no frame yet
                    return  # session ended: close so the page can show its placeholder
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    # -- POST ---------------------------------------------------------------------------------
    def do_POST(self) -> None:
        if not self._host_ok():
            return self._error(403, "bad host")
        if self.headers.get("X-Requested-With") != "ctrack":
            return self._error(403, "missing X-Requested-With header")
        path = urlparse(self.path).path
        e = self.engine
        try:
            # Always consume the request body first: on a keep-alive connection an unread body would be
            # parsed as the start of the next request.
            raw = self._body()
            if path == "/api/session":
                action = self._json_body(raw).get("action")
                if action == "start":
                    e.start()
                elif action == "stop":
                    e.stop()
                else:
                    return self._error(400, "action must be start or stop")
                return self._json(e.snapshot())
            if path == "/api/settings":
                return self._json(e.update_settings(self._json_body(raw)))
            if path == "/api/command":
                e.command(str(self._json_body(raw).get("name")))
                return self._json({"ok": True})
            if path == "/api/teach/freeze":
                if not e.freeze_frame():
                    return self._error(409, "Kamera läuft nicht – zuerst starten oder ein Bild laden")
                e.learn_new_photo()
                return self._json({"ok": True})
            if path == "/api/teach/upload":
                e.load_teach_image(raw)
                e.learn_new_photo()
                return self._json({"ok": True})
            if path == "/api/learn/segment":
                d = self._json_body(raw)
                return self._json({"area": e.learn_segment(d.get("points") or [], d.get("labels") or [])})
            if path == "/api/learn/empty":
                e.learn_empty_start()
                return self._json({"ok": True})
            if path == "/api/learn/empty/clear":
                e.learn_empty_clear()
                return self._json(e.snapshot())
            if path == "/api/learn/add":
                return self._json({"count": e.learn_add()})
            if path == "/api/learn/remove":
                e.learn_remove(int(self._json_body(raw).get("index", -1)))
                return self._json(e.snapshot())
            if path == "/api/learn/reset":
                e.learn_reset()
                return self._json(e.snapshot())
            if path == "/api/learn/train":
                e.learn_train(str(self._json_body(raw).get("name", "")))
                return self._json({"ok": True})
            if path == "/api/learned/delete":
                e.delete_learned(str(self._json_body(raw).get("name")))
                return self._json(e.snapshot())
            if path == "/api/vision-models":
                try:
                    e.download_vision_models()
                except OSError as ex:
                    return self._error(502, f"Download fehlgeschlagen: {ex}")
                return self._json(e.snapshot())
            if path == "/api/teach/save":
                d = self._json_body(raw)
                roi = d.get("roi")
                if not (isinstance(roi, list) and len(roi) == 4):
                    return self._error(400, "roi must be [x, y, w, h]")
                return self._json(e.save_template(str(d.get("name", "")), roi, bool(d.get("mask_auto", True))))
            if path == "/api/hand-model":
                try:
                    e.download_hand_model()
                except OSError as ex:
                    return self._error(502, f"Download fehlgeschlagen: {ex}")
                return self._json(e.snapshot())
            if path == "/api/templates/delete":
                e.delete_template(str(self._json_body(raw).get("name")))
                return self._json(e.snapshot())
            return self._error(404, "not found")
        except (ValueError, KeyError, TypeError) as ex:
            return self._error(400, str(ex))
        except (BrokenPipeError, ConnectionResetError):
            pass


def serve(root: Path, port: int = 8765, open_browser: bool = True) -> None:
    engine = Engine(root)
    server = None
    for p in range(port, port + 10):
        try:
            Handler.engine, Handler.port = engine, p
            server = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            server.daemon_threads = True
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"Ports {port}-{port + 9} sind belegt. Läuft die Oberfläche schon?")
    url = f"http://127.0.0.1:{Handler.port}/"
    print(f"Conveyor Tracking GUI läuft: {url}\n(Beenden mit Ctrl+C oder Fenster schließen)", flush=True)
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="ctrack-gui", description="Web GUI for the conveyor tracking prototype")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="do not open the browser automatically")
    p.add_argument("--root", type=Path, default=Path.cwd(), help="project folder (templates/, logs/, models/)")
    args = p.parse_args(argv)
    serve(args.root.resolve(), args.port, not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
