#!/usr/bin/env python3
"""Reference receiver for the ctrack UDP stream (P0-9).

Shows message rate, tracking state, packet loss (gaps in seq) and latency t_exposure -> receive.
Standalone: only uses the standard library, so it can run on any machine (e.g. a robot PC).

    python tools/receiver.py --port 5005
"""

from __future__ import annotations

import argparse
import json
import signal
import socket
import sys
import time
from collections import Counter, deque


class StreamStats:
    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self.received = 0
        self.lost = 0
        self.reordered = 0
        self.last_seq: int | None = None
        self.states: Counter[str] = Counter()
        self._arrivals: deque[float] = deque()
        self._latency: deque[float] = deque(maxlen=200)
        self.last: dict | None = None

    def add(self, msg: dict, t_recv: float) -> None:
        seq = int(msg["seq"])
        if self.last_seq is not None:
            if seq > self.last_seq + 1:
                self.lost += seq - self.last_seq - 1
            elif seq <= self.last_seq:
                if seq < self.last_seq - 1000:  # sender restarted
                    self.lost = 0
                    self.received = 0
                else:
                    self.reordered += 1
                    return
        self.last_seq = seq
        self.received += 1
        self.states[msg["state"]] += 1
        self._arrivals.append(t_recv)
        while self._arrivals and self._arrivals[0] < t_recv - self.window_s:
            self._arrivals.popleft()
        self._latency.append((t_recv - float(msg["t_exposure"])) * 1000.0)
        self.last = msg

    @property
    def rate_hz(self) -> float:
        if len(self._arrivals) < 2:
            return 0.0
        span = self._arrivals[-1] - self._arrivals[0]
        return (len(self._arrivals) - 1) / span if span > 0 else 0.0

    @property
    def loss_pct(self) -> float:
        total = self.received + self.lost
        return 100.0 * self.lost / total if total else 0.0

    def latency_p(self, q: float) -> float:
        if not self._latency:
            return float("nan")
        values = sorted(self._latency)
        return values[min(int(q / 100 * len(values)), len(values) - 1)]

    def line(self) -> str:
        m = self.last or {}
        pose = m.get("pose")
        pose_s = f"x {pose['x']:7.1f} y {pose['y']:7.1f} th {pose['theta']:+.2f}" if pose else "pose -" + " " * 28
        return (f"{m.get('state', '-'):<9} {self.rate_hz:5.1f} Hz  seq {m.get('seq', '-'):>6}  "
                f"lost {self.lost} ({self.loss_pct:.2f}%)  lat p50 {self.latency_p(50):5.1f} "
                f"p95 {self.latency_p(95):5.1f} ms  {pose_s} [{m.get('unit', '')}]")


def _raise_interrupt(*_: object) -> None:
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--raw", action="store_true", help="print every message as JSON")
    args = p.parse_args(argv)
    # Print the summary on `kill` too, not only on Ctrl+C.
    signal.signal(signal.SIGTERM, _raise_interrupt)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.5)
    print(f"listening on udp://{args.host}:{args.port}  (Ctrl+C to stop)")
    stats = StreamStats()
    last_print = 0.0
    try:
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                if stats.last is not None:
                    print(f"\r{'(no data)':<120}", end="", flush=True)
                continue
            t_recv = time.time()
            msg = json.loads(data)
            stats.add(msg, t_recv)
            if args.raw:
                print(json.dumps(msg))
            elif t_recv - last_print > 0.1:
                last_print = t_recv
                print(f"\r{stats.line():<120}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    print(f"\nreceived {stats.received}, lost {stats.lost} ({stats.loss_pct:.2f}%), reordered {stats.reordered}")
    print("states:", dict(stats.states))
    return 0


if __name__ == "__main__":
    sys.exit(main())
