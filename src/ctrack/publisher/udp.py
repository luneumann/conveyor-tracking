from __future__ import annotations

import json
import socket

from ..registry import PUBLISHERS
from ..types import Message
from .base import Publisher


@PUBLISHERS.register("udp")
class UdpPublisher(Publisher):
    """One JSON datagram per message (PRD 5.4, ADR-006). rate_hz=0 sends every frame."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5005, rate_hz: float = 0.0) -> None:
        self.addr = (host, port)
        self.min_interval = 1.0 / rate_hz if rate_hz > 0 else 0.0
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._last_sent: float | None = None

    def publish(self, message: Message) -> bool:
        if self._last_sent is not None and message.t_sent - self._last_sent < self.min_interval:
            return False
        payload = json.dumps(message.to_dict(), separators=(",", ":")).encode()
        try:
            self.sock.sendto(payload, self.addr)
        except OSError:
            return False
        self._last_sent = message.t_sent
        return True

    def close(self) -> None:
        self.sock.close()
