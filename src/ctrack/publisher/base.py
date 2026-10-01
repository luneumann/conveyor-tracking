from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

from ..registry import PUBLISHERS
from ..types import Message


class Publisher(ABC):
    """Output plugin. Stage 2 adds robot adapters / encoder emulation (P2-7)."""

    @abstractmethod
    def publish(self, message: Message) -> bool:
        """Send the message. Returns False if it was dropped (e.g. rate limit)."""

    def close(self) -> None:
        pass


@PUBLISHERS.register("none")
class NullPublisher(Publisher):
    def __init__(self, **_: object) -> None:
        # Bounded: a long GUI session must not grow memory without limit.
        self.messages: deque[Message] = deque(maxlen=10000)

    def publish(self, message: Message) -> bool:
        self.messages.append(message)
        return True
