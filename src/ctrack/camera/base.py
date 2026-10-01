from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Frame


class CameraSource(ABC):
    """Produces timestamped frames. `t_exposure` is mandatory (P2-1)."""

    #: True if t_exposure is on the live wall clock (latency = t_sent - t_exposure is meaningful).
    is_live: bool = True

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or None when the source is exhausted."""

    def close(self) -> None:
        pass

    def __enter__(self) -> CameraSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
