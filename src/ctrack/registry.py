"""Name -> class registries that let YAML config pick module implementations (P0-10)."""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._classes: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._classes:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._classes[name] = cls
            return cls

        return deco

    def names(self) -> list[str]:
        return sorted(self._classes)

    def create(self, cfg: dict[str, Any]) -> T:
        """Instantiate from a config block: `type` selects the class, the rest are kwargs."""
        cfg = dict(cfg)
        name = cfg.pop("type", None)
        if name not in self._classes:
            raise KeyError(f"Unknown {self.kind} type '{name}'. Available: {', '.join(self.names())}")
        return self._classes[name](**cfg)


CAMERAS: Registry = Registry("camera")
DETECTORS: Registry = Registry("detector")
TRANSFORMS: Registry = Registry("transform")
PUBLISHERS: Registry = Registry("publisher")


def load_builtins() -> None:
    """Import built-in modules so their @register decorators run."""
    from . import camera, detector, publisher, transform  # noqa: F401
