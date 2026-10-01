"""CLI entry point: pipeline loop with keyboard control (L lock, R reset, Q quit)."""

from __future__ import annotations

import argparse
import math
import sys
import time

from .camera.video_file import Recorder
from .config import deep_merge, load_config, parse_override
from .metrics import MetricsLogger
from .pipeline import Pipeline
from .types import TrackState

AUTO_RELOCK_S = 1.0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ctrack", description="Conveyor tracking prototype")
    p.add_argument("--config", "-c", default="config/default.yaml", help="YAML config file")
    p.add_argument("--set", "-s", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config value, e.g. -s predictor.horizon_ms=150 (repeatable)")
    p.add_argument("--headless", action="store_true", help="no window (overrides visualizer.enabled)")
    p.add_argument("--auto-lock", action="store_true",
                   help="lock on the first detection; after LOST for >1 s reset and lock the next object")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--record", metavar="PATH", help="record raw frames to PATH.mp4 + PATH.csv (P1-1)")
    p.add_argument("--no-csv", action="store_true", help="do not write the metrics CSV")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict = {}
    for expr in args.set:
        overrides = deep_merge(overrides, parse_override(expr))
    cfg = load_config(args.config, overrides)
    headless = args.headless or not cfg["visualizer"].get("enabled", True)

    pipeline = Pipeline.from_config(cfg)
    tracker = pipeline.tracker
    metrics = MetricsLogger(csv=None if args.no_csv else cfg["metrics"].get("csv"),
                            match_tolerance_ms=cfg["metrics"]["match_tolerance_ms"],
                            window=cfg["metrics"]["window"])
    recorder = Recorder(args.record) if args.record else None
    visualizer = None
    if not headless:
        from .visualizer import Visualizer

        visualizer = Visualizer(horizon_ms=cfg["predictor"]["horizon_ms"],
                                reacquire_radius_px=cfg["tracker"]["reacquire_radius_px"])

    print(f"ctrack: camera={cfg['camera']['type']} detector={cfg['detector']['type']} "
          f"output={cfg['output']['type']}:{cfg['output'].get('host', '')}:{cfg['output'].get('port', '')}")
    if not args.auto_lock:
        print("Press L (in the video window) to lock onto the object, R to reset, Q to quit.")

    n = 0
    last_print = time.time()
    try:
        while args.max_frames is None or n < args.max_frames:
            if args.auto_lock:
                if tracker.state is TrackState.SEARCHING:
                    tracker.request_lock()
                elif tracker.state is TrackState.LOST and tracker.t_lost is not None:
                    if time.time() - tracker.t_lost > AUTO_RELOCK_S or not pipeline.camera.is_live:
                        tracker.reset()
                        tracker.request_lock()
            step = pipeline.step()
            if step is None:
                break
            n += 1
            if recorder:
                recorder.write(step.frame)
            metrics.log(step)
            stats = metrics.stats()
            if visualizer:
                key = visualizer.show(step, stats, tracker.reference_pose(step.frame.t_exposure))
                if key == "l":
                    tracker.request_lock()
                elif key == "r":
                    tracker.reset()
                elif key in ("q", "\x1b"):
                    break
            elif time.time() - last_print > 1.0:
                last_print = time.time()
                print(_status_line(step.state, stats), end="\r", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.close()
        metrics.close()
        if recorder:
            recorder.close()
        if visualizer:
            visualizer.close()

    stats = metrics.stats()
    print()
    print(f"frames: {n}")
    print(_status_line(tracker.state, stats))
    if not args.no_csv and cfg["metrics"].get("csv"):
        print(f"metrics CSV: {cfg['metrics']['csv']}")
    return 0


def _status_line(state: TrackState, s: dict[str, float]) -> str:
    def fmt(v: float, d: int = 1) -> str:
        return "-" if math.isnan(v) else f"{v:.{d}f}"

    return (f"{state.value:<9} fps {fmt(s['fps'])}  latency p95 {fmt(s['latency_p95'], 0)} ms  "
            f"pred err p95 {fmt(s['pred_err_p95'])} px ({fmt(s['pred_err_p95_pct'], 2)}%)")


if __name__ == "__main__":
    sys.exit(main())
