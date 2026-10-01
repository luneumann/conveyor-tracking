#!/usr/bin/env python3
"""Evaluate a ctrack metrics CSV (P1-5).

Writes <csv>_report.png with: latency histogram, prediction error over time,
prediction error vs speed. Prints a summary against the PRD success metrics.

    python tools/analyze.py logs/run.csv [--show]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, np.ndarray]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: no rows")

    def col(name: str) -> np.ndarray:
        return np.array([float(r[name]) if r[name] not in ("", None) else math.nan for r in rows])

    data = {name: col(name) for name in rows[0] if name != "state"}
    data["state"] = np.array([r["state"] for r in rows])
    return data


def summary(d: dict[str, np.ndarray], image_width: float) -> list[str]:
    def pct(a: np.ndarray, q: float) -> float:
        a = a[~np.isnan(a)]
        return float(np.percentile(a, q)) if a.size else math.nan

    t = d["t_read"]
    fps = (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else math.nan
    lat95 = pct(d["latency_ms"], 95)
    err95 = pct(d["pred_err_px"], 95)
    states, counts = np.unique(d["state"], return_counts=True)
    lines = [
        f"frames            {len(t)}",
        f"states            {dict(zip(states.tolist(), counts.tolist()))}",
        f"fps               {fps:.1f}           (goal >= 25, stretch >= 30)",
        f"latency p50/p95   {pct(d['latency_ms'], 50):.1f} / {lat95:.1f} ms   (goal p95 < 100, stretch < 60)",
        f"pred err p50/p95  {pct(d['pred_err_px'], 50):.1f} / {err95:.1f} px = {err95 / image_width * 100:.2f}% "
        f"of {image_width:.0f} px width   (goal < 2%, stretch < 1%)",
        f"pred err deg p95  {pct(d['pred_err_deg'], 95):.2f} deg",
    ]
    if not np.all(np.isnan(d["gt_err_px"])):
        lines.append(f"ground truth p95  {pct(d['gt_err_px'], 95):.2f} px")
    return lines


def plot(d: dict[str, np.ndarray], out: Path, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = d["t_exposure"] - d["t_exposure"][0]
    speed = np.hypot(d["vx"], d["vy"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    lat = d["latency_ms"][~np.isnan(d["latency_ms"])]
    axes[0].hist(lat, bins=40, color="#3b82f6")
    for q, ls in ((50, "-"), (95, "--")):
        axes[0].axvline(np.percentile(lat, q), color="#111", ls=ls, lw=1, label=f"p{q} {np.percentile(lat, q):.1f} ms")
    axes[0].set(title="Latency t_exposure → t_sent", xlabel="ms", ylabel="frames")
    axes[0].legend()

    axes[1].plot(t, d["pred_err_px"], ".", ms=3, color="#ef4444")
    axes[1].set(title="Prediction error over time", xlabel="t [s]", ylabel="px")

    axes[2].plot(speed, d["pred_err_px"], ".", ms=3, color="#8b5cf6")
    axes[2].set(title="Prediction error vs speed", xlabel="speed [px/s]", ylabel="px")

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    if show:
        plt.show()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--image-width", type=float, default=1280.0)
    p.add_argument("--show", action="store_true", help="open the plot window")
    args = p.parse_args(argv)

    d = load(args.csv)
    print("\n".join(summary(d, args.image_width)))
    out = args.csv.with_name(args.csv.stem + "_report.png")
    plot(d, out, args.show)
    print(f"plots: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
