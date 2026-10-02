"""Measure the learned detector on YOUR recording: accuracy of mask/pose and false alarms vs. score threshold.

    python tools/eval_recording.py tools/eval_labels_demo.json [--video recordings/x.mp4] [--objects phone,square]

The labels file lists, per object, frames used for teaching ('train') and for measuring ('test') with a rough box each
(SAM turns the box into the reference mask, so check the printed IoU of your boxes by eye once), the frame range where
the object is present, and frames of the empty scene. Everything is measured on frames the model was NOT taught on.
Negatives are the empty frames AND the frames where another object is present (look-alike test).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from ctrack.detector.learned import LearnedObjectDetector
from ctrack.objectmodel import train_object_model
from ctrack.types import Frame, wrap_angle
from ctrack.vision.onnx_models import DinoFeatures, SamSegmenter

MODELS = Path("models")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def axis(mask: np.ndarray) -> tuple[float, float, float]:
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    return m["m10"] / m["m00"], m["m01"] / m["m00"], 0.5 * math.atan2(2 * m["mu11"], m["mu20"] - m["mu02"])


def contour_mask(contour: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    cv2.fillPoly(out, [np.round(contour).astype(np.int32)], 1)
    return out.astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--video", help="overrides the video named in the labels file")
    ap.add_argument("--objects", help="comma-separated subset")
    ap.add_argument("--view", type=int, default=168)
    ap.add_argument("--refine", type=int, default=336, help="0 = no refinement stage")
    args = ap.parse_args()
    lab = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(args.video or lab["video"])
    if not cap.isOpened():
        raise SystemExit(f"Video nicht lesbar: {args.video or lab['video']}")

    def frame(i: int) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            raise SystemExit(f"Bild {i} fehlt im Video")
        return img

    names = args.objects.split(",") if args.objects else list(lab["objects"])
    sam, dino = SamSegmenter(MODELS), DinoFeatures(MODELS, "dinov2_small.onnx")
    empty_train = [frame(i) for i in lab["empty_train"]]
    empty_test = lab["empty_test"]
    tmp = Path("logs") / "eval_models"
    tmp.mkdir(parents=True, exist_ok=True)

    def reference(i: int, box: list[int]) -> tuple[np.ndarray, np.ndarray]:
        img = frame(i)
        sam.set_image(img)
        x0, y0, x1, y1 = box
        return img, sam.segment([(x0 - 4, y0 - 4), (x1 + 4, y1 + 4)], [2, 3])

    for name in names:
        o = lab["objects"][name]
        t0 = time.time()
        train = [reference(int(i), b) for i, b in o["train"].items()]
        model_path = tmp / f"{name}.npz"
        train_object_model(train, dino, name, empty_scenes=empty_train).save(model_path)

        def fresh() -> LearnedObjectDetector:           # min_score 0: we record the score and apply thresholds afterwards
            d = LearnedObjectDetector(str(model_path), min_score=0.0, global_interval=1, view_size=args.view, refine_size=args.refine)
            d.force_refine = True
            return d

        rows, hit_scores = [], []
        for i, b in o["test"].items():
            img, ref = reference(int(i), b)
            ref_bbox = max(np.ptp(np.nonzero(ref)[1]), np.ptp(np.nonzero(ref)[0]))
            det = fresh().detect(Frame(img, 0.0, int(i)))
            if det is None or det.contour is None:
                hit_scores.append(0.0)
                rows.append((0.0, 99.0, 90.0))
                continue
            hit_scores.append(det.confidence)
            m = contour_mask(det.contour, ref.shape)
            gx, gy, gth = axis(ref)
            rows.append(((m & ref).sum() / max((m | ref).sum(), 1), math.hypot(det.x - gx, det.y - gy) / ref_bbox * 100,
                         abs(math.degrees(wrap_angle(2 * (det.theta - gth)) / 2))))
        a = np.array(rows)
        print(f"\n=== {name}: {len(a)} Testbilder, Training {len(train)} Bilder + {len(empty_train)} leere Szenen ({time.time() - t0:.0f} s)")
        print(f"IoU Median {np.median(a[:, 0]):.2f} (p10 {np.percentile(a[:, 0], 10):.2f}) | Mittelpunktfehler Median "
              f"{np.median(a[:, 1]):.1f} % der Objektgröße (p90 {np.percentile(a[:, 1], 90):.1f}) | Winkelfehler Median "
              f"{np.median(a[:, 2]):.1f}° (p90 {np.percentile(a[:, 2], 90):.1f})")

        def scores(frames: list[int]) -> list[float]:
            out = []
            for i in frames:
                d = fresh().detect(Frame(frame(i), 0.0, i))
                out.append(0.0 if d is None else d.confidence)
            return out

        other = [i for k, v in lab["objects"].items() if k != name for i in range(v["present"][0], v["present"][1], 15)]
        neg = {"leere Szene": scores(empty_test), "anderes Objekt": scores(other)}
        print(f"{'Schwelle':>9} | {'Treffer':>8} | " + " | ".join(f"{k} ({len(v)})" for k, v in neg.items()) + "   (Fehlalarm-Anteil)")
        for t in THRESHOLDS:
            fa = " | ".join(f"{np.mean(np.array(v) >= t) * 100:>{len(k) + 5}.0f} %" for k, v in neg.items())
            print(f"{t:>9.2f} | {np.mean(np.array(hit_scores) >= t) * 100:>6.0f} % | {fa}")


if __name__ == "__main__":
    main()
