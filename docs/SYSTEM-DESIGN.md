# System Design

## 1. Komponenten

```
                 ┌──────────── config/*.yaml ────────────┐
                 ▼                                        ▼
CameraSource ─Frame─▶ Detector ─Detection?─▶ Tracker ──▶ Transform ──▶ Publisher (UDP)
 (webcam,             (hand, blob)           │ owns          (identity)
  video_file,                                ▼
  synthetic)                              Predictor (Kalman)
                                             │
                       Pipeline.step() ──StepResult──▶ MetricsLogger (CSV, p50/p95)
                                             └──────▶ Visualizer (optional, cv2 window)
```

| Modul | Datei | Rolle |
|---|---|---|
| Typen | `src/ctrack/types.py` | `Frame`, `Detection`, `Pose`, `Velocity`, `TrackState`, `Message`, `wrap_angle` |
| Registry | `src/ctrack/registry.py` | Name → Klasse, `create(cfg)` |
| Config | `src/ctrack/config.py` | YAML laden, mit Defaults mergen |
| Kamera | `src/ctrack/camera/` | `WebcamSource`, `VideoFileSource`, `SyntheticConveyorSource`, `Recorder` |
| Detektor | `src/ctrack/detector/` | `HandDetector` (MediaPipe), `ShapeMatchDetector` (Template, ADR-009), `MarkerBlobDetector` (Synthetik) |
| Tracker | `src/ctrack/tracker.py` | Zustandsmaschine, Lock/Reset, Re-Acquire-Gate |
| Predictor | `src/ctrack/predictor.py` | Kalman (x, y, θ, vx, vy, ω) |
| Transform | `src/ctrack/transform.py` | `IdentityTransform` (px, image) |
| Publisher | `src/ctrack/publisher/` | `UdpPublisher`, `NullPublisher` |
| Pipeline | `src/ctrack/pipeline.py` | Ein Frame → `StepResult`; UI-frei (P2-8) |
| Metriken | `src/ctrack/metrics.py` | CSV, Prädiktionsfehler-Matching, Live-Perzentile |
| Visualizer | `src/ctrack/visualizer.py` | Overlay + Tastatur |
| Einstieg | `src/ctrack/main.py` | CLI, Loop, Tasten `L`/`R`/`Q` |
| Tools | `tools/` | `receiver.py`, `analyze.py`, `latency_probe.py`, `fetch_model.py`, `check_hand.py`, `teach.py` |

## 2. Datenmodell

```python
Frame(image: ndarray[H,W,3] BGR, t_exposure: float, frame_id: int,
      ground_truth: Pose | None = None)            # nur Synthetik
Pose(x: float, y: float, theta: float)              # theta ∈ (−π, π]
Velocity(vx: float, vy: float, omega: float)        # px/s, rad/s
Detection(x, y, theta, confidence, keypoints: ndarray | None)  # keypoints nur fürs Overlay
TrackState = SEARCHING | TRACKING | COASTING | LOST
```

`Message` entspricht PRD 5.4 plus `v` (Formatversion) und optional `predicted`:

| Feld | Typ | Wann gesetzt |
|---|---|---|
| `v` | int | immer, aktuell `1` |
| `seq` | int | immer, +1 pro Nachricht, auch bei SEARCHING |
| `state` | str | immer |
| `t_exposure`, `t_sent` | float s | immer |
| `pose` | `{x,y,theta}` \| null | TRACKING (Filterschätzung), COASTING (Prädiktion); sonst null |
| `velocity` | `{vx,vy,omega}` \| null | wie `pose` |
| `predicted` | `{t, x, y, theta}` \| null | wenn `output.include_predicted` und `pose` gesetzt |
| `unit`, `frame` | str | vom Transform (`px`, `image`) |
| `confidence` | float | Detektionskonfidenz, 0.0 ohne Detektion |

Invariante (P0-6): `state == TRACKING ⇒ pose != null` — erzwungen im Konstruktor von `Message`.

## 3. Zustandsmaschine (Tracker)

| Von | Ereignis | Nach | Filter |
|---|---|---|---|
| SEARCHING | Lock angefordert (`L`) + Detektion | TRACKING | init mit Detektion, v = 0 |
| SEARCHING | sonst | SEARCHING | – |
| TRACKING | Detektion | TRACKING | predict + update |
| TRACKING | keine Detektion | COASTING | nur predict (on demand) |
| COASTING | Detektion, t − t_last_seen ≤ `coast_ms` | TRACKING | predict + update |
| COASTING | t − t_last_seen > `coast_ms` | LOST | – |
| LOST | Detektion mit Abstand ≤ Gate(t) zur Referenzposition | TRACKING | **re-init** mit Detektion |
| beliebig | Reset (`R`) | SEARCHING | verworfen |

Ein Lock-Request bleibt bestehen, bis eine Detektion kommt ("im nächsten Frame mit Hand").
**Referenzposition in LOST** = prädizierte Position, Extrapolation aber gedeckelt auf
`coast_ms` nach dem letzten Sichtkontakt. Sonst würde die Prädiktion bei längerem Verlust aus dem
Bild laufen und ein Re-Acquire unmöglich machen. Re-Init statt Update, weil die
Geschwindigkeitsschätzung nach einem Verlust veraltet ist.

**Wachsendes Gate:** `Gate(t) = reacquire_radius_px + reacquire_growth_px_s · (t − t_lost)`.
Direkt nach dem Verlust ist das Fenster eng (80 px), danach wächst es (600 px/s). Grund: Im zweiten
Live-Lauf kam die Hand nach 14 s ≥ 272 px entfernt vom Verlustort zurück; mit festem 80-px-Fenster war
keine einzige von 346 Detektionen in LOST akzeptabel. Auf den geloggten Detektionen dieses Laufs
wird der Verlust jetzt nach 0,53 s wieder aufgenommen. Bei später mehreren Objekten muss die
Zuordnung (Identität statt reiner Distanz) ergänzt werden — dann ist das Wachstum zu überdenken.

## 4. Konfiguration

Siehe `config/default.yaml` (PRD 5.5) und `config/synthetic.yaml`. Ergänzungen gegenüber PRD:

| Schlüssel | Default | Zweck |
|---|---|---|
| `detector.model_path` | `models/hand_landmarker.task` | MediaPipe-Modell (ADR-003) |
| `tracker.reacquire_growth_px_s` | 600 | Wachstum des Re-Acquire-Gates in LOST |
| `detector.template` / `mask` | – | Referenzbild bzw. Maske für `shape_match` (`tools/teach.py`) |
| `detector.min_score` | 0.85 | Mindest-Korrelation (`shape_match`) |
| `detector.downscale` | 4 | Verkleinerung der globalen Suche (8 = ~2× schneller, kleinere Teile gehen verloren) |
| `detector.angle_step_deg` / `fine_step_deg` | 6 / 2 | Winkelraster Grob-/Feinstufe |
| `detector.local_margin_px` / `local_angle_range_deg` | 100 / 18 | Suchfenster um die letzte Pose |
| `detector.global_interval` | 3 | Globale Suche bei fehlendem Teil nur jeden n-ten Frame |
| `predictor.process_noise_theta` | 2.0 | ADR-005 |
| `predictor.measurement_noise_theta` | 0.05 | ADR-005 |
| `output.include_predicted` | true | P1-4 |
| `metrics.match_tolerance_ms` | 25 | max. Abstand Prädiktionsziel ↔ Frame beim Fehler-Matching |
| `metrics.window` | 300 | Fenster (Frames) für Live-p50/p95 |
| `visualizer.enabled` | true | P2-8: Kern läuft headless |

## 5. Metriken (CSV)

Eine Zeile pro Frame, Spalten:

```
frame_id, t_exposure, t_read, t_sent, latency_ms, state,
x, y, theta, vx, vy, omega,                 # publizierte Schätzung (leer ohne Pose)
det_x, det_y, det_theta, confidence,        # Rohdetektion (leer ohne Detektion)
pred_t, pred_x, pred_y, pred_theta,         # Prädiktion dieses Frames für t_exposure + horizon
pred_err_px, pred_err_deg,                  # Fehler dieser Prädiktion (s. u.)
gt_err_px, gt_err_deg                       # Schätzung vs. Ground Truth (nur Synthetik)
```

**Prädiktionsfehler (P0-8):** Jede Prädiktion auf `t + Δ` wartet, bis ein Frame mit
`t_exposure ≥ t + Δ` vorliegt. Dann wird unter allen Frames mit Detektion der mit dem
nächstliegenden `t_exposure` gewählt (nur wenn Abstand ≤ `match_tolerance_ms`) und der Abstand
Prädiktion ↔ Rohdetektion in px und Grad geschrieben. Zeilen werden dafür um ≈ `horizon_ms`
verzögert geschrieben, damit jede Zeile ihren eigenen Fehler enthält.

`latency_ms` = `t_sent − t_exposure` bei Live-Quellen, `t_sent − t_read` bei Replay (ADR-007).

## 6. Datenfluss — "Hand einlocken und streamen"

1. `WebcamSource.read()`: `grab()` → `t = time.time() − offset` → `retrieve()` → `Frame`.
2. `HandDetector.detect(frame)`: BGR→RGB, `detect_for_video(img, ts_ms)`; Landmarks 0/5/17 →
   `hand_pose_from_landmarks` → `Detection` oder `None` (keine Hand / Konfidenz < Schwelle).
3. Nutzer drückt `L` → `main` ruft `tracker.request_lock()`.
4. `Tracker.update(det, t_exposure)`: SEARCHING + Lock + Detektion → Predictor init → TRACKING.
5. `Pipeline.step` holt Schätzung bei `t_exposure` und Prädiktion bei `t_exposure + horizon`,
   wendet `Transform.to_target` an.
6. `t_sent = time.time()`, `Message` bauen, `UdpPublisher.publish` → `127.0.0.1:5005`.
7. `MetricsLogger.log(step)` puffert die Zeile, löst ältere Prädiktionen auf, schreibt CSV.
8. `Visualizer.show(step, stats)` zeichnet Overlay, liefert gedrückte Taste zurück.
9. `tools/receiver.py` empfängt, zeigt Rate, Zustand, Paketverlust, Latenz.
