# Teststrategie

```bash
source .venv/bin/activate
pytest            # alle Tests, ~8 s, ohne Kamera und ohne MediaPipe-Modell
```

## Ebenen

| Ebene | Was | Wie | Dateien |
|---|---|---|---|
| Unit | Winkel-Wrap, Message-Invariante, Hand-Pose aus Landmarks, Kalman, Zustandsmaschine, Registry/Config, Metrik-Matching, Rate-Limit, Paketverlust-Zählung | Reine Funktionen mit konstruierten Eingaben | `test_types.py`, `test_hand_pose.py`, `test_predictor.py`, `test_tracker.py`, `test_registry_config.py`, `test_metrics.py`, `test_publisher_receiver.py` |
| Integration | UDP-Roundtrip über localhost; neuer Detektor nur per Registry (P0-10) | Echte Sockets, echte Pipeline | `test_publisher_receiver.py`, `test_registry_config.py` |
| End-to-End | Synthetisches Band → Blob-Detektor → Tracker → Kalman → Publisher → Metriken; Verdeckung kurz/lang; Aufnahme + Replay | `SyntheticConveyorSource(realtime=False)` mit Ground Truth | `test_pipeline_e2e.py` |
| Manuell | Webcam + Hand, Visualizer, Latenz-Probe | Demo-Checkliste | `docs/DEMO-CHECKLIST.md` |

## Abgedeckte PRD-Kriterien

| PRD | Test |
|---|---|
| P0-3 Lock → TRACKING im nächsten Frame | `test_lock_goes_tracking_on_next_frame_with_detection` |
| P0-4 θ-Wrap ohne Sprung, `predict(t)` beliebig | `test_theta_crossing_pi_has_no_jump`, `test_predict_future_pose` |
| P0-5 Coasting + Re-Acquire < 1 s | `test_short_occlusion_coasts_then_resumes`, `test_long_occlusion_lost_then_reacquired_within_1s` |
| P0-6 kein TRACKING ohne Pose | `test_tracking_message_requires_pose`, `test_every_tracking_message_has_pose_and_seq_is_contiguous` |
| P0-8 Prädiktionsfehler beim nächstliegenden Frame | `test_prediction_error_matched_to_nearest_frame` u. a. |
| P0-9 Paketverlust aus `seq`-Lücken | `test_receiver_counts_seq_gaps` |
| P0-10 Modultausch ohne Änderung am Kern | `test_new_detector_needs_only_class_and_registry_entry` |
| P1-1 Replay reproduziert Zeitstempel | `test_record_and_replay_reproduce_timestamps` |
| P2-6 Pose aus Referenzbild (θ über ±π, 180°-Mehrdeutigkeit, Verlust/Rückkehr, Template-Parität) | `tests/test_shape_match.py` |
| Erfolgsmetrik Prädiktionsfehler p95 < 2 % | `test_prediction_error_meets_prd_goal` (synthetisch) |

## Coverage-Ziele

- Kernlogik (`types`, `predictor`, `tracker`, `metrics`, `config`, `registry`, `pipeline`): jeder Zweig
  der Zustandsmaschine und jede PRD-Akzeptanzbedingung, die ohne Hardware prüfbar ist.
- Keine prozentuale Zeilen-Coverage als Ziel — die Hardware-Adapter würden die Zahl verzerren.

## Bewusst nicht automatisiert

| Teil | Warum | Stattdessen |
|---|---|---|
| `WebcamSource` | Braucht Kamera + macOS-Berechtigung | Demo-Checkliste, Schritt "Kamera" |
| `HandDetector` (MediaPipe-Inferenz) | Braucht Modelldatei und echte Handbilder; Detektionsrate ≥ 90 % ist eine Feldmessung | Pose-Mathematik ist unit-getestet; Rate über `logs/run.csv` (Anteil Zeilen mit `det_x`) |
| `Visualizer` | UI, visuelle Prüfung | Demo |
| `shape_match` mit echtem Bauteil | Braucht Kamera + Teil; Tests nutzen ein synthetisches Teil. Maßstab-/Verdeckungsgrenzen (ADR-009) sind eine Einmalmessung, kein Test | Mit echtem Teil einlernen und `analyze.py` auswerten |
| `tools/latency_probe.py` | Braucht Kamera, die den Bildschirm sieht | Manuell, Ergebnis in Config übernehmen |
| Latenz p95 < 100 ms, fps ≥ 25 mit Webcam | Hardwareabhängig | `tools/analyze.py logs/run.csv` nach Demo-Lauf |

## Filter-Tuning reproduzierbar

1. Einmal aufnehmen: `python -m ctrack --record recordings/demo`
2. Mit verschiedenen Parametern abspielen:
   `python -m ctrack -c config/replay.yaml --headless --auto-lock -s predictor.process_noise=200 -s metrics.csv=logs/q200.csv`
3. Vergleichen: `python tools/analyze.py logs/q200.csv`
