# Conveyor Tracking Prototyp (`ctrack`)

Kameragestützter "virtueller Encoder": ein Objekt wird einmal eingelockt, seine planare Pose
(x, y, θ) kontinuierlich verfolgt, per Kalman-Filter latenzkompensiert prädiziert und als
UDP/JSON-Stream ausgegeben. Stufe 1 = Laptop-Webcam + Hand (MediaPipe), Stufe 2 = IDS-Kamera,
ArUco/Objekt, Kalibrierung in mm (siehe `docs/PRD.md`).

**Kernprinzip:** Jeder Pipeline-Block ist ein austauschbares Modul hinter einer festen
Schnittstelle, gewählt per YAML-Registry. Kamera und Detektor wechseln = neue Klasse +
Registry-Eintrag; Tracker, Predictor und Publisher bleiben unverändert.

```
CameraSource → Detector → Tracker (State Machine + Predictor/Kalman) → Transform → Publisher
                                                                      ↘ Visualizer, MetricsLogger
```

## Konventionen

- Python 3.11 (MediaPipe hat keine Wheels für 3.13+), venv in `.venv/`
- Code und Kommentare Englisch, Dokumentation Deutsch
- Zeitstempel immer `time.time()`-Sekunden (float, Epoch); `Frame.t_exposure` ist Pflichtfeld
- Winkel in Radiant, normiert auf (−π, π] über `ctrack.types.wrap_angle`
- Kernlogik (`pipeline.py` und alles darunter) darf **nicht** `cv2.imshow`/UI importieren — der
  Visualizer ist optional (P2-8); die Web-GUI (`ctrack/gui/`) sitzt außerhalb des Kerns (ADR-010)
- GUI: HTTP-Handler dürfen Tracker/Filter nie direkt anfassen — nur Einstellungen setzen oder `Engine.command()`
- Neue Module: Klasse mit `@REGISTRY.register("name")` dekorieren und im `__init__.py` des
  Pakets importieren

## Befehle

```bash
source .venv/bin/activate
python -m ctrack.gui                                   # Web-Oberfläche (Doppelklick: Conveyor Tracking starten.command)
python -m ctrack --config config/default.yaml          # Live-Demo (Webcam + Hand)
python -m ctrack --config config/synthetic.yaml        # Synthetisches Band, keine Kamera nötig
python tools/teach.py --camera 0 --out templates/part.png --mask-auto   # Referenz einlernen
python -m ctrack --config config/shape_match.yaml      # Pose per Template-Matching
python tools/receiver.py --port 5005                   # Referenz-Empfänger
python tools/analyze.py logs/run.csv                   # Auswerte-Plots
pytest                                                 # Tests
```

## Workflow / Fortschritt

| # | Schritt | Ergebnis | Status |
|---|---|---|---|
| 1 | CLAUDE.md | Diese Datei | ✅ Abgeschlossen |
| 2 | Architecture Decision Record | `docs/ARCHITECTURE.md` | ✅ Abgeschlossen |
| 3 | System Design | `docs/SYSTEM-DESIGN.md` | ✅ Abgeschlossen |
| 4 | Projektgerüst (Phase 1a) | Registry, Config, Types, Kamera, Hand-Detektor | ✅ Abgeschlossen |
| 5 | Tracking-Kern (Phase 1b) | Tracker, Kalman, Transform, UDP, Receiver | ✅ Abgeschlossen |
| 6 | Metriken + P1 (Phase 1c) | CSV, Prädiktionsfehler, Aufnahme/Replay, Synthetik, Latenz-Probe, Auswertung | ✅ Abgeschlossen |
| 7 | Teststrategie + Tests | `docs/TESTING.md`, `tests/` | ✅ Abgeschlossen |
| 8 | Visualizer / Demo-Handoff | `docs/DEMO-HANDOFF.md` | ✅ Abgeschlossen |
| 9 | README + Demo-Checkliste | `README.md`, `docs/DEMO-CHECKLIST.md` | ✅ Abgeschlossen |
| 10 | SETUP.md | Manuelle Schritte für den Nutzer | ✅ Abgeschlossen |
| 11 | Stufe 2 (Phase 2a–2d) | IDS, ArUco, Kalibrierung, Robot-Adapter | ⬜ Offen (P2, blockiert durch offene Fragen) |
