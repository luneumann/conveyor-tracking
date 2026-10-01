# ctrack — Conveyor Tracking Prototyp

Kameragestützter "virtueller Encoder": ein Objekt einlocken, seine Pose (x, y, θ) verfolgen,
latenzkompensiert prädizieren und als UDP/JSON-Stream an einen (simulierten) Roboter senden.

**Stack:** Python 3.11 · OpenCV · MediaPipe Tasks (Hand) · NumPy (Kalman) · YAML-Config · UDP/JSON

## Schnellstart

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                              # 56 Tests, ohne Kamera

# Ohne Kamera: synthetisches Band mit Ground Truth
python -m ctrack -c config/synthetic.yaml --auto-lock

# Live mit Webcam + Hand (Modell einmalig laden)
python tools/fetch_model.py
python tools/check_hand.py                          # Modell + MediaPipe ok?
python -m ctrack                                    # Hand zeigen, L drücken

# Zweites Terminal: Stream empfangen
python tools/receiver.py

# Auswertung
python tools/analyze.py logs/run.csv
```

## CLI

```
python -m ctrack [-c CONFIG] [-s section.key=value ...] [--headless] [--auto-lock]
                 [--max-frames N] [--record PATH] [--no-csv]
```

| Option | Zweck |
|---|---|
| `-s predictor.horizon_ms=150` | Einzelwert überschreiben (mehrfach möglich) |
| `--auto-lock` | Erste Detektion automatisch einlocken; nach > 1 s LOST nächstes Objekt |
| `--record recordings/demo` | Rohbilder + Zeitstempel für reproduzierbares Replay (`config/replay.yaml`) |
| `--headless` | Ohne Fenster (z. B. auf IPC) |

## Stream-Format

```json
{"v":1,"seq":1042,"state":"TRACKING","t_exposure":1759329123.412,"t_sent":1759329123.471,
 "pose":{"x":412.3,"y":228.9,"theta":0.412},"velocity":{"vx":85.1,"vy":-2.4,"omega":0.01},
 "unit":"px","frame":"image","confidence":0.94,
 "predicted":{"t":1759329123.571,"x":420.8,"y":228.7,"theta":0.413}}
```

`pose` gilt für `t_exposure`; `null` in SEARCHING/LOST. Details: `docs/SYSTEM-DESIGN.md` §2.

## Projektstruktur

```
config/          default.yaml (Webcam+Hand), synthetic.yaml, replay.yaml
src/ctrack/      types, registry, config, pipeline, tracker, predictor, transform,
                 metrics, visualizer, main; camera/, detector/, publisher/
tools/           receiver.py, analyze.py, latency_probe.py, fetch_model.py
tests/           pytest
docs/            PRD, ARCHITECTURE (ADRs), SYSTEM-DESIGN, TESTING, DEMO-HANDOFF, DEMO-CHECKLIST
```

## Erweitern (Stufe 2)

Neue Kamera / neuer Detektor / Transform / Publisher = Klasse von der jeweiligen Basisklasse ableiten,
mit `@CAMERAS.register("ids_peak")` (bzw. `DETECTORS`, `TRANSFORMS`, `PUBLISHERS`) dekorieren, im
`__init__.py` des Pakets importieren, in der YAML per `type:` wählen. Tracker, Predictor und
Publisher bleiben unverändert.

## Docs

- [PRD](docs/PRD.md) · [Architektur-Entscheidungen](docs/ARCHITECTURE.md) · [System Design](docs/SYSTEM-DESIGN.md)
- [Tests](docs/TESTING.md) · [Demo-Handoff](docs/DEMO-HANDOFF.md) · [Demo-Checkliste](docs/DEMO-CHECKLIST.md)
- [SETUP — was du selbst tun musst](SETUP.md)
