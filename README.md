# ctrack — Conveyor Tracking Prototyp

Kameragestützter "virtueller Encoder": ein Objekt einlocken, seine Pose (x, y, θ) verfolgen,
latenzkompensiert prädizieren und als UDP/JSON-Stream an einen (simulierten) Roboter senden.

**Stack:** Python 3.11 · OpenCV · MediaPipe Tasks (Hand) · ONNX Runtime (MobileSAM, DINOv2) · NumPy (Kalman) · lokale Web-GUI (stdlib) · YAML-Config · UDP/JSON

## Schnellstart (ohne Terminal)

**Doppelklick auf `Conveyor Tracking starten.command`.** Beim ersten Mal richtet es die Umgebung ein, danach
öffnet sich im Browser die Oberfläche (`http://127.0.0.1:8765`):

1. **Quelle** wählen: *Kamera*, *Demo-Band* (simuliert, ohne Kamera) oder *Aufnahme* → **Starten**.
2. **Erkennung** wählen:
   - **Gelernt** (für Objekte, die du in die Kamera hältst): **Neues Objekt anlernen** → ~5 Fotos aufnehmen
     (anderer Abstand, leicht gedreht; auch „In 3 s aufnehmen“), jedes Mal **einmal aufs Objekt klicken** →
     **Trainieren** (Sekunden). Beim ersten Mal einmalig **Modelle laden** (ca. 133 MB).
   - **Referenzbild** (flache Teile mit gleichbleibendem Abstand): **Aus Livebild einlernen**, Rechteck ziehen.
   - **Hand**.
3. **Tracking starten** (oder „Automatisch einlocken“). Kennzahlen (Bildrate, Latenz, Vorhersagefehler)
   erscheinen live mit Zielwerten.
4. Optional: **Pose per UDP senden**, **Video aufzeichnen**, Feineinstellungen.

Alles Weitere steht in `SETUP.md`. Für Entwickler gibt es zusätzlich die Kommandozeile:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                              # 116 Tests, ohne Kamera
python -m ctrack.gui                                # Web-Oberfläche (wie der Doppelklick)
python -m ctrack -c config/synthetic.yaml --auto-lock   # OpenCV-Fenster statt Web-GUI
python tools/receiver.py                            # Stream-Empfänger (zweites Terminal)
python tools/analyze.py logs/run.csv                # Auswertung mit Plots
```

## Kommandozeile (CLI)

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
config/          default.yaml (Webcam+Hand), shape_match.yaml, synthetic*.yaml, replay.yaml
templates/       eingelernte Referenzbilder
src/ctrack/      types, registry, config, pipeline, tracker, predictor, transform,
                 metrics, visualizer, teach, models, main; camera/, detector/, publisher/, gui/
tools/           receiver.py, analyze.py, latency_probe.py, fetch_model.py, check_hand.py, teach.py
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
- [Optimierung](docs/OPTIMIERUNG.md) · [Tests](docs/TESTING.md) · [Demo-Handoff](docs/DEMO-HANDOFF.md) · [Demo-Checkliste](docs/DEMO-CHECKLIST.md)
- [SETUP — was du selbst tun musst](SETUP.md)
