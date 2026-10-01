# Setup — Was du selbst tun musst

Der Code ist fertig und getestet (56 Tests grün, synthetische Pipeline läuft). Die folgenden Schritte
brauchen dich, weil sie Kamera, Downloads oder Entscheidungen betreffen.

## Vor dem ersten Start

### 1. Python-Umgebung

Bereits erledigt in `.venv/` (Python 3.11). Auf einem anderen Rechner:

```bash
cd ~/Projects/Tracking
python3.11 -m venv .venv          # 3.11 oder 3.12 — MediaPipe hat keine Wheels für 3.13+
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

### 2. MediaPipe-Handmodell laden (~7,5 MB, einmalig)

```bash
python tools/fetch_model.py
```

Lädt `hand_landmarker.task` von Googles MediaPipe-Modellserver nach `models/` (nicht im Git).
Ohne das Modell läuft nur die Synthetik (`config/synthetic.yaml`). Auf diesem Rechner bereits geladen.

Prüfen, ob Modell und MediaPipe zusammenpassen (ohne Kamera):

```bash
python tools/check_hand.py        # erwartet: "DETECTOR OK: … ms/frame"
```

### 3. Kamera-Berechtigung (macOS)

Beim ersten `python -m ctrack` fragt macOS nach Kamerazugriff für die Terminal-App (Terminal,
iTerm, VS Code …). Falls versehentlich abgelehnt:
**Systemeinstellungen → Datenschutz & Sicherheit → Kamera →** Terminal-App aktivieren, Terminal neu starten.

### 4. Belichtungs-Offset messen (P1-3, empfohlen vor Messungen)

Der Default `exposure_offset_ms: 30` ist geschätzt. Messen:

```bash
python tools/latency_probe.py
```

Die Kamera muss das Fenster "probe" sehen — eine Laptop-Webcam sieht den eigenen Bildschirm nicht:
Spiegel davor halten oder eine externe Webcam auf den Laptop-Bildschirm richten. Den ausgegebenen
Wert in `config/default.yaml` → `camera.exposure_offset_ms` eintragen.

### 5. Filter tunen (nach dem ersten Live-Lauf)

`process_noise: 50` aus dem PRD ist für eine Hand eher träge (Konstant-Geschwindigkeits-Modell,
Einheit px²/s³, siehe ADR-005). Wenn das gestrichelte Kreuz bei Richtungswechseln deutlich
hinterherläuft: aufnehmen und per Replay vergleichen:

```bash
python -m ctrack --record recordings/demo                      # Hand bewegen, Q
for q in 50 500 5000; do
  python -m ctrack -c config/replay.yaml --headless --auto-lock \
    -s camera.path=recordings/demo.mp4 -s predictor.process_noise=$q -s metrics.csv=logs/q$q.csv
  python tools/analyze.py logs/q$q.csv | grep "pred err"
done
```

### 6. Git-Remote (optional)

Das Repo ist lokal initialisiert (Branch `main`), hat aber **keinen Remote** — ich habe kein
GitHub-Repo ohne Rückfrage angelegt. Falls gewünscht:

```bash
gh repo create conveyor-tracking --private --source=. --remote=origin --push
```

## Nach jeder Änderung

- `pytest` laufen lassen
- Bei neuen Modulen: Registry-Eintrag + Import im `__init__.py` des Pakets
- Bei Änderungen am Stream-Format: `MESSAGE_VERSION` in `src/ctrack/types.py` erhöhen (P2-7)

## Offene Fragen und bekannte Probleme

**Aus dem PRD (unverändert offen):**

| Frage | Wer | Blockiert |
|---|---|---|
| Zielroboter (FANUC/UR/KUKA/ABB) und Protokoll | Engineering, BD | P2-7 |
| Typische Bandgeschwindigkeiten und Genauigkeiten Automotive | BD, Vertrieb | Zielwerte Stufe 2 |
| Encoder-Hardware-Emulation vs. Software-Positionsdaten | Engineering | P2-5, P2-7 |
| MSS-Anbindung (Prozess, Schnittstelle, Python vs. C++) | Engineering | P2-8 |
| Welche IDS-Kamera für Stufe 2 | Luki | P2-1 |
| Konstant-Geschwindigkeit ausreichend? | Ergebnis Stufe 1 | – |

**Beim Aufbau entstanden / bekannt:**

- **Live-Hand-Pfad nur teilweise getestet.** Modell lädt, `HandDetector` läuft auf Leerbildern
  (11,6 ms/Frame, keine Fehldetektion). Webcam und echte Hand habe ich nicht gesehen — Mathematik der
  Hand-Pose ist unit-getestet, die Detektionsrate ≥ 90 % (P0-2) ist noch ungemessen. Erster Live-Start
  = erster echter Test.
- **MediaPipe auf 0.10.21 gepinnt.** 1.0.1 stürzt auf macOS/Apple Silicon beim Start ab (Metal-Service,
  auch mit CPU-Delegate; ADR-003). 1.x hat außerdem die alte `mp.solutions.hands`-API entfernt → Tasks-API
  + separate Modelldatei.
- **`confidence`** der Hand ist der Handedness-Score von MediaPipe — ein Proxy, keine echte
  Detektionsgüte.
- **Re-Acquire-Referenz** in LOST: Extrapolation auf `coast_ms` gedeckelt (SYSTEM-DESIGN §3) —
  Interpretation von "nahe prädizierter Position", bitte bestätigen.
- **Stream-Erweiterung:** Feld `"v": 1` zusätzlich zu PRD 5.4 (Versionierung für P2-7).
- **Synthetik am Bildrand:** Wird das Objekt vom Bildrand abgeschnitten, verschiebt sich der
  Schwerpunkt (Ground-Truth-Fehler bis ~20 px nur dort; innen ~0,05 px). Betrifft nur die Synthetik.
- **Latenz bei Replay** misst nur die Verarbeitungszeit (`t_sent − t_read`), weil `t_exposure`
  aus der Aufnahme stammt (ADR-007).
- **Single-Thread:** Falls Webcam + MediaPipe < 25 fps, Kamera-Grab in eigenen Thread auslagern.

## Kosten-Übersicht

Keine laufenden Kosten. Alle Abhängigkeiten sind Open Source (OpenCV: Apache 2.0, MediaPipe:
Apache 2.0, NumPy: BSD, PyYAML: MIT, Matplotlib: PSF-basiert). Kein Cloud-Dienst, kein Account.
Stufe 2: Kosten der IDS-Kamera; IDS peak SDK ist kostenlos.
