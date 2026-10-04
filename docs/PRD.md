# PRD: Conveyor Tracking Prototyp

**Status:** Entwurf v1 · **Owner:** Luki · **Datum:** 01.10.2026
**Scope:** Stufe 1 (Laptop, Webcam, Hand) baubar · Stufe 2 (IPC, IDS, ArUco/Objekt, Kalibrierung) als P2-Architekturvorgabe

---

## 1. Problem Statement

Bestehende Vision-Software misst Bauteile typischerweise nur statisch: Bild aufnehmen, Pose bestimmen, Ergebnis übergeben. Für Roboteranwendungen an bewegten Bändern fehlt die Fähigkeit, ein Bauteil einmal zu erfassen und seine Bewegung kontinuierlich zu verfolgen, sodass der Roboter laufend eine aktuelle, latenzkompensierte Pose erhält. Heute übernimmt diese Rolle ein Encoder am Band; ohne ihn ist Bandtracking nicht möglich. Am Markt gibt es Lösungen, die genau diese Fähigkeit anbieten (Roboterführung an laufenden Montagelinien). Ohne eigene Lösung fehlt im Robot-Vision-Portfolio ein relevanter Baustein, etwa für die Automobilindustrie.

## 2. Ziele

| # | Ziel | Messbar durch |
|---|---|---|
| G1 | Technische Machbarkeit eines kameragestützten "virtuellen Encoders" belegen | Prädiktionsfehler und Latenz werden live gemessen und protokolliert |
| G2 | Latenzkompensation verstehen und quantifizieren | Latenz Belichtung → Stream p95 < 100 ms (Webcam) |
| G3 | Architektur so schneiden, dass Stufe 2 ohne Umbau möglich ist | Kamera und Detektor per Konfiguration tauschbar, Tracker/Output unverändert |
| G4 | Demonstrierbares Ergebnis für interne Diskussion (Engineering, BD) | Live-Demo mit Overlay aus aktueller und prädizierter Pose |

## 3. Nicht-Ziele (Stufe 1)

| Nicht-Ziel | Begründung |
|---|---|
| Ansteuerung eines echten Roboters | Erst sinnvoll mit kalibrierter Kamera und mm-Koordinaten (Stufe 2+) |
| 6-DoF-Pose | Bandbewegung ist planar; 3-DoF (x, y, θ) deckt den Kernfall ab |
| Mehrere Objekte gleichzeitig | Erhöht Komplexität (Zuordnung) ohne Mehrwert für den Machbarkeitsnachweis |
| Metrische Genauigkeit | Webcam unkalibriert, Ausgabe in Pixeln |
| Integration in ein Produktivsystem | Prototyp läuft standalone; ein Dienst wäre eine eigene Initiative |
| Produktionsreife (Safety, Robustheit, GUI) | Prototyp, keine Kundenauslieferung |

## 4. User Stories

**Entwickler/PM (Luki)**
- Als Entwickler will ich ein Objekt im Kamerabild per Tastendruck einlocken, damit das System ab diesem Moment genau dieses Objekt verfolgt.
- Als Entwickler will ich live sehen, wo das System das Objekt in Δt ms erwartet, damit ich die Qualität der Prädiktion intuitiv beurteilen kann.
- Als Entwickler will ich Latenz und Prädiktionsfehler als Kennzahlen und CSV-Log erhalten, damit ich Parameter gezielt optimieren und Ergebnisse vergleichen kann.
- Als Entwickler will ich, dass das System bei kurzem Objektverlust weiterprädiziert und danach automatisch wieder aufsetzt, damit Verdeckungen nicht sofort zum Abbruch führen.
- Als Entwickler will ich Kamera und Detektor per Konfigurationsdatei wechseln, damit ich ohne Codeänderung zwischen Webcam, Videodatei und später IDS-Kamera umschalten kann.

**Robot-Integrator (simuliert)**
- Als Robot-Integrator will ich einen kontinuierlichen Pose-Stream mit Zeitstempel und Geschwindigkeit empfangen, damit ich die Position zum Ausführungszeitpunkt selbst berechnen kann.
- Als Robot-Integrator will ich den Tracking-Zustand (sucht, verfolgt, verloren) im Stream sehen, damit der Roboter nicht auf ungültige Daten reagiert.

## 5. Architektur

Pipeline, jeder Block ein austauschbares Modul mit fester Schnittstelle:

```
CameraSource → Detector → Tracker (State Machine) → Predictor (Kalman) → Transform → Publisher
                                                                       ↘ Visualizer, MetricsLogger
```

### 5.1 Schnittstellen

| Modul | Schnittstelle | Stufe 1 | Stufe 2 (P2) |
|---|---|---|---|
| `CameraSource` | `read() -> Frame(image, t_exposure, frame_id)` | `WebcamSource`, `VideoFileSource` | `IdsPeakSource` |
| `Detector` | `detect(frame) -> Detection(x, y, theta, confidence) \| None` | `HandDetector` (MediaPipe) | `ArucoDetector`, `ShapeMatchDetector` |
| `Tracker` | `update(detection, t) -> TrackState` | Zustandsmaschine (s. 5.2) | unverändert |
| `Predictor` | `update(pose, t)`, `predict(t_target) -> Pose` | Kalman, konstante Geschwindigkeit | unverändert |
| `Transform` | `to_target(pose) -> pose` | `IdentityTransform` (Pixel) | `ConveyorPlaneTransform` (mm, Bandsystem) |
| `Publisher` | `publish(message)` | `UdpPublisher` | + Robot-Adapter, Encoder-Emulation |

### 5.2 Tracking-Zustände

| Zustand | Bedeutung | Übergang |
|---|---|---|
| `SEARCHING` | Kein Objekt eingelockt | Taste `L` bei gültiger Detektion → `TRACKING` |
| `TRACKING` | Detektion vorhanden, Filter wird aktualisiert | Keine Detektion → `COASTING` |
| `COASTING` | Kurzzeitiger Verlust, reine Prädiktion | Detektion innerhalb `coast_ms` → `TRACKING`; sonst → `LOST` |
| `LOST` | Objekt verloren | Detektion nahe prädizierter Position → `TRACKING`; Taste `R` → `SEARCHING` |

### 5.3 Pose-Definition Hand

Starres Koordinatensystem aus MediaPipe-Landmarks 0 (Handgelenk), 5 (Zeigefinger-Grundgelenk), 17 (kleiner Finger-Grundgelenk):
- Ursprung: Schwerpunkt der drei Punkte
- x-Achse: Handgelenk → Mittelpunkt(5, 17)
- θ: Winkel der x-Achse zur Bildhorizontalen, auf (−π, π] normiert
- Finger werden ignoriert

### 5.4 Stream-Nachricht (UDP, JSON)

```json
{
  "seq": 1042,
  "state": "TRACKING",
  "t_exposure": 1759329123.412,
  "t_sent": 1759329123.471,
  "pose": { "x": 412.3, "y": 228.9, "theta": 0.412 },
  "velocity": { "vx": 85.1, "vy": -2.4, "omega": 0.01 },
  "unit": "px",
  "frame": "image",
  "confidence": 0.94
}
```

Pose bezieht sich auf `t_exposure`. Der Empfänger rechnet mit `velocity` auf seinen Ausführungszeitpunkt hoch; optional liefert der Publisher zusätzlich eine bereits prädizierte Pose für `t_sent + horizon_ms`.

### 5.5 Konfiguration (YAML)

```yaml
camera:   { type: webcam, device: 0, width: 1280, height: 720, exposure_offset_ms: 30 }
detector: { type: hand, min_confidence: 0.6 }
tracker:  { coast_ms: 300, reacquire_radius_px: 80 }
predictor:{ process_noise: 50.0, measurement_noise: 4.0, horizon_ms: 100 }
transform:{ type: identity }
output:   { type: udp, host: 127.0.0.1, port: 5005, rate_hz: 0 }   # 0 = pro Frame
metrics:  { csv: logs/run.csv }
```

## 6. Anforderungen

### P0 – Must-Have

| ID | Anforderung | Akzeptanzkriterien |
|---|---|---|
| P0-1 | Webcam-Quelle mit Zeitstempel | Frames mit `t_exposure` (Erfassungszeit minus konfigurierbarer Offset) und fortlaufender `frame_id`; ≥ 25 fps bei 1280×720 auf dem Laptop |
| P0-2 | Hand-Detektor mit 3-DoF-Pose | Pose gemäß 5.3; bei Hand im Bild Detektionsrate ≥ 90 % der Frames; `None` wenn keine Hand oder Konfidenz unter Schwelle |
| P0-3 | Lock-on und Zustandsmaschine | Zustände und Übergänge gemäß 5.2; Gegeben Hand im Bild, wenn `L` gedrückt, dann Zustand `TRACKING` im nächsten Frame |
| P0-4 | Kalman-Prädiktor | Zustand (x, y, θ, vx, vy, ω); θ-Wrap korrekt (kein Sprung bei ±π); `predict(t)` für beliebige zukünftige Zeitpunkte |
| P0-5 | Coasting und Re-Acquire | Verdeckung < `coast_ms` → Stream läuft mit Zustand `COASTING` weiter; nach Rückkehr der Hand Wiederaufnahme < 1 s ohne erneutes Einlocken |
| P0-6 | UDP-Publisher | Nachricht gemäß 5.4 pro Frame; keine Nachricht mit Zustand `TRACKING` ohne gültige Pose |
| P0-7 | Visualizer | Overlay: aktuelle Pose (Achsenkreuz), prädizierte Pose für `horizon_ms` (gestrichelt), Zustand, fps, Latenz |
| P0-8 | Metriken | Pro Frame CSV-Zeile: t_exposure, t_sent, Zustand, Pose, Geschwindigkeit; Prädiktionsfehler: für jede Prädiktion auf t+Δ wird beim Frame mit nächstliegendem t_exposure der Abstand in px und Grad geloggt; Live-Anzeige p50/p95 |
| P0-9 | Referenz-Empfänger | Separates Skript `receiver.py` empfängt den Stream, zeigt Rate, Zustand und Paketverlust (Lücken in `seq`) |
| P0-10 | Konfiguration und Modultausch | Alle Module über YAML gewählt (Registry); neuer Detektor/Kamera erfordert nur neue Klasse + Registry-Eintrag, keine Änderungen an Tracker, Predictor, Publisher |

### P1 – Nice-to-Have

| ID | Anforderung | Akzeptanzkriterien |
|---|---|---|
| P1-1 | Videodatei-Quelle mit Aufnahme-Modus | Aufnahme speichert Video + Zeitstempel-CSV; Wiedergabe reproduziert identische Zeitstempel → reproduzierbare Vergleiche von Filterparametern |
| P1-2 | Synthetische Bandquelle | Generiert Bildfolge mit Objekt bei konstanter, konfigurierbarer Geschwindigkeit + Rauschen; erlaubt Fehlermessung gegen Ground Truth |
| P1-3 | Latenz-Messhilfe | Modus, der auf dem Bildschirm einen Zeitstempel/Blinkmuster anzeigt, den die Kamera filmt, zur Bestimmung des realen `exposure_offset_ms` |
| P1-4 | Prädizierte Pose im Stream | Optionales Feld `predicted` für `t_sent + horizon_ms` |
| P1-5 | Auswerte-Skript | Liest CSV, erzeugt Plots: Latenz-Histogramm, Prädiktionsfehler über Zeit, Fehler vs. Geschwindigkeit |

### P2 – Stufe 2 (Architekturvorgabe, nicht im Scope v1)

| ID | Anforderung | Designauswirkung heute |
|---|---|---|
| P2-1 | `IdsPeakSource` für IDS-Industriekamera auf IPC, Hardware-Zeitstempel, optional Hardware-Trigger | `Frame.t_exposure` ist von Anfang an Pflichtfeld; Software-Zeitstempel sind nur eine Implementierung |
| P2-2 | `ArucoDetector` (OpenCV) | Detektor-Schnittstelle liefert generische 3-DoF-Pose, nicht handspezifisch |
| P2-3 | Intrinsische Kalibrierung per ChArUco-Board als eigenes CLI-Tool, Ergebnis als YAML | Kalibrierdaten werden geladen, nicht im Code verankert; Entzerrung als optionaler Schritt in `CameraSource` |
| P2-4 | Ebenenkalibrierung Kamera → Band: Homographie aus Marker auf Bandebene, Laufrichtung aus Markerbewegung | `Transform`-Modul existiert bereits in Stufe 1 (Identity); `unit` und `frame` im Stream gesetzt |
| P2-5 | Virtueller Encoder: Position und Geschwindigkeit entlang Bandrichtung als Zählwert/Inkremente | Geschwindigkeit ist Teil des Filterzustands und des Streams |
| P2-6 | `ShapeMatchDetector`: Objekt aus einem Referenzbild einlernen | Lock-on-Logik unabhängig vom Detektor |
| P2-7 | Robot-Adapter (z. B. FANUC, UR RTDE, KUKA) bzw. Encoder-Emulation | Publisher ist Plugin; Nachrichtenformat versioniert |
| P2-8 | Überführung als Dienst in ein Produktivsystem | Kernlogik ohne UI-Abhängigkeit, Visualizer optional abschaltbar |

## 7. Erfolgsmetriken

| Typ | Metrik | Erfolg | Stretch | Messung |
|---|---|---|---|---|
| Leading | Latenz t_exposure → t_sent (p95) | < 100 ms | < 60 ms | CSV, Live-Anzeige |
| Leading | Prädiktionsfehler bei 100 ms Horizont, gleichmäßige Bewegung (p95) | < 2 % der Bildbreite | < 1 % | P0-8, Hand gleichmäßig in eine Richtung bewegen |
| Leading | Re-Acquire nach Verdeckung | < 1 s | < 300 ms | Hand aus Bild und zurück, 10 Wiederholungen |
| Leading | Framerate End-to-End | ≥ 25 fps | ≥ 30 fps | Visualizer |
| Leading | Paketverlust im Stream (lokal) | 0 % | – | `receiver.py` |
| Lagging | Aufwand Wechsel auf IDS + ArUco | Nur neue Module + Konfiguration | – | Code-Diff bei Stufe 2 |
| Lagging | Entscheidung zur Weiterverfolgung als eigenständiger Dienst | Go/No-Go nach Demo | – | Abstimmung mit Engineering/BD |

Auswertung: nach Abschluss Stufe 1 (Demo-Termin), erneut nach Stufe 2 mit mm-Werten.

## 8. Offene Fragen

| Frage | Wer | Blockierend für |
|---|---|---|
| Welche Roboterplattform ist Zielsystem (FANUC, UR, KUKA, ABB) und welches Protokoll (Conveyor-Tracking-Option vs. externe Positionskorrektur)? | Engineering, BD | Stufe 2 (P2-7) |
| Welche Bandgeschwindigkeiten und Positioniergenauigkeiten fordern typische Automotive-Anwendungen? | BD, Vertrieb | Zielwerte Stufe 2 |
| Erwartet die Robotersteuerung Encoder-Signale (Hardware-Emulation) oder Software-Positionsdaten? | Engineering | P2-5, P2-7 |
| Wie wird ein Dienst in ein Produktivsystem eingebunden (Prozess, Schnittstelle, Sprache Python vs. C++)? | Engineering | P2-8 |
| Welche IDS-Kamera (USB3/GigE, Global Shutter, Trigger-Eingang) steht für Stufe 2 zur Verfügung? | Luki | P2-1 |
| Reicht ein Konstant-Geschwindigkeits-Modell, oder braucht es Beschleunigung (Anfahren/Stoppen des Bands)? | Ergebnis Stufe 1 | Nicht blockierend |

## 9. Timeline und Phasen

Keine harte Deadline. Vorgeschlagene Iterationen, je eine abgeschlossene, lauffähige Version:

| Phase | Inhalt | Ergebnis |
|---|---|---|
| 1a | Projektgerüst, Registry, Config, WebcamSource, HandDetector, Visualizer | Hand-Pose live im Bild |
| 1b | Tracker-Zustandsmaschine, Kalman, UDP-Publisher, Receiver | Stream mit Lock-on, Coasting, Prädiktions-Overlay |
| 1c | Metriken, CSV, P1-Punkte nach Bedarf | Messbare Kennzahlen, Demo-fähig |
| 2a | IdsPeakSource, ArucoDetector auf IPC | Tracking mit Industriekamera |
| 2b | ChArUco-Kalibrierung, Ebenenkalibrierung, mm-Ausgabe, virtueller Encoder | Metrischer Stream im Bandsystem |
| 2c | ShapeMatchDetector, Objekt-Teach-in | Tracking eines echten Bauteils |
| 2d | Robot-Adapter, Systemintegration | Abhängig von offenen Fragen |

## 10. Vorgeschlagene Projektstruktur

```
ctrack/
├── config/default.yaml
├── src/ctrack/
│   ├── main.py              # Pipeline-Loop, Tastensteuerung
│   ├── registry.py
│   ├── types.py             # Frame, Detection, Pose, TrackState, Message
│   ├── camera/  webcam.py, video_file.py
│   ├── detector/ hand.py
│   ├── tracker.py
│   ├── predictor.py
│   ├── transform.py
│   ├── publisher/ udp.py
│   ├── visualizer.py
│   └── metrics.py
├── tools/receiver.py
├── logs/
└── requirements.txt         # opencv-python, mediapipe, numpy, pyyaml
```
