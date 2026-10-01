# Architecture Decision Records

Stand: 01.10.2026 · Bezug: `docs/PRD.md` (Entwurf v1)

---

## ADR-001 — Sprache und Laufzeit: Python 3.11

**Entscheidung:** Python 3.11 in einem lokalen venv (`.venv/`).

**Begründung:** PRD 10 schlägt Python vor (OpenCV, MediaPipe, NumPy). Schnellste Iteration für
einen Machbarkeitsnachweis; OpenCV und MediaPipe sind nativ beschleunigt, die Python-Schicht ist
nur Orchestrierung (≈ 1 Detektion + 6×6-Kalman pro Frame). 3.11 statt der System-Version 3.14,
weil MediaPipe nur Wheels bis 3.12 anbietet.

**Alternativen:** C++ (näher an MSS, aber 3–5× Entwicklungsaufwand für den Prototyp);
Python 3.12 (wäre ebenfalls möglich, auf dem Rechner nicht installiert).

**Konsequenzen:** GIL — Kamera, Detektion und Ausgabe laufen sequentiell in einem Thread.
Für ≥ 25 fps ausreichend; falls nicht, Kamera-Grab in eigenen Thread auslagern. Ob MSS-Dienst
in Python oder C++ wird, ist offene Frage (PRD 8) — die Kernlogik ist bewusst klein und
portierbar gehalten.

---

## ADR-002 — Modulschnitt über Registry + YAML

**Entscheidung:** Jede Stufe (`camera`, `detector`, `transform`, `publisher`) ist eine abstrakte
Basisklasse. Implementierungen registrieren sich per Dekorator unter einem Namen;
`config/*.yaml` wählt per `type:` aus. Restliche Schlüssel des YAML-Blocks werden als
Keyword-Argumente an den Konstruktor übergeben.

**Begründung:** Erfüllt P0-10 und G3 direkt. Stufe 2 (IDS, ArUco, Ebenen-Transform,
Robot-Adapter) ist dann nur neue Klasse + Eintrag.

**Alternativen:** Entry-Points/Plugins via `importlib.metadata` (Overkill für ein Repo);
Dependency-Injection-Framework (zusätzliche Abhängigkeit ohne Mehrwert).

**Konsequenzen:** Tippfehler im YAML fallen erst zur Laufzeit auf → `Registry.create` gibt
eine klare Fehlermeldung mit allen verfügbaren Namen aus.

---

## ADR-003 — Hand-Detektor über MediaPipe Tasks `HandLandmarker`

**Entscheidung:** `HandDetector` nutzt die MediaPipe-Tasks-API im `VIDEO`-Modus
(`detect_for_video`), Modell `hand_landmarker.task` aus `models/`.

**Begründung:** Die Legacy-API `mp.solutions.hands` ist in MediaPipe 1.x entfernt. Der
`VIDEO`-Modus nutzt internes Tracking zwischen Frames (schneller als `IMAGE`) und ist
synchron (deterministisch, keine Callback-Reihenfolge).

**Alternativen:** `LIVE_STREAM` (asynchron, Ergebnis kommt in einem späteren Loop-Durchlauf
→ verfälscht die Latenzmessung); eigene Hand-Segmentierung (unzuverlässig).

**Konsequenzen:** Das Modell (~7,5 MB) wird nicht mit pip installiert, sondern einmalig per
`tools/fetch_model.py` geladen (siehe `SETUP.md`). Die Pose-Berechnung aus den Landmarks ist
eine reine Funktion (`hand_pose_from_landmarks`) und ohne MediaPipe testbar.

---

## ADR-004 — Tracker besitzt den Predictor

**Entscheidung:** `Tracker` (Zustandsmaschine aus PRD 5.2) bekommt den `Predictor` injiziert
und entscheidet, wann er ihn initialisiert, aktualisiert oder nur extrapoliert.

**Begründung:** Die Übergänge hängen von der Prädiktion ab (Re-Acquire "nahe prädizierter
Position") und die Prädiktion hängt vom Zustand ab (COASTING = kein Measurement-Update).
Eine Klasse, die beides koordiniert, verhindert inkonsistente Zustände (z. B. Filter-Update
während `LOST`).

**Alternativen:** Tracker und Predictor nebeneinander in der Pipeline (Pipeline müsste die
Kopplung kennen — Logik würde nach `main.py` wandern).

**Konsequenzen:** `Predictor` bleibt eine eigene Klasse mit eigener Schnittstelle (PRD 5.1)
und ist separat testbar und austauschbar (z. B. Konstant-Beschleunigung, offene Frage PRD 8).

---

## ADR-005 — Kalman-Filter: Konstant-Geschwindigkeit, NumPy, θ-Wrap

**Entscheidung:** Linearer Kalman-Filter mit Zustand `[x, y, θ, vx, vy, ω]`,
Prozessmodell "Continuous White Noise Acceleration" je Achse, selbst implementiert in NumPy.
Winkel-Innovation und Zustand werden nach jedem Schritt auf (−π, π] normiert.

Parameter (YAML `predictor`):

| Schlüssel | Bedeutung | Einheit |
|---|---|---|
| `process_noise` | Spektraldichte q der Beschleunigung für x/y | px²/s³ |
| `process_noise_theta` | dito für θ | rad²/s³ |
| `measurement_noise` | Standardabweichung der Messung x/y | px |
| `measurement_noise_theta` | Standardabweichung der Messung θ | rad |
| `horizon_ms` | Prädiktionshorizont für Overlay, Metriken, `predicted` | ms |

**Begründung:** P0-4. Bandbewegung ist näherungsweise konstant; 6×6-Matrizen brauchen keine
Bibliothek. Eigene Implementierung macht θ-Wrap explizit und testbar.

**Alternativen:** `cv2.KalmanFilter` (kein Winkel-Wrap, kein variables dt ohne manuelles
Setzen der Matrizen); `filterpy` (unmaintained); UKF/EKF (unnötig, Modell ist linear).

**Konsequenzen:** Bei Anfahren/Stoppen des Bands entsteht Lag — ein Konstant-Beschleunigungs-
Predictor wäre eine zweite Klasse hinter derselben Schnittstelle.

---

## ADR-006 — Ausgabe: UDP + JSON, versioniert

**Entscheidung:** Ein Datagramm pro Frame (oder gedrosselt per `rate_hz`), JSON gemäß PRD 5.4,
plus Feld `"v": 1` (Format-Version, gefordert durch P2-7) und optional `"predicted"` (P1-4).

**Begründung:** Verbindungslos, minimal, kein Head-of-Line-Blocking — veraltete Posen sind für
einen Roboter wertlos, Retransmits daher unerwünscht. JSON ist mit jedem Werkzeug lesbar.

**Alternativen:** TCP (Blocking bei langsamem Empfänger), ZeroMQ/ROS 2 (zusätzliche Infrastruktur
für einen Standalone-Prototyp), Binärformat (erst bei Robot-Adapter sinnvoll).

**Konsequenzen:** Paketverlust ist möglich und wird über Lücken in `seq` im Receiver sichtbar.
`pose` ist `null`, wenn kein gültiger Wert vorliegt (`SEARCHING`, `LOST`); `TRACKING` ohne Pose
ist per Konstruktion ausgeschlossen (P0-6).

---

## ADR-007 — Zeitbasis: Software-Zeitstempel mit Belichtungs-Offset

**Entscheidung:** `t_exposure = time.time()` direkt nach `VideoCapture.grab()` minus
`exposure_offset_ms`. Der Offset wird mit `tools/latency_probe.py` gemessen (P1-3).

**Begründung:** Webcams liefern keine Hardware-Zeitstempel. `grab()` vor `retrieve()` trennt
Erfassung von Dekodierung. Der Offset fasst Belichtung + Sensor-Auslesen + USB/Treiber zusammen.

**Konsequenzen:** Die gemessene Latenz (`t_sent − t_exposure`) ist nur so gut wie der Offset.
In Stufe 2 liefert `IdsPeakSource` Hardware-Zeitstempel hinter derselben Schnittstelle (P2-1).
Für nicht-live Quellen (Videodatei-Replay) loggen die Metriken die Verarbeitungslatenz
(`t_sent − t_read`), weil `t_exposure` dort aus der Aufnahme stammt.

---

## ADR-008 — Synthetische Quelle mit eigenem Detektor

**Entscheidung:** `SyntheticConveyorSource` (P1-2) rendert ein Objekt (heller Körper + roter
Richtungspunkt) mit konstanter Geschwindigkeit und Rauschen und hängt die Ground-Truth-Pose an
den Frame. `MarkerBlobDetector` erkennt es klassisch per Farbschwelle.

**Begründung:** Vollständige Pipeline (inkl. Detektion) ohne Kamera, ohne MediaPipe-Modell und
reproduzierbar — Grundlage für End-to-End-Tests und Filter-Tuning gegen Ground Truth.

**Alternativen:** Ground Truth direkt als "Detektion" durchreichen (testet den Detektor-Pfad
nicht); ArUco-Marker rendern (wäre Stufe 2 vorweggenommen — bleibt P2-2).

**Konsequenzen:** Der Blob-Detektor ist nur für die Synthetik gedacht, nicht für echte Szenen.
