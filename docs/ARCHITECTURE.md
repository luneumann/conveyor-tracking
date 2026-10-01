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

**Begründung:** Die Legacy-API `mp.solutions.hands` ist in MediaPipe 1.x entfernt (die Tasks-API gibt es
auch in 0.10.x). Der
`VIDEO`-Modus nutzt internes Tracking zwischen Frames (schneller als `IMAGE`) und ist
synchron (deterministisch, keine Callback-Reihenfolge).

**Alternativen:** `LIVE_STREAM` (asynchron, Ergebnis kommt in einem späteren Loop-Durchlauf
→ verfälscht die Latenzmessung); eigene Hand-Segmentierung (unzuverlässig).

**Version gepinnt auf `mediapipe==0.10.21`:** MediaPipe 1.0.1 bricht auf macOS (Apple Silicon) beim
Anlegen des Landmarkers mit `Check failed: service_ Service is unavailable` ab
(`DrishtiMetalHelper` in `TensorsToDetectionsCalculator::Open`), auch mit `delegate=CPU`.
Reproduziert im eigenen Terminal (nicht sandbox-bedingt); 0.10.21 läuft (11,6 ms/Frame auf M3,
CPU). Vor einem Upgrade `python tools/check_hand.py` ausführen.

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

---

## ADR-009 — Pose aus Referenzbild: maskiertes Template-Matching über alle Drehwinkel

**Entscheidung:** `ShapeMatchDetector` (`detector/shape_match.py`) lernt ein Referenzbild ein
(`tools/teach.py`, optional mit Maske). Orientierung im Referenzbild = θ 0, Template-Mitte
((w−1)/2, (h−1)/2, Pixelzentrum-Koordinaten) = Pose-Ursprung. Gesucht wird per
`cv2.matchTemplate(TM_CCOEFF_NORMED, mask)` über alle Winkel in drei Stufen: Grobsuche im verkleinerten
Bild, Feinstufe in voller Auflösung, danach Parabel-Fit über Position und Winkel (Sub-Pixel,
Sub-Grad). Im laufenden Betrieb wird nur lokal um die letzte Pose gesucht (±`local_margin_px`,
±`local_angle_range_deg`); die globale Suche läuft beim Start und nach Verlust, bei fehlendem Teil
nur jeden `global_interval`-ten Frame.

**Begründung:** Projektentscheidung für Stufe 2c: ein echtes Bauteil ohne Marker, eingelernt aus einem
Referenzmerkmal. Die Normierung (CCOEFF_NORMED) macht das Verfahren unempfindlich gegen Helligkeit und
Kontrast. Alle 360° im Suchraum lösen die 180°-Mehrdeutigkeit, solange das Teil asymmetrische Merkmale hat.

**Alternativen:** Halcon-artiges kantenbasiertes / CAD-basiertes Matching (robuster gegen Verdeckung und
Maßstab, aber kommerziell bzw. eigener Aufwand — später als weiterer Detektor hinter derselben
Schnittstelle möglich); ArUco (Marker nötig); Momente/Hauptachse (180°-mehrdeutig); gelerntes Netz
(Trainingsdaten).

**Konsequenzen / gemessene Grenzen** (synthetisches Teil, `tests/test_shape_match.py` + Einmalmessung):

| Bedingung | Ergebnis |
|---|---|
| sauber / Helligkeit ×0,5 / +60 / Kontrast ×0,6 | 0,2 px, 0,08° |
| Unschärfe σ 8 px, Rauschen σ 50 | Score ≥ 0,90, Fehler < 0,3 px |
| Maßstab ±10 % | Score ≈ 0,8, aber **4,8 px Fehler** (stille Fehlmessung) |
| Maßstab ±20 % | nicht gefunden |
| ≥ 30 % verdeckt | nicht gefunden (Tracker läuft per Prädiktion weiter) |

Es gibt keine Maßstabssuche: feste Kamerahöhe und gleichartige Teile sind Voraussetzung.
`min_score` (Default 0.85) ist deshalb bewusst hoch — lieber kein Treffer als eine um Pixel falsche Pose.
Laufzeit auf M3, 1280×720: lokal ≈ 25 ms, global ≈ 250 ms (`downscale: 4`) bzw. ≈ 115 ms (`downscale: 8`).

---

## ADR-010 — Bedienung über lokale Web-Oberfläche (stdlib-HTTP + MJPEG)

**Entscheidung:** `python -m ctrack.gui` (bzw. Doppelklick auf `Conveyor Tracking starten.command`) startet
einen HTTP-Server auf `127.0.0.1:8765` (nächster freier Port bis 8774) und öffnet den Browser. Eine
Seite (`gui/static/index.html`, ohne Build-Schritt und ohne externe Bibliotheken) bedient Quelle,
Erkennung, Einlernen, Tracking, Ausgabe, Aufnahme. Das Live-Bild kommt als MJPEG-Stream
(`/stream.mjpg`), Status per Polling (`/api/status`, 300 ms), Aktionen per JSON-POST.
Die Pipeline läuft in einem Hintergrund-Thread (`gui/engine.py`); Tracker und Filter werden nur von
diesem Thread angefasst, HTTP-Handler wirken über Einstellungen (jeden Frame neu gelesen) und eine
Kommando-Queue (Einlocken, Zurücksetzen).

**Begründung:** Bedienung ohne Terminalbefehle; Einlernen per Rechteck-Ziehen auf dem Bild braucht eine
echte GUI. Eine Webseite statt OpenCV-Fenster erlaubt Buttons, Galerie und Kennzahlen mit Zielwerten
und ließe sich später auch von einem anderen Rechner aus bedienen (dafür fehlt aktuell bewusst der Netzzugriff, siehe Sicherheit).

**Alternativen:** Flask/FastAPI + WebSocket (Abhängigkeit ohne Mehrwert für einen Nutzer);
Tkinter/Qt (Installation, kein Browser-Zugriff von anderem Rechner); Electron (Overkill);
OpenCV-Fenster mit Tasten (keine Galerie, kein Einlernen per Maus).

**Konsequenzen:**
- **Sicherheit:** Jede Webseite im Browser kann Anfragen an `localhost` schicken. Der Server prüft deshalb
  den `Host`-Header (gegen DNS-Rebinding) und verlangt für jede ändernde Anfrage `X-Requested-With: ctrack`
  (erzwingt CORS-Preflight, der nie freigegeben wird). Dateinamen (Referenzbilder, Logs) sind per Regex
  auf `[A-Za-z0-9_-]` bzw. `*.csv` beschränkt. Es gibt **keine Authentifizierung** — nur auf `127.0.0.1`
  betreiben; für Zugriff aus dem Netz wäre ein Token nötig.
- **POST-Body:** wird vor dem Routing immer vollständig gelesen (sonst zerlegt ein ungelesener Body bei
  Keep-Alive die nächste Anfrage; Regressionstest vorhanden).
- **Einstellungen:** Quelle/Detektor/Referenzbild neu → Sitzung startet neu; Horizont, Überbrückung,
  Bewegungsprofil, Mindestscore, Senden, Aufnahme gelten live. Zuletzt benutzte Werte liegen in
  `logs/gui_settings.json`; Senden, Aufnahme und Auto-Lock werden bewusst **nicht** wiederhergestellt.
- **Eine Sitzung, ein Nutzer:** kein Mehrbenutzerbetrieb, ein Browser-Tab genügt (mehrere sehen denselben Stream).
- **Aufwand:** MJPEG-Overlay + JPEG ≈ 3 ms pro Frame, beeinflusst die Bildrate nicht messbar.

---

## ADR-011 — Objekt aus ~5 Fotos anlernen: SAM-Klick + DINOv2-Merkmale + linearer Kopf

**Entscheidung:** Der Detektor `learned` findet ein vom Nutzer angelerntes Objekt, das in die Kamera gehalten wird
(wechselnder Abstand, leichte Drehung). Ablauf:
1. **Anlernen (Oberfläche):** ~5 Fotos aufnehmen, je **ein Klick** aufs Objekt → MobileSAM liefert die Maske (bei Bedarf
   weitere Klicks, rechte Maustaste = „gehört nicht dazu“).
2. **Training (~3 s/Foto):** Pro Foto entstehen ~12 Varianten (Objekt auf 15–80 % des Ausschnitts gezoomt, ±30° gedreht,
   verschoben, Helligkeit/Kontrast/Unschärfe/Rauschen; 60 % auf Hintergründe der anderen Fotos geklebt). Dazu kommen
   zufällige **Störformen** (Kreise, Vielecke, Rechtecke in Zufallsfarben) als Gegenbeispiele. Ein eingefrorenes
   DINOv2-small (14-px-Patches, 384-d) liefert Merkmale; ein klassengewichteter logistischer Kopf (Newton/IRLS in NumPy)
   lernt „Patch gehört zum Objekt“. Danach **Hard-Negative-Mining** (Hintergrund-Ansichten, die noch als Objekt gelten,
   werden Negativbeispiele) und Neutraining. Gespeichert wird nur der Kopf (`models/objects/<name>.npz`, wenige KB),
   dazu das Seitenverhältnis-Band der Masken (×0,7 … ×1,4).
3. **Erkennen:** Wahrscheinlichkeitskarte → Maske → Schwerpunkt = (x, y), Hauptachse = θ, Kontur fürs Overlay. Im Verfolgen
   wird nur ein Ausschnitt (1,8 × Objektgröße) um die letzte Position bewertet (Ausschnittskante `view_size`, 140/168/224 px),
   sonst/bei Unplausiblem das ganze Bild bei 448 px Breite (jeden 3. Frame, solange nichts gefunden wird). Masken mit
   unplausiblem Seitenverhältnis werden verworfen.

**Begründung:** Vorgabe des Nutzers: wenige Fotos, ein Klick, Objekt „zu variabel“ für Template-Matching (ADR-009) —
ein Referenzbild trägt weder Abstandsänderung noch Griffwechsel. Ein Kopf auf vortrainierten Merkmalen braucht keine
Trainingsdaten im Hunderterbereich und trainiert in Sekunden. Alle Komponenten sind Apache-2.0/MIT und laufen mit
`onnxruntime` auf der CPU.

**Alternativen:** YOLO(-OBB) feinjustieren (braucht Hunderte Bilder; Ultralytics ist AGPL — Lizenzfrage für VMT);
SAM-2-Videotracking (schwer, kein dauerhaftes Anlernen, kein Wiederfinden); OpenCV-Tracker (finden nichts, nur Box);
SIFT/ORB-Merkmale (brauchen Textur; dunkle glatte Objekte fallen durch); Farbsegmentierung (nur farbige Objekte).

**Gemessen** (Apple M3, ein Foto-Satz, synthetisch komponiert — *nicht* mit echtem Kamerabild und nicht über mehrere Objekte):

| Messung | Ergebnis |
|---|---|
| Position, ganzes Bild (30 Szenen, Größe ×0,4–1,5, Drehung ±25°, Hintergrund mit Hand und Gesicht) | Median 2,5 %, p90 7,0 %, max 8,8 % der Objektgröße; 30/30 gefunden |
| Winkel | Median 2,7°, p90 8° |
| Ohne Störformen und Seitenverhältnis (Zwischenstand) | p90 78 %: Verwechslung mit Hintergrund; roter Kreis wurde mit 92 % als Objekt erkannt |
| Gesamtpfad (Quelle + Erkennung + Tracker + Kalman) | 11 fps „Genau“ (224 px) · 15,5 fps „Ausgewogen“ (168 px) · 19,7 fps „Schnell“ (140 px); Positionsfehler 1,9 / 3,7 / 4,6 px |
| Backbone allein | 224 px: 62 ms · 168 px: 41 ms · 140 px: 31 ms · ganzes Bild 448×252: 157 ms |

**Grenzen / Entscheidungen:**
- **Bildrate:** das PRD-Ziel 25 fps wird nicht erreicht (max. ~20 fps). CoreML als ONNX-Provider war langsamer als die CPU.
  Möglicher Hebel: int8-quantisiertes DINOv2 (24 MB, nicht geladen) oder Erkennung asynchron zur Kamera.
- **θ nur modulo 180°:** Ein länglicher Körper hat eine Hauptachse ohne Richtung. θ wird gegen den Vorwert entfaltet
  (keine Sprünge), aber „oben/unten“ ist nicht unterscheidbar.
- **Objektgröße:** unter ~12 % der Bildbreite (< ~150 px bei 1280 px) werden Treffer unzuverlässig.
- **Ein Klick reicht oft nicht:** SAM liefert bei einem Klick teils nur einen Teil des Objekts; die Oberfläche zeigt die Maske
  sofort und lädt zu weiteren Klicks ein.
- **Hintergrund:** Gegenbeispiele stammen aus den eigenen Fotos. Fotos mit **wechselndem Hintergrund und Griff** anlernen.
- **Ausprobiert, verworfen:** echte Bildausschnitte (Hand, Gesicht, Raum) als zusätzliche Störobjekte aufs Bild kleben.
  Auf dem einzigen verfügbaren Foto-Satz wurde die Erkennung dadurch deutlich schlechter (Median 2,5 → 18–30 %, Kreis
  wieder akzeptiert); Ursache unklar (Teil vermutlich: ausgemalte Objektfläche sieht dem Objekt ähnlich). Mit echten
  Aufnahmen mehrerer Objekte erneut prüfen.
- **Download:** MobileSAM (28,2 + 16,5 MB, MIT) und DINOv2-small (88,5 MB, Apache-2.0), Button „Modelle laden“.
