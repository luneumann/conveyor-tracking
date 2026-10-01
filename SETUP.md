# Setup — Was du selbst tun musst

Der Code ist fertig und getestet (92 Tests grün, synthetische Pipeline und Web-Oberfläche laufen). Die folgenden Schritte
brauchen dich, weil sie Kamera, Downloads oder Entscheidungen betreffen.

## Erster Start (alles per Klick)

### 1. Programm starten

**Doppelklick auf `Conveyor Tracking starten.command`** (im Projektordner). Es öffnet ein Terminal-Fenster
(das offen bleiben muss, solange du arbeitest) und danach den Browser mit der Oberfläche
(`http://127.0.0.1:8765`). Beim allerersten Mal richtet es die Python-Umgebung selbst ein (einige Minuten).

- Meldet macOS „nicht geöffnet werden, da der Entwickler nicht verifiziert ist“: Rechtsklick auf die Datei →
  **Öffnen** → **Öffnen**. Das ist nur beim ersten Mal nötig.
- Beenden: Terminal-Fenster schließen (oder Ctrl+C).

### 2. Kamera-Berechtigung (macOS)

Beim ersten **Starten** mit Quelle *Kamera* fragt macOS nach Zugriff für das **Terminal**. Erlauben.
Falls abgelehnt: **Systemeinstellungen → Datenschutz & Sicherheit → Kamera →** Terminal aktivieren, das
Terminal-Fenster schließen und die Start-Datei erneut öffnen. Die Oberfläche zeigt bei diesem Fehler eine
Erklärung.

### 3. Ausprobieren ohne Kamera

Quelle **Demo-Band** → **Starten** → **Tracking starten**. Ein simuliertes Teil fährt durchs Bild; Kennzahlen
sollten grün werden. So siehst du in einer Minute, ob alles funktioniert.

### 4. Hand ausprobieren

Erkennung **Hand** wählen. Fehlt das Modell, erscheint **Handmodell laden** (7,8 MB von Google, einmalig).
Auf diesem Rechner ist es bereits geladen. Für bewegte Hand das Bewegungsprofil unter *Feineinstellungen* auf
**Hand** stellen (wird beim Wechsel der Erkennung automatisch gesetzt).

### 5. Objekt anlernen („Gelernt“, empfohlen)

1. Beim ersten Mal: Erkennung **Gelernt** → **Modelle laden** (ca. 133 MB, einmalig).
2. Quelle *Kamera* → **Starten**. Die Kamera läuft zunächst nur als Vorschau.
3. **Neues Objekt anlernen**. Objekt in die Kamera halten → **In 3 s aufnehmen** (so hast du beide Hände frei)
   oder **Foto aufnehmen**.
4. Auf das **Objekt klicken**. Zeigt die türkise Maske nur einen Teil: weiter auf den fehlenden Teil klicken.
   Zu viel markiert (z. B. die Hand): **rechte Maustaste** auf den Überschuss. → **Foto übernehmen**.
5. Das Ganze **~5 Mal** mit **anderem Abstand, leicht gedrehtem Objekt, anderem Griff und gern anderem Hintergrund**.
6. Name eintragen → **Trainieren** (ca. 15 s). Danach ist das Objekt aktiv; **Tracking starten**.

Tipps: Je unterschiedlicher die 5 Fotos, desto robuster. Wird fälschlich etwas anderes erkannt: **Mindest-Sicherheit**
(Feineinstellungen) erhöhen oder mit mehr/anderen Fotos neu anlernen. Ist es zu langsam: **Tempo** auf *Schnell*
(ca. 20 statt 15 Bilder/s, etwas ungenauer). Grenzen siehe ADR-011 (θ nur modulo 180°, Objekte kleiner als ~12 % der
Bildbreite werden unzuverlässig, max. ~20 fps).

### 5b. Flaches Teil per Referenzbild einlernen (Template-Matching)

1. Quelle *Kamera* → **Starten**, Erkennung **Bauteil**.
2. Teil so vor die Kamera legen, wie θ = 0 gelten soll.
3. **Aus Livebild einlernen**, Rechteck **eng um das Teil** ziehen, Namen eingeben, **Speichern**.
   (Alternativ **Bild laden …** mit einem Foto.)
4. **Tracking starten**.

Hinweise: Gleichbleibender Abstand Kamera–Teil (±10 % Größe verfälscht die Pose, ±20 % wird nicht erkannt),
ein asymmetrisches Merkmal am Teil, mindestens ~70 % sichtbar. **Kleine Referenzbilder sind schneller**
(Matching-Aufwand wächst mit der Fläche): lieber eng ziehen. Findet er das Teil nicht, unter
*Feineinstellungen* die **Mindest-Übereinstimmung** senken (Standard 85 %; niedriger heißt mehr Fehlmessungen).

### 6. Daten an einen Empfänger senden

Karte *Ausgabe & Aufnahme*: **Pose per UDP senden**, Adresse und Port eintragen (Standard
`127.0.0.1:5005`). Zum Testen auf demselben Rechner im Terminal `python tools/receiver.py`
(das ist der einzige Schritt, der ein Terminal braucht).

## Für Fortgeschrittene (Terminal)

Diese Schritte brauchen die Kommandozeile (`source .venv/bin/activate` vorher).

### Belichtungs-Offset messen (P1-3, empfohlen vor Messungen)

Der Default `exposure_offset_ms: 30` ist geschätzt. Messen:

```bash
python tools/latency_probe.py
```

Die Kamera muss das Fenster "probe" sehen — eine Laptop-Webcam sieht den eigenen Bildschirm nicht:
Spiegel davor halten oder eine externe Webcam auf den Laptop-Bildschirm richten. Den ausgegebenen
Wert in `config/default.yaml` → `camera.exposure_offset_ms` eintragen.

### Filter-Tuning (Hand-Profil ist voreingestellt, Bestätigung steht aus)

`process_noise: 50` aus dem PRD ist für eine Hand viel zu träge: Im ersten Live-Lauf (19 s getrackt)
war die 100-ms-Prädiktion damit **schlechter als gar keine Prädiktion** (p95 97 px = 7,6 % der Breite
gegenüber 39 px, wenn man die letzte Pose einfach hält). Offline über die geloggten Detektionen
nachgerechnet (Einheit px²/s³, Messrauschen 4 px):

| process_noise | p95 Fehler | % Bildbreite |
|---|---|---|
| 50 (PRD) | 97,5 px | 7,61 % |
| 5 000 | 35,9 px | 2,81 % |
| 100 000 | 20,0 px | 1,57 % |
| **300 000** (jetzt Default) | 16,7 px | 1,31 % |
| 10 000 000 | 18,3 px | 1,43 % |

Plateau zwischen 1e5 und 1e6, daher 3e5. `config/default.yaml` und `config/replay.yaml` sind angepasst;
`config/synthetic.yaml` bleibt bei 50 (konstante Bandgeschwindigkeit). Einschränkungen: **ein Lauf**
mit frei bewegter Hand, nicht der PRD-Messung "gleichmäßig in eine Richtung" — die steht noch aus.
Hohes Rauschen macht außerdem die Geschwindigkeitsschätzung im Stream unruhiger (noch nicht gemessen).

Selbst nachtunen mit Aufnahme und Replay:

```bash
python -m ctrack --record recordings/demo                      # Hand bewegen, Q
for q in 5000 50000 300000; do
  python -m ctrack -c config/replay.yaml --headless --auto-lock \
    -s camera.path=recordings/demo.mp4 -s predictor.process_noise=$q -s metrics.csv=logs/q$q.csv
  python tools/analyze.py logs/q$q.csv | grep "pred err"
done
```

### GitHub

Das Projekt liegt in einem privaten Repo: https://github.com/luneumann/conveyor-tracking (Branch `main`).

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
- **Re-Acquire:** Referenzposition in LOST ist auf `coast_ms` gedeckelt, das Gate wächst mit der Zeit in LOST
  (SYSTEM-DESIGN §3). Offline an einem Live-Lauf geprüft (0,53 s), live noch nicht erneut gemessen.
  Der Startwert 600 px/s ist geschätzt.
- **Stream-Erweiterung:** Feld `"v": 1` zusätzlich zu PRD 5.4 (Versionierung für P2-7).
- **Synthetik am Bildrand:** Wird das Objekt vom Bildrand abgeschnitten, verschiebt sich der
  Schwerpunkt (Ground-Truth-Fehler bis ~20 px nur dort; innen ~0,05 px). Betrifft nur die Synthetik.
- **Latenz bei Replay** misst nur die Verarbeitungszeit (`t_sent − t_read`), weil `t_exposure`
  aus der Aufnahme stammt (ADR-007).
- **Gelerntes Objekt nur synthetisch und an einem Foto-Satz geprüft.** Alle Genauigkeitswerte (ADR-011) stammen von einem
  synthetisch komponierten Teil bzw. von Szenen, die aus *einem* Handyfoto zusammengesetzt wurden. Mit echten
  Kameraaufnahmen, anderen Objekten und anderen Hintergründen ist noch nichts gemessen — dafür sind echte Aufnahmen nötig.
- **Bildrate des gelernten Objekts 11–20 fps** (PRD-Ziel 25): Das Bildmodell läuft auf der CPU (CoreML war langsamer).
- **θ des gelernten Objekts nur modulo 180°** (Hauptachse der Maske).
- **Beim Beenden von Python erscheint gelegentlich** `libc++abi: terminating … recursive_mutex lock failed` aus einer
  nativen Bibliothek (MediaPipe/ONNX). Exit-Code und Ergebnisse sind nicht betroffen; Ursache nicht untersucht.
- **Web-Oberfläche ohne Anmeldung.** Sie läuft nur auf diesem Rechner (`127.0.0.1`) und prüft Host und Header gegen
  Angriffe aus dem Browser, hat aber kein Login (ADR-010). Nicht ins Netz öffnen.
- **Demo-Band hat ~25–28 fps** (die Simulation selbst kostet Rechenzeit); die Kamera-Bildrate ist noch nicht in der GUI gemessen.
- **Template-Matching nur synthetisch geprüft.** Genauigkeit 0,1 px / 0,09° und alle Tests beziehen sich
  auf ein gerendertes Teil. Mit einem echten Bauteil (Spiegelungen, Schatten, Perspektive) ist noch nichts
  gemessen. Grenzen (Maßstab, Verdeckung) in ADR-009.
- **Globale Suche ist langsam** (≈ 250 ms bei 1280×720): nach einem Verlust sinkt die Bildrate kurz.
- **Single-Thread:** Falls Webcam + MediaPipe < 25 fps, Kamera-Grab in eigenen Thread auslagern.

## Kosten-Übersicht

Keine laufenden Kosten. Alle Abhängigkeiten sind Open Source (OpenCV: Apache 2.0, MediaPipe:
Apache 2.0, NumPy: BSD, PyYAML: MIT, Matplotlib: PSF-basiert). Kein Cloud-Dienst, kein Account.
Stufe 2: Kosten der IDS-Kamera; IDS peak SDK ist kostenlos.
