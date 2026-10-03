# Optimierung des gelernten Objekt-Detektors — Recherche und Messungen

Stand: 02.10.2026 (Ergebnisse der Umsetzung siehe Abschnitt 6) · Bezug: ADR-011 (`docs/ARCHITECTURE.md`)

**Lesehinweis:** „Gemessen" = auf diesem Rechner (Apple M3, 4 Leistungs- + 4 Effizienzkerne) selbst nachgemessen.
„Recherche" = aus den verlinkten Quellen, **nicht** hier überprüft. Alle Genauigkeitszahlen stammen von
synthetisch komponierten Szenen, nicht von echten Kameraaufnahmen.

## 1. Wo geht die Zeit hin? (gemessen)

| Messung | Ergebnis |
|---|---|
| Erkennung pro verfolgtem Frame (168-px-Ausschnitt) | 33,7 ms, davon **Bildmodell 33,0 ms (98 %)**; Zuschneiden 0,2 · Klassifikator 0,1 · Maske/Pose 0,4 ms |
| Gesamtpfad, Objekt dauerhaft sichtbar, ruhiger Rechner, vorab gerenderte Bilder | **224 px: 15,1 fps · 168 px: 25,4 fps · 140 px: 32,7 fps** (Positionsfehler median 2,1 / 3,6 / 3,7 px) |
| Dieselbe Last, aber Oberfläche + Browser laufen parallel, Demo-Quelle, Objekt zeitweise außerhalb des Bildes | 11 / 15,5 / 19,7 fps — die früher genannten Werte sind **durch Last und Demo-Eigenheiten zu pessimistisch** |
| Bildmodell, ONNX-Runtime-Threads (168 px) | 1: 75 ms · 2: 44 · **4: 27,6** · **6: 27,1** · 8: 33,1 ms (Standard 29,4) — die 4 Effizienzkerne bremsen |
| Bildmodell, 4 Ausschnitte in einem Aufruf (Batch) | 84 ms je Aufruf = 21 ms je Ausschnitt (−30 %) |
| CoreML (ONNX-Runtime-Provider) | langsamer als CPU: 74 Teilgraphen, 425 von 648 Knoten unterstützt; MLProgram+statisch 76 ms statt 62 ms bei 224 px |

**Folgerung:** Nur das Bildmodell zählt. Nachbearbeitung zu optimieren lohnt nicht. Hebel sind: das Modell schneller
machen, es seltener aufrufen, oder die Kamera-/Ausgaberate von ihm entkoppeln.

## 2. Geschwindigkeit — Hebel nach Nutzen/Aufwand

| # | Hebel | Erwarteter Gewinn | Aufwand | Risiko / offene Punkte |
|---|---|---|---|---|
| S1 | **Erkennung asynchron zur Kamera**, Kalman verarbeitet die verspätete Messung mit ihrem Aufnahmezeitpunkt (Out-of-Sequence-Messung: Zustand aus dem Verlauf zurückholen, aktualisieren, neu vorwärts rechnen) | Ausgabe- und Kamerarate unabhängig von der Erkennung (30 fps und mehr), **konsistente Latenzkompensation** — genau der Zweck des Projekts | mittel | Erkennung läuft mit ~10–25 Hz; Kalman-Verlauf nötig. [Stone Soup: Kalman-Filter mit OOSM](https://stonesoup.readthedocs.io/en/v1.8/auto_examples/oosm/KalmanFilterOOSMExample.html) |
| S2 | **INT8-Quantisierung von DINOv2** | auf ARM-CPUs mit Dot-Product-Befehlen 1,8–2,1× (Messstudie, [arXiv 2609.16085](https://arxiv.org/pdf/2609.16085)) → ~15 ms statt ~30 ms | gering (fertige Datei `model_int8.onnx`, 24,4 MB, existiert laut [Hugging Face](https://huggingface.co/onnx-community/dinov2-small/tree/main/onnx)) | Die Quelle misst nicht DINOv2 selbst; Merkmalsqualität muss mit unserer Auswertung gegengeprüft werden. **Download nötig (nicht freigegeben).** |
| S3 | **Schneller Tracker zwischen den Erkennungen** (OpenCV `TrackerVit`: 767 KB, 4,2 ms auf Apple M2 im Einzel-Thread, liefert Konfidenz zum Erkennen von Verlust — [OpenCV](https://docs.opencv.org/4.9.0/d9/d26/classcv_1_1TrackerVit.html), [opencv_zoo](https://huggingface.co/opencv/opencv_zoo/blob/8a1a70cce93b53bd55cebced420389736ccb27e7/models/object_tracking_vittrack/README.md)) | Erkennung nur jeden 3.–5. Frame → Kamerarate bleibt voll | mittel | liefert nur eine Box, keine Maske/θ; Drift; Modelldateien müssten geladen werden. Mit S1 teils überflüssig |
| S4 | **Kleinerer Ausschnitt** („Schnell“, 140 px) | schon vorhanden: 32,7 fps | keiner | Winkel etwas ungenauer (Median 3,1° statt 1,9°) |
| S5 | **ONNX-Runtime-Threads auf 4 begrenzen** | ~0–10 %, weniger Schwankung | sehr gering | Wert hängt vom Rechner ab (hier 4 Leistungskerne) |
| S6 | **Neural Engine über `coremltools`** (fester Eingang 168×168) | möglicherweise mehrfach schneller; die Neural Engine braucht feste Formen, dynamische sind 25–50× langsamer ([Apple-Forum](https://developer.apple.com/forums/thread/724930), [ORT-Doku](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider)) | hoch | neue Abhängigkeit, Konvertierung; ungeklärt, ob alle Operationen der ViT auf die Neural Engine passen. Der ORT-Weg hat nichts gebracht. Als Orientierung: [EdgeTAM](https://arxiv.org/abs/2501.07256) läuft mit 16 fps auf dem iPhone 15 Pro Max |
| S7 | **Batch** bei mehreren Ausschnitten | −30 % je Ausschnitt | gering | nur sinnvoll bei mehreren Objekten/Maßstäben |
| S8 | **DINOv3-ViT-S/16** (21 M Parameter, 384-d) | bessere dichte Merkmale (Qualität, nicht Tempo) | mittel | Download **gesperrt** (Anmeldung, Weitergabe der Kontaktdaten) und eigene [„DINOv3 License“](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) — gewerbliche Nutzung rechtlich prüfen. Patchgröße 16 statt 14 |

## 3. Qualität — Hebel

| # | Hebel | Nutzen | Aufwand | Risiko / offene Punkte |
|---|---|---|---|---|
| Q1 | **„Leere Szene aufnehmen“**: 3–5 s Live-Bild ohne Objekt als echte Gegenbeispiele (+ Hard-Negative-Mining darauf) | direkt die Idee „alles im Bildfeld außer dem Objekt“, aber mit echten Pixeln statt Ausschnitten aus dem Foto (das hat auf unserem Testsatz geschadet) | gering | muss mit echten Aufnahmen bestätigt werden |
| Q2 | **Auswertung mit echten Aufnahmen**: Aufnahme abspielen, in wenigen Frames per Klick Referenzmasken (SAM) setzen → IoU, Schwerpunkt-, Winkelfehler, Erkennungsrate | Voraussetzung für jede ernsthafte Optimierung; ersetzt die synthetischen Zahlen | mittel | braucht von dir Aufnahmen verschiedener Objekte |
| Q3 | **Drehrichtung (θ nur modulo 180°)**: zusätzlicher Kopf, der die Position entlang der Achse aus den Patch-Merkmalen vorhersagt (Trainingsbeschriftung ergibt sich aus der bekannten Augmentations-Drehung); sein Vorzeichen löst ±180° auf. Optional ein zweiter Klick „Vorderseite“ beim Anlernen | eindeutiges θ bei asymmetrischen Objekten | mittel | funktioniert nur, wenn das Objekt wirklich eine erkennbare Vorder-/Rückseite hat. Klassisch löst man das Problem mit einer zweiten Suchstufe, die beide Kandidaten (θ und θ+180°) prüft — die gefundenen Quellen dazu sind Patente und nur Hinweise |
| Q4 | **Leichte Kippung** (Homographie-Augmentierung) | robuster bei „leicht gedreht“ im Raum | gering | Wirkung nur mit echten Aufnahmen prüfbar |
| Q5 | **Schärfere Maske**: Konturverfeinerung im Ausschnitt in voller Auflösung (z. B. GrabCut, vom Klassifikator vorbelegt) oder kantenbewusstes Glätten; [EdgeSAM](https://arxiv.org/pdf/2312.06660) (laut Autoren >30 fps auf dem iPhone 14, Eingabeaufforderungen in der Destillation) als schnelleres SAM zum Verfeinern | Positionsfehler von ~3 % der Objektgröße weiter senken | mittel | aktuelle Genauigkeit reicht für den Prototyp vermutlich; zunächst Q2 |
| Q6 | **Selbstlernen im Betrieb**: sichere Treffer als zusätzliche Beispiele nachtrainieren | passt sich Licht/Hintergrund an | mittel | Drift-Gefahr (Fehler verstärken sich); nur mit Kontrolle |
| Q7 | **Messrauschen im Kalman an Maskengröße/Konfidenz koppeln** | ruhigere Pose, weniger Geschwindigkeits-Jitter | gering | Parameter erst mit echten Daten sinnvoll einstellbar |

## 4. Einordnung gegenüber der Literatur

Der gewählte Ansatz (eingefrorene DINOv2-Merkmale + einfacher Kopf, SAM für die Maske) entspricht dem, was die
Few-Shot-Literatur verwendet: [Matcher](https://arxiv.org/pdf/2305.13310) kombiniert DINOv2 (ViT-L/14) und SAM
trainingsfrei und schlägt [PerSAM](https://arxiv.org/pdf/2305.13310) deutlich; für DINOv2 gilt, dass einfaches
Linear-Probing mit aufwendigeren Anpassungen mithalten kann. Diese Verfahren laufen mit großen Modellen auf Server-GPUs;
unsere Variante tauscht Genauigkeit gegen CPU-Tempo (ViT-S, 168-px-Ausschnitte).
Für das Folgen nach einem Klick gibt es End-to-End-Alternativen ([EdgeTAM](https://arxiv.org/abs/2501.07256):
SAM-2-Qualität, 22× schneller als SAM 2, Video-Version bereits in [Transformers](https://huggingface.co/docs/transformers/model_doc/edgetam_video)),
sie speichern aber nichts dauerhaft Gelerntes und finden das Objekt nicht von selbst wieder.

## 5. Empfohlene Reihenfolge

1. **Q2 – echte Aufnahmen und Messwerkzeug.** Ohne sie sind alle weiteren Entscheidungen Raten.
2. **Q1 – leere Szene aufnehmen** (klein, direkt nutzbar).
3. **S1 – asynchrone Erkennung mit verspäteter Messung im Kalman.** Größter Hebel auf Kamera-/Ausgaberate und der Kern
   der Latenzkompensation.
4. **S2 – INT8** gegenmessen (nach deiner Freigabe des Downloads) und nur übernehmen, wenn die Genauigkeit hält.
5. **Q3 – Drehrichtung**, falls die Anwendung „vorne/hinten“ braucht.
6. S3/S6/S8 nur, wenn S1+S2 nicht reichen bzw. nach rechtlicher Prüfung (S8).

## 6. Was daraus umgesetzt wurde und was dabei herauskam (02.10.2026)

| Punkt | Ergebnis |
|---|---|
| **S1 asynchrone Erkennung** | umgesetzt (ADR-012). 30 fps Kamera-/Ausgaberate unabhängig vom Detektor. Kein Out-of-Sequence-Handling nötig, weil `predict` den Filter nie verändert. Kennzahl „Alter der Messung“ ergänzt. |
| **S2 INT8** | umgesetzt und gemessen: 1,6–1,7× schneller als fp32 auf der CPU (168 px: 17 statt 28 ms), Genauigkeit im Rauschen gleich. Wird nur auf der CPU genutzt (CoreML kann INT8 nicht). Erzeugt mit `onnxruntime.quantization` in einer **getrennten** Umgebung `.venv-tools` (`onnx` braucht `protobuf ≥ 6.31`, MediaPipe `< 5`). |
| **S5 Threads** | `intra_op_num_threads = 4` fest. |
| **S6 Neural Engine** | **funktioniert** (ADR-013), entgegen dem ersten Versuch: 6 ms statt 28 ms bei 168 px, ganzes Bild 27 statt 157 ms. Voraussetzung sind feste Formen, kein `coremltools`/PyTorch nötig. |
| **Q1 leere Szene aufnehmen** | umgesetzt (Knopf in der Oberfläche; 12 Live-Bilder als echte Gegenbeispiele). Wirkung nur im harten Lookalike-Test gemessen (ADR-011-Nachtrag), nicht mit echten Aufnahmen. |
| **Training stabilisieren** | *nicht* auf der Liste, aber der größte Befund: Das Training war instabil (31 % Ausfälle, je Lauf 0–52 von 60). `l2 = 300` statt 1 → 3 %. Die früheren „30/30“ waren Glück. |
| **Q2, Q3, Q4, Q5, Q6, Q7, S3, S7, S8** | offen. Q2 (echte Aufnahmen auswerten) bleibt der wichtigste nächste Schritt. |

**Zeiten sind nur auf ruhigem Rechner vergleichbar:** Zeitweise belegten macOS-Hintergrunddienste (`photolibraryd`, `photoanalysisd`, `appstoreagent`) den
Rechner (Last > 50). Messungen aus solchen Phasen wurden verworfen bzw. wiederholt; ebenso eine Überlastung durch drei gleichzeitig gestartete eigene Jobs.

## 7. Genauigkeit der Maske/Pose auf einer echten Aufnahme (02.10.2026)

**Anlass:** Die markierte Region um das Objekt war eine ungenaue „Wolke" (14-px-Kachelraster, hochgerechnete Wahrscheinlichkeitskarte).

**Aufbau:** Eine eigene Webcam-Aufnahme (Handy und weißes Quadrat in der Hand, dazu leere Szenen). Je Objekt 5 Anlernbilder, getestet auf
anderen Bildern desselben Clips (14 Testbilder, davon 8 Handy, 6 Quadrat). Referenzmasken = SAM mit von Hand gesetzten Boxen, visuell geprüft
(drei Bilder verworfen). Wegen n = 14 und nur einer Aufnahme zählt die Reihenfolge, nicht die Nachkommastelle; die Referenz stammt selbst von
SAM, was SAM-basierte Verfahren begünstigt.

| Methode | IoU (Median) | Mittelpunktfehler, % der Objektgröße (Median / p90) | Winkelfehler ° (Median / p90) |
|---|---|---|---|
| bisher: ganzes Bild, 448 px breit | 0,74 | 6,4 / 14 | 6,3 / 48 |
| DINO-Ausschnitt 168 px | 0,70 | 8,3 / 13 | 6,0 / 31 |
| **DINO-Ausschnitt 336 px** | **0,83** | **3,3 / 8** | **4,3 / 23** |
| SAM-Verfeinerung (Box + Punkt, ~500 ms CPU) | 0,82 | 7,0 / 13 | 10,8 / 28 |
| SAM auf 2×-Ausschnitt | 0,60 | 9,1 / 19 | 19,7 / 49 |

**Folge:** Die Einstellung „Genau" nutzt jetzt 336 px (vorher 224). SAM-Verfeinerung bringt gegenüber dem 336-px-Ausschnitt keinen Gewinn und
ist wesentlich langsamer → vorerst nicht eingebaut.

**Kosten je Erkennung** (Detektor-Schritt auf derselben Aufnahme, 500 Bilder, Modell mit fp32-Backbone):

| Ausschnitt | CPU | Neural Engine |
|---|---|---|
| 168 px | 33 ms (29/s) | 11 ms (88/s) |
| 224 px | 56 ms (18/s) | 18 ms (56/s) |
| 336 px | 257 ms (3,8/s) | 28 ms (34/s) |

Mit der Neural Engine kostet „Genau" also kaum Tempo; ohne sie ist es deutlich langsamer (INT8 auf CPU: ca. 150 ms). Ein Modell, das mit dem
INT8-Backbone gelernt wurde, kann die Neural Engine nicht nutzen (CoreML nimmt nur fp32) — beim Messen aufgefallen, weil „beschleunigt" und
„CPU" zunächst identische Zeiten ergaben.

**Offen:** Fehlalarme auf leeren Bildern (ohne Score-Schwelle: 14 von 38 Bildern mit irgendeinem Fleck); Anteil im echten Detektor mit
`min_score` noch nicht gemessen. Genauigkeit des weißen Quadrats (IoU 0,80) liegt unter dem des Handys (0,86).

## 8. „Genau" verlor das Teil öfter als „Schnell" — Ursache und Korrektur (02.10.2026)

**Messung:** Echtzeit-Simulation (Aufnahme mit 30 fps in die asynchrone Pipeline gespeist, Handy-Segment, 480 Bilder), Anteil Bilder im Zustand TRACKING:

| Variante | TRACKING | LOST |
|---|---|---|
| nur 168 px (Neural Engine) | 95 % | 0 % |
| nur 336 px (Neural Engine, warm) | 85 % | 10 % |
| nur 336 px (CPU, ca. 250 ms je Messung) | **0 %** | 89 % |
| nur 336 px, direkt nach dem Start (Modell lädt noch) | 6 % | – (88 % SEARCHING) |
| **zweistufig: 168 px verfolgt, 336 px verfeinert** (Neural Engine) | **95 %** | **0 %** |
| zweistufig auf CPU (Verfeinerung wird übersprungen, da > 80 ms) | 83 % | 4 % |

**Ursachen:** (1) Mit großem Ausschnitt dauert jede Messung länger, die Hand wandert aus dem Suchausschnitt, bevor das Ergebnis da ist. (2) Das Laden
eines CoreML-Modells hält den GIL 6–14 s und legte die Pipeline lahm, sobald eine neue Größe zum ersten Mal gebraucht wurde.

**Korrektur:** „Genau" = zweistufig: gefunden und verfolgt wird mit 168 px (robust, jedes Bild), die Maske wird danach aus 336 px an derselben Stelle
neu berechnet (feinere Kontur). Schlägt die Verfeinerung fehl oder ist sie langsamer als 80 ms (z. B. nur CPU), bleibt das grobe Ergebnis.
Beschleuniger-Größen werden beim Start vorbereitet, bevor die Kamera läuft (mit vorhandenem Cache ca. 25 s, beim allerersten Mal länger).

## 9. Suchausschnitt an der Kalman-Vorhersage ausgerichtet (03.10.2026)

Im asynchronen Betrieb gibt die Pipeline dem Detektor je Bild die vom Tracker erwartete Pose zum Belichtungszeitpunkt dieses Bildes mit
(`Detector.hint`). Der lernende Detektor sucht zuerst dort und erst bei Misserfolg (und wenn der Hinweis vom letzten Fundort abweicht) am letzten Fundort.

Echtzeit-Simulation wie in Abschnitt 8 (Handy-Segment, 480 Bilder, 2 Läufe je Variante auf der CPU, 224 px; Neural Engine 168 px mit Verfeinerung):

| Variante | TRACKING | LOST | Verlust-Episoden |
|---|---|---|---|
| 224 px CPU, ohne Hinweis | 76 / 73 % | 3 / 4 % | 1 / 4 |
| 224 px CPU, mit Hinweis | 76 / 74 % | 1 / 2 % | 2 / 2 |
| 168+336 Neural Engine, ohne Hinweis | 93 % | 1 % | 1 |
| 168+336 Neural Engine, mit Hinweis | 95 % | 0 % | 0 |

Der Effekt ist klein und klar im Rauschbereich eines einzelnen Clips (LOST ungefähr halbiert, TRACKING-Anteil unverändert). Er verändert nichts am
Grundproblem langsamer Messungen: 336 px allein auf der CPU bleibt unbrauchbar (0 % TRACKING).

## 10. Fehlalarme: Messung und Gegenmaßnahme (03.10.2026)

**Werkzeug:** `python tools/eval_recording.py tools/eval_labels_demo.json` misst auf einer eigenen Aufnahme Maskengenauigkeit und Fehlalarme je
Score-Schwelle (Training auf wenigen Bildern, Messung auf anderen). Negative = leere Szene **und** Bilder mit einem anderen Objekt (Verwechslungstest).
Die Labels-Datei enthält nur Boxen (SAM macht daraus die Referenzmaske), keine Bilder.

**Befund (Handy-Modell, Ganzbildsuche, Score = Erkennungssicherheit):**

| Schwelle | Treffer Handy | Fehlalarm leere Szene (16 Bilder) | Fehlalarm weißes Quadrat im Bild (20 Bilder) |
|---|---|---|---|
| 0,5 | 100 % | 6 % | 55 % |
| 0,6 (bisheriger Standard) | 100 % | 0 % | 50 % |
| 0,7 | 100 % | 0 % | 25 % |
| 0,8 | 100 % | 0 % | 5 % |
| 0,9 | 50 % | 0 % | 0 % |

Für das Quadrat-Modell sieht es ähnlich aus (Verwechslung mit dem Handy: 25 % bei 0,6, 3 % bei 0,7, 0 % bei 0,8).
Echte Treffer lagen bei 0,84–0,93, Verwechslungen bei 0,60–0,82. **Das Hauptproblem ist also nicht die leere Szene, sondern die Verwechslung
mit einem anderen, ähnlich gehaltenen Objekt.** (Meine erste Auswertung zeigte 16 % „Fehlalarme" in leeren Szenen; das waren Bilder, in denen das Handy
teilweise im Bild war — die Labels waren falsch, nicht der Detektor.) Solidität und Rechteckigkeit der Kontur trennen Treffer und Verwechslungen nicht.

**Maßnahme:** Zwei Schwellen. Ein bereits verfolgtes Objekt wird mit `min_score` (0,6) weiterverfolgt (Ausschnittssuche). Eine Ganzbildsuche hat keinen Anker und
verlangt `min_score + 0,2` (0,8), außer das Ergebnis liegt dort, wo der Tracker das Objekt erwartet (Hinweis, Abschnitt 9).

**Preis:** Neu einlocken geht etwas schwerer. Neural Engine, Echtzeit-Simulation: TRACKING 95 % → 91–93 %, weiterhin 0 % verloren. Auf der CPU waren die
Simulationen am 03.10. wegen Last auf dem Rechner (Load ≈ 10) nicht belastbar; der Vergleich dort ist offen.

**Bekannte Grenzen:** Das Quadrat hat keine Hauptachse — der Winkel aus den Momenten ist bei quadratischen Masken praktisch zufällig (Median 34° Fehler in
dieser Messung). Eine Winkelbestimmung aus der Kontur (modulo 90°) fehlt noch.

## 11. Zwischen-Tracking mit optischem Fluss (03.10.2026, Prototyp, nicht in die Pipeline eingebunden)

**Idee:** Zwischen zwei Erkennungen verfolgt `ctrack/flow.py` (`FlowTracker`) die Textur des Objekts von Bild zu Bild (Lucas-Kanade auf Punkten innerhalb der
letzten Kontur, Vorwärts-Rückwärts-Prüfung, RANSAC-Ähnlichkeitsfit) und schreibt (x, y, θ) fort. Jede Erkennung korrigiert den Fluss rückwirkend
(`correct`: die Verschiebung seit dem Aufnahmezeitpunkt der Erkennung wird aufgerechnet). Ca. 1–3 ms je Bild, kein Modell, kein Download, liefert auch θ.
(Ein Box-Tracker wie TrackerVit hätte nur eine Box und ein Modell zum Herunterladen gebraucht.)

**Messung:** Deterministische Simulation auf dem Handy-Segment der Aufnahme. Referenzpose = präziser Detektor (168 + 336 px) je Bild. Simuliert wird ein
langsamer Detektor: ein Ergebnis alle K Bilder, das D Bilder später eintrifft. Fehler der veröffentlichten Pose gegenüber der Referenz:

| Erkennung | ohne Fluss: Median / p90 / max (px) | mit Fluss: Median / p90 / max (px) |
|---|---|---|
| alle 2 Bilder, 100 ms alt | 17,5 / 43 / 129 | 11,1 / 67 / 173 |
| alle 4 Bilder, 133 ms alt | 30,7 / 81 / 226 | 14,3 / 104 / 260 |
| alle 8 Bilder, 267 ms alt | 68 / 148 / 328 (nur 49 von 440 Bildern verfolgt) | 50 / 250 / 512 (410 von 440 verfolgt) |

Winkelfehler (Median): 2,8° → 2,3° bzw. 4,4° → 2,8°.

**Ergebnis, ehrlich:** Der Fluss halbiert den *typischen* Fehler und hält das Teil bei seltenen Erkennungen im Verfolgen — verschlechtert aber die
*schlechtesten 10 %* (p90 und Maximum). Ursache laut Einzelbild-Analyse: Bei schneller Handbewegung (30–60 px je Bild) einigen sich die Punkte auf
„nichts hat sich bewegt" (Bewegungsunschärfe, statischer Hintergrund gewinnt die RANSAC-Abstimmung), der Fluss meldet dem Filter hoch-zuversichtlich Stillstand und
zerstört dessen Geschwindigkeitsschätzung. Drei Gegenmittel wurden gemessen und halfen **nicht**: höheres Messrauschen für Fluss-Messungen, Verwerfen großer
Schritte, Abgleich mit der erwarteten Verschiebung des Filters; ebenso eine Geschwindigkeitsgrenze (Median wird dadurch wieder schlechter, die Ausreißer bleiben).

**Deshalb nicht eingebunden.** Der Test ist an *ruckartiger Handbewegung* gemessen, dem schwierigsten Fall für Fluss; ein gleichmäßig laufendes Band (das Ziel)
ist günstiger für den Fluss, aber dort ist auch das Kalman-Modell mit konstanter Geschwindigkeit bereits gut. Ob sich die Einbindung lohnt, sollte an einer
Bandaufnahme entschieden werden. Neu ist außerdem `Detection.noise_scale` (Messrauschen je Messung), vorbereitet für gewichtete Fluss-Messungen.

## 12. Test auf einem Bildschirmvideo: Karosse auf Förderband (03.10.2026)

**Material:** Bildschirmaufnahme eines Stock-Videos (Karosse auf Förderband, Kamera schwenkt und zoomt leicht, 344 Bilder, 27 fps, 1534×862, mit Wasserzeichen). Liegt lokal
in `recordings/` und wird nicht committet. Labels (Boxen) in `tools/eval_labels_belt.json`. Angelernt mit 5 Bildern, **ohne** leere Szenen (das Objekt ist immer im Bild);
gemessen auf 6 anderen Bildern (Referenzmasken per SAM, visuell geprüft).

| Kennzahl | Wert |
|---|---|
| IoU Maske (Median / p10) | 0,79 / 0,77 |
| Mittelpunktfehler (% der Objektgröße, Median / p90) | 1,9 / 2,3 |
| Winkelfehler (Median / p90) | 0,5° / 1,6° |
| Bilder mit Treffer bei Erkennung je Bild, Standardeinstellung | 330 von 344 (Fehlstellen v. a. am Bildrand, wo die Karosse abgeschnitten ist) |

**Echtzeit-Simulation** (asynchron, Kalman „Band"-Profil, mit Vorhersage-Hinweis), Anteil Bilder TRACKING / verloren:
Neural Engine 140 px: 98 % / 0 % · 168 px: 91 % / 0 % · 168 + 336 px Verfeinerung („Genau"): 96 % / 0 % · nur CPU 168 px: 89 % / 0 %.
Schwankung zwischen Läufen ca. ±5 Prozentpunkte.

**Fluss-Zwischentracking auf diesem Video** (Abschnitt 11, gleiche Simulation): Positionsfehler Median 7,8 → 7,1 px, p90 21,7 → 18,3 px bei einer Erkennung alle 2 Bilder;
bei gleichmäßiger Bewegung also kein Schaden mehr, aber auch nur ein kleiner Gewinn, weil das Kalman-Profil „Band" schon gut ist. Nur bei sehr seltenen Erkennungen
(alle 8 Bilder = 3 Hz) hält der Fluss das Objekt im Verfolgen, wo die Kalman-Variante es verliert (308 statt 33 von 310 Bildern). Mit den schnellen Erkennungsraten dieser
Maschine bleibt er nicht nötig.

**Grenzen:** Ein einziges Objekt, ein einziges Video, 6 Testbilder, Stock-Szene (kein echter Aufbau, keine mm). Aussagekräftig ist vor allem: das Verfahren trägt auf einem
großen, gleichmäßig bewegten Objekt, auch mit schwenkender Kamera.

## 13. Mögliche Erweiterung, nicht Standard: YOLO-Detektor (03.10.2026)

**Entscheidung (Projektleitung):** YOLO wird als optionaler Zusatz-Detektor im Hinterkopf behalten, **nicht** als Standard. Standard bleibt der Few-Shot-Detektor (5 Fotos, ein Klick).

**Begründung:** YOLO braucht gelabelte Bilder je Klasse (typisch Hunderte), liefert ohne OBB/Segmentierungsvariante nur achsenparallele Boxen (kein θ), die Ultralytics-Lizenz ist meines Wissens AGPL-3.0 (vor Einsatz klären) und es wurde in diesem Projekt **nicht getestet**.

**Wann es sich lohnt:** feste, bekannte Serienteile mit 100–300 labelbaren Bildern und Bedarf an wenigen ms je Erkennung (Stufe 2). Einhängen als neue `Detector`-Klasse per Registry, Tracker/Ausgabe bleiben unverändert; für θ eine OBB- oder Segmentierungsvariante nehmen und mit `tools/eval_recording.py` gegen den Standard messen.

**Quelle der Einordnung:** liveimagetrackingtools.org ist eine Zell-Tracking-Community (Mikroskopie, offline, mehrere Objekte); übertragbar wäre höchstens die Zuordnung mehrerer Teile (LAP-Verfahren) und die Trennung von Erkennungs- und Verknüpfungsmetriken.

## 14. EdgeTAM-Test (03.10.2026): Maskenverfolgung mit Memory statt Kachelmaske

**Was getestet wurde:** EdgeTAM (Apache-2.0, 13,9 M Parameter, SAM-2-Ableger) in einer getrennten Umgebung (PyTorch, nicht im Projekt-venv). Eingabe: **eine Box im ersten Bild**, danach
wird die Maske ohne weiteres Zutun durchs Video fortgeführt. Gemessen gegen dieselben Referenzmasken wie in Abschnitt 7/12 (SAM-Masken aus Boxen — das begünstigt
SAM-artige Verfahren, siehe Vorbehalt).

| Video | Methode | IoU (Median) | Mittelpunktfehler (% Objektgröße) | Winkelfehler (Median) |
|---|---|---|---|---|
| Karosse, 6 Testbilder | unser Detektor (5 Anlernbilder) | 0,79 | 1,9 | 0,5° |
| Karosse, 6 Testbilder | EdgeTAM (1 Box) | **0,94** | **0,8** | 1,0° |
| Handy, 8 Testbilder | unser Detektor (5 Anlernbilder) | 0,81 | 3,7 | 3,6° |
| Handy, 8 Testbilder | EdgeTAM (1 Box, 475 Bilder fortgeführt) | **0,97** | **0,5** | 0,7° |

EdgeTAM hielt das Handy auch hochkant und fast kantenparallel in der Hand (Bilder 500–559) mit sauberer Maske.

**Schwächen (gemessen):**
- **Abwesenheit:** Nachdem das Handy das Bild verließ (ab Bild ~570), ging die Maske richtig auf 0, zeigte aber in 25 von 430 Bildern (5,8 %) eine falsche Maske (einmal 12,6 % der Bildfläche).
  Es gibt keine Wiedererkennung: kehrt das Objekt zurück, braucht EdgeTAM einen neuen Prompt.
- **Tempo:** PyTorch auf Apple-GPU (MPS): ca. 2,3 Bilder/s (430 ms je Bild) — nicht echtzeitfähig. Das CoreML-Export (nur Bildencoder, Prompt-Encoder, Masken-Decoder) ergab für den
  **Bildencoder allein 23 ms** (Neural Engine; 93 ms Apple-GPU, 200 ms CPU). Die Memory-Module (Gedächtnis-Aufmerksamkeit/-Encoder), die das Fortführen erst ermöglichen, sind in dem Export
  **nicht enthalten**. Die Gesamtzeit je Bild ist daher **nicht gemessen**; die Quelle nennt 16 fps auf einem iPhone 15 Pro Max.
- **Plattform:** Der Weg über CoreML gilt nur für Mac/iOS; ein ONNX-Export ist nicht dokumentiert (Windows offen).
- **Aufwand:** Export klappte erst mit PyTorch 2.7 (mit 2.14 brach er ab), Repo 348 MB, mehrere Hilfspakete.

**Vorbehalt:** Referenzmasken stammen aus SAM, EdgeTAM ist SAM-artig; der Abstand zu unserem Detektor ist vermutlich teilweise darauf zurückzuführen. Dass die Masken optisch sauber sind, zeigte aber auch die Sichtprüfung
(Handy, Bilder 160–555). Nur 14 Testbilder, ein Prompt je Video.

**Mögliche Architektur (nicht gebaut):** Unser Detektor erkennt und liefert die Box (ersetzt die Handbox), EdgeTAM führt die präzise Maske mit, unser Detektor prüft alle N Bilder, ob die Maske noch das gelernte Objekt ist
(fängt Fehlmasken ab) und gibt bei Verlust eine neue Box. Voraussetzung: Echtzeit-Gesamtzeit auf CoreML belegen.

### 14a. EdgeTAM auf CoreML: Gesamtzeit je Bild (03.10.2026)

Das offizielle CoreML-Export enthält nur Bildencoder, Prompt-Encoder und Masken-Decoder. Die Gedächtnis-Aufmerksamkeit (memory attention) habe ich selbst exportiert
(Rotations-Positionscodierung von komplexen auf reelle Zahlen umgeschrieben, da CoreML keine komplexen Zahlen kennt; Ergebnis gegen PyTorch geprüft: Kosinus-Ähnlichkeit 1,00000,
größte Abweichung 0,005 bei Mittelwert 0,415). Messung auf diesem Mac (Median, Neural Engine wo möglich, feste Formen wie im eingeschwungenen Zustand: 4096 Anfrage-Tokens, 3640 Gedächtnis-Tokens):

| Baustein | CoreML (Neural Engine) | PyTorch Apple-GPU (MPS) |
|---|---|---|
| Bildencoder (1024²) | 23 ms | 187 ms |
| Gedächtnis-Aufmerksamkeit | **72 ms** | 92 ms |
| Masken-Decoder | 18 ms | 72 ms (inkl. Zusatzköpfe) |
| Gedächtnis-Encoder | nicht exportiert | 25 ms |
| Rest (Mask hochskalieren, Löcher füllen, Zusammenbau) | nicht gemessen | ca. 80 ms |

**Untergrenze:** 23 + 72 + 18 = **113 ms je Bild, also höchstens ca. 9 Bilder/s**, noch ohne Gedächtnis-Encoder, Perceiver und Nachbearbeitung. Realistisch eher 6–8 Bilder/s. Zum Vergleich unser Detektor:
11–30 ms je Erkennung. Die Gedächtnis-Aufmerksamkeit (zwei Schichten, volle 64×64-Anfrage gegen 3640 Gedächtnis-Tokens) ist der Engpass und lässt sich auf der Neural Engine kaum schneller machen als in PyTorch.

**Folgerung:** EdgeTAM ist auf diesem Gerät **nicht echtzeitfähig im Sinne von 25–30 Bildern/s** und taugt, wenn überhaupt, als *langsamer Verfeinerer* (wenige Hz) neben dem schnellen Detektor, nicht als Haupt-Tracker.
Die bessere Maskenqualität (IoU 0,94–0,97) bleibt ein Argument für einen Genau-Modus bei niedriger Rate; das müsste gegen den Aufwand (PyTorch 2.7, eigener Export, nur Mac) abgewogen werden.

## 15. Große Objekte: Merkmals-Anker statt Silhouette (03.10.2026, Prototyp, nicht eingebaut)

**Beobachtung (Nutzer):** Bei großen Objekten (Karosse) schwankt die Pose der ganzen Maske stark — Schwerpunkt und Hauptachse einer Maske folgen jedem Fehler an ihrem Rand.

**Idee:** Die Maske liefert nur die grobe Lage (Suchgebiet). Die Pose kommt aus automatisch gefundenen Bildmerkmalen (SIFT) innerhalb der Maske: Referenzmerkmale aus den Anlernbildern werden im aktuellen Bild
wiedergefunden, eine Ähnlichkeitstransformation (RANSAC) liefert Verschiebung und Drehung; abgebildet wird ein **fester Ankerpunkt am Objekt** (optional vom Nutzer als „dedizierte Merkmalsregion" gewählt,
sonst z. B. Maskenschwerpunkt des Referenzbildes). Referenzen werden über Nachbaransichten verkettet, damit der Anker in allen Ansichten derselbe physische Punkt bleibt.

**Messung** (Karosse-Video, 344 Bilder, 5 Anlernansichten, Anker = Tür; Rauschmaß = Betrag der zweiten Differenz der Position je Bild, Median / 90. Perzentil; die Bewegung ist glatt, Sprünge sind Rauschen):

| Verfahren | Rauschen x (px) | Rauschen y (px) | Abdeckung |
|---|---|---|---|
| Maskenschwerpunkt (jetzt) | 13,9 / p90 64 | 5,8 / p90 24 | 98 % |
| Merkmals-Anker (SIFT, Ähnlichkeitstransformation) | **0,49** / p90 4,8 | **0,15** / p90 2,0 | 100 % |

Etwa 28-mal ruhiger im Median. Median 58 stimmige Merkmale je Bild. Aufwand: ca. 79 ms je Bild (SIFT im Suchgebiet + Abgleich mit 5 Referenzen, nur CPU, ungetuned).

**Grenzen/Vorbehalte:** (1) Gemessen ist nur **Glätte**, nicht die absolute Genauigkeit des Ankers; die Verkettung der Referenzen kann einen Versatz tragen (erste Verkettung nur 10 stimmige Merkmale). Dafür bräuchte es einen Marker
oder gemessene Soll-Positionen. (2) Die Karosse ist stark texturiert; glatte, einfarbige Teile liefern kaum Merkmale — dort bleibt die Maske die einzige Quelle. (3) Mit großer Blickwinkeländerung scheitert der Abgleich einzelner Referenzen (Referenz 279 ließ sich nicht verketten).
(4) Szenenschnitte im Stock-Video erzeugen vereinzelte große Sprünge (Bild 255) — sie wirken auf beide Verfahren.

**Entwurf, falls eingebaut:** Modell speichert je Anlernfoto Schlüsselpunkte + Deskriptoren (innerhalb der Maske, bei gewählter Region nur dort) und den Anker; Umschalter „Merkmals-Tracking" (Aus / Auto / An; Auto = an, wenn das Objekt > ca. 20 % der Bildbreite einnimmt);
bei fehlgeschlagenem Abgleich Rückfall auf die Maske, **in derselben Anker-Konvention** (Anker aus Maskenlage und -winkel abgeleitet), damit die Pose nicht zwischen zwei Bezugspunkten springt.

**Umgesetzt (03.10.2026):** `src/ctrack/features.py`, Anlern-Dialog mit optionaler Merkmalsregion, Schalter „Merkmals-Tracking“ (Aus/Automatisch/An), siehe ADR-014. Gemessen mit dem fertigen Detektor (nicht nur Prototyp):
Rauschen x 8,64 → **0,46 px**, y 4,38 → **0,19 px** (Median), p90 x 25,7 → 2,4 px; Zeit je Bild 42 → 61 ms. Die Abdeckung schwankte zwischen zwei Läufen leicht (96 % / 94 %), vermutlich Lastschwankung des Rechners.

**Nachtrag (03.10.2026, nach Nutzerhinweis „wie beim Türgriff“):** Der Nutzer wollte die Merkmalsposition auf **jedem** Foto markieren und im Tracking nur in einem kleinen Bereich relativ zur gefundenen Karosse suchen. Gemessen:
- Das kleine Merkmalsgebiet allein (14 % der Objektgröße um die Tür) hat nur **0–8 SIFT-Schlüsselpunkte**; zwischen Fotos stimmen auch bei 30 %-Gebieten nur 4–7 überein. Kein verlässlicher Abgleich.
- Dichter Schablonenabgleich (NCC) des Gebiets im kleinen Fenster: ca. 15 ms, aber 7 px Rauschen.
- SIFT in der Nachbarschaft (30 %): nur 265 von 344 Bildern abgeglichen, Ausreißer.
- **Umgesetzt:** Markierung je Foto legt Anker und Anzeige fest; abgeglichen wird das ganze Objekt (im Begrenzungsrahmen der Maske), mit Plausibilitätsprüfung gegen die aus der Maske erwartete Stelle, ORB (schnell) oder SIFT (genau),
  Extraktion parallel zum Detektor. Rauschen Maske 39 / 22 px → 0,5 / 0,25 px; Anzeige von Objekt **und** Merkmal. Details ADR-014.
- Zeit: Merkmalsextraktion kostet bei SIFT ca. 22–25 ms (unabhängig von der Merkmalszahl; Abgleich nur ca. 1,4 ms), bei SIFT mit auf 300 px normierter Objektgröße ca. 13–15 ms, bei ORB ca. 4 ms. Schneller und gleich genau wird es vor allem durch ORB und die Parallelität.
