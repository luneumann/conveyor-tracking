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
