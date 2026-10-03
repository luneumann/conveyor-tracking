# Demo-Handoff — Oberfläche und Live-Demo (G4)

## Web-Oberfläche (Standard)

Start per Doppelklick auf `ctrack starten.command`. Links Live-Bild mit Overlay und Kennzahlen,
rechts die Bedienung in der Reihenfolge des Ablaufs:

| Karte | Inhalt |
|---|---|
| 1 Quelle | Kamera (Auswahl 0–3) · Demo-Band · Aufnahme · **Starten / Stoppen** |
| 2 Erkennung | Bauteil (Galerie der Referenzbilder, Auswahl per Klick, × löscht, **Aus Livebild einlernen**, **Bild laden …**) · Hand |
| 3 Tracking | **Tracking starten** (wartet auf Objekt) / **zurücksetzen** · Schalter „Automatisch einlocken“ |
| Ausgabe & Aufnahme | UDP-Senden mit Adresse/Port und Paketzähler · Video aufzeichnen |
| Feineinstellungen | Bewegungsprofil (Band/Hand) · Vorhersage-Horizont · Überbrückung bei Verdeckung · Mindest-Übereinstimmung |
| Letzter Lauf | Zusammenfassung und CSV-Download |

**Zustandsanzeige** (oben rechts und im Video): grau *Suche Objekt* · grün *Verfolge* · orange *Verdeckt –
Vorhersage* · rot *Verloren* (roter Kreis = Suchbereich für die Wiederaufnahme, wächst mit der Zeit).

**Kennzahlen-Kacheln** färben sich grün/rot gegen die PRD-Ziele: Bildrate ≥ 25 fps, Latenz p95 < 100 ms,
Vorhersagefehler p95 < 2 % der Bildbreite.

**Einlernen:** Bild friert ein → Rechteck ziehen (Mittelpunkt-Kreuz = Bezugspunkt, roter Strich = Richtung
θ = 0) → Name → *Speichern*. „Maske automatisch“ lässt den Hintergrund im Rechteck außen vor. Das neue
Referenzbild ist sofort aktiv; Enter speichert, Esc bricht ab.

Die folgenden Abschnitte beschreiben das OpenCV-Fenster der Kommandozeile (`python -m ctrack`).

---

## OpenCV-Fenster (CLI)

Die Web-Oberfläche ist ein Bedienpanel für den Prototyp, keine produktionsreife GUI (PRD 3, Nicht-Ziel).
Dieses Dokument beschreibt außerdem das Overlay-Fenster der Kommandozeile und den Demo-Ablauf für Engineering/BD.

## Fenster `ctrack`

```
┌────────────────────────────────────────────────────────────┐
│ TRACKING                       ← Zustand, farbcodiert        │
│ fps 30.1  latency p50 41 / p95 58 ms                        │
│ pred err @100ms p50 3.1 / p95 9.8 px (0.77% width), …       │
│ v (212, -4) px/s  omega 0.02 rad/s                          │
│ [L] lock  [R] reset  [Q] quit                               │
├────────────────────────────────────────────────────────────┤
│                 ·  ·  · (Hand-Landmarks, grau)              │
│             ●━━━━━━▶   aktuelle Pose (x rot, y grün)         │
│                  ┆ ╌ ╌ ╌ ▷   prädizierte Pose t+horizon (cyan, gestrichelt) │
└────────────────────────────────────────────────────────────┘
```

| Element | Wann sichtbar | Darstellung |
|---|---|---|
| Landmarks | Detektion vorhanden (nur Hand) | graue Punkte |
| Detektierte Pose | SEARCHING mit Detektion | graues Achsenkreuz — zeigt, was `L` einlocken würde |
| Aktuelle Pose | TRACKING, COASTING | Achsenkreuz durchgezogen, x-Pfeil rot, y grün, dunkle Outline |
| Prädizierte Pose | TRACKING, COASTING | Achsenkreuz gestrichelt cyan, `horizon_ms` voraus |
| Re-Acquire-Radius | LOST | roter Kreis um die Referenzposition |
| HUD | immer | halbtransparente Box oben links |

### Zustandsfarben

| Zustand | Farbe | Bedeutung für den Zuschauer |
|---|---|---|
| SEARCHING | grau | Nichts eingelockt |
| TRACKING | grün | Messung + Filter aktiv |
| COASTING | orange | Kurz verdeckt — Achsenkreuz läuft per Prädiktion weiter |
| LOST | rot | Verloren — Hand in den roten Kreis zurückführen |

### Tasten

| Taste | Wirkung |
|---|---|
| `L` | Lock auf nächste gültige Detektion |
| `R` | Reset → SEARCHING |
| `Q` / `Esc` | Beenden (CSV wird geschrieben) |

Das Fenster muss Fokus haben, damit Tasten ankommen.

## Demo-Ablauf (≈ 5 min)

1. **Ohne Hardware-Risiko starten:** `python -m ctrack -c config/synthetic.yaml --auto-lock`
   — erklärt Pipeline, Zustände und die gestrichelte Prädiktion an einem perfekten "Band".
2. **Receiver daneben:** `python tools/receiver.py` in zweitem Terminal — "das sieht der Roboter".
3. **Live:** `python -m ctrack`, Hand ins Bild, `L` drücken.
4. **Gleichmäßige Bewegung:** Hand langsam quer durchs Bild — gestricheltes Kreuz läuft voraus.
5. **Coasting:** Hand kurz mit der anderen Hand verdecken → orange, Kreuz läuft weiter.
6. **Lost + Re-Acquire:** Hand > 0,3 s aus dem Bild → rot, Kreis erscheint; Hand in den Kreis →
   grün, ohne `L`.
7. **Kennzahlen:** `Q`, dann `python tools/analyze.py logs/run.csv` → Plots zeigen.

## Kernaussagen für BD

- Ein Bauteil einmal erfassen, danach kontinuierliche Pose **mit Geschwindigkeit** → Roboter
  rechnet auf seinen Ausführungszeitpunkt hoch (kein Encoder nötig).
- Latenz und Prädiktionsfehler werden gemessen, nicht geschätzt.
- Hand ist Platzhalter: Kamera und Detektor sind per Config tauschbar (IDS + ArUco/Bauteil in Stufe 2).
