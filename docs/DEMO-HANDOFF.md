# Demo-Handoff — Visualizer und Live-Demo (G4)

Es gibt keine GUI im Produktsinn (PRD 3: "Produktionsreife (… GUI)" ist Nicht-Ziel). Dieses Dokument
beschreibt das Overlay-Fenster und den Demo-Ablauf für Engineering/BD.

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
