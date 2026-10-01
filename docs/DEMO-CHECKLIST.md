# Demo- und Mess-Checkliste

Ersetzt die Deploy-Checkliste einer Web-App: der Prototyp läuft lokal, "Deploy" = Demo-Termin
bzw. Messkampagne.

## Einmalig

- [ ] venv + Abhängigkeiten installiert (`SETUP.md` §1)
- [ ] MediaPipe-Modell geladen: `models/hand_landmarker.task` existiert
- [ ] Kamera-Berechtigung für die Terminal-App erteilt (macOS)
- [ ] `exposure_offset_ms` mit `tools/latency_probe.py` gemessen und in `config/default.yaml` eingetragen

## Vor jeder Demo / Messung

- [ ] `pytest` grün
- [ ] Synthetik läuft: `python -m ctrack -c config/synthetic.yaml --auto-lock`
- [ ] Webcam läuft, fps im HUD ≥ 25 (sonst: andere Apps mit Kamera schließen, Licht verbessern,
      `-s camera.width=960 -s camera.height=540`)
- [ ] `tools/receiver.py` zeigt 0 % Verlust
- [ ] Gleichmäßiges, helles Licht, ruhiger Hintergrund (MediaPipe-Detektionsrate)
- [ ] Laptop am Netzteil (Energiesparmodus drosselt fps)

## Messprotokoll für die Erfolgsmetriken (PRD 7)

| Metrik | Ablauf | Ablesen |
|---|---|---|
| Latenz p95 | 60 s normal tracken | `analyze.py` → latency p95 |
| Prädiktionsfehler p95 | Hand 10× gleichmäßig quer durchs Bild | `analyze.py` → pred err p95 % |
| Re-Acquire | 10× Hand raus (≥ 0,5 s) und zurück | CSV: Zeit LOST → TRACKING |
| fps | während aller Läufe | `analyze.py` → fps |
| Paketverlust | während aller Läufe | `receiver.py`-Zusammenfassung |

Jeden Lauf mit eigenem CSV: `-s metrics.csv=logs/2026-10-xx_gleichmaessig.csv`.
