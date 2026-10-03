#!/bin/bash
# Doppelklick startet die Oberfläche und öffnet den Browser.
cd "$(dirname "$0")" || exit 1

if [ ! -x .venv/bin/python ]; then
  echo "Erstmalige Einrichtung (dauert einige Minuten) ..."
  PY=$(command -v python3.11 || command -v python3.12)
  if [ -z "$PY" ]; then
    echo "Python 3.11 oder 3.12 wurde nicht gefunden (MediaPipe unterstützt kein neueres Python)."
    echo "Installieren, z. B. mit:  brew install python@3.11"
    read -r -p "Enter zum Beenden ..." _; exit 1
  fi
  "$PY" -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -e . || {
    echo "Einrichtung fehlgeschlagen."; read -r -p "Enter zum Beenden ..." _; exit 1; }
fi

echo "Starte Lockstep ..."
.venv/bin/python -m ctrack.gui
echo
read -r -p "Beendet. Enter zum Schließen ..." _
