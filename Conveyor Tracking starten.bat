@echo off
rem Doppelklick startet die Oberflaeche und oeffnet den Browser (Windows).
cd /d "%~dp0"
if exist .venv\Scripts\python.exe goto :run

echo Erstmalige Einrichtung dauert einige Minuten ...
set "PY="
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -3.11"
if defined PY goto :setup
py -3.12 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -3.12"
if defined PY goto :setup
echo Python 3.11 oder 3.12 wurde nicht gefunden (MediaPipe unterstuetzt kein neueres Python).
echo Installieren: https://www.python.org/downloads/windows/  (Haken bei "Add python.exe to PATH" setzen)
pause
exit /b 1

:setup
%PY% -m venv .venv || goto :fail
.venv\Scripts\python.exe -m pip install -q --upgrade pip || goto :fail
.venv\Scripts\python.exe -m pip install -q -e . || goto :fail

:run
echo Starte Conveyor Tracking ...
.venv\Scripts\python.exe -m ctrack.gui
echo.
pause
exit /b 0

:fail
echo Einrichtung fehlgeschlagen.
pause
exit /b 1
