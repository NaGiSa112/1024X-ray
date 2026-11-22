@echo off
rem Put the project venv first on PATH for this cmd.exe session.
rem
rem   envutils\env.cmd
rem
rem Not required: E:\X光\.venv\Scripts\python.exe already works with no
rem environment variables, because the tempfile fix loads from the venv's
rem site-packages.  This only saves typing and keeps scratch files local.

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "VIRTUAL_ENV=%ROOT%\.venv"
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"
set "TMP=%ROOT%\.tmp"
set "TEMP=%TMP%"
set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "YOLO_CONFIG_DIR=%ROOT%\.ultralytics"

if not exist "%TMP%" mkdir "%TMP%"
if not exist "%YOLO_CONFIG_DIR%" mkdir "%YOLO_CONFIG_DIR%"

echo venv   : %VIRTUAL_ENV%
echo python : %VIRTUAL_ENV%\Scripts\python.exe
