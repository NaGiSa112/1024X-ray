# Optional convenience: put the project venv first on PATH.
#
#   . .\envutils\activate.ps1
#
# You do NOT strictly need this -- E:\X光\.venv\Scripts\python.exe already works
# with no environment variables at all, because the tempfile fix is loaded from
# the venv's site-packages.  This just saves typing `python` and keeps TMP and
# ultralytics' config inside the project.
#
# If PowerShell refuses to run the script ("not digitally signed"), either
#   pwsh -ExecutionPolicy Bypass -Command ". .\envutils\activate.ps1; python src\check_env.py"
# or use .\envutils\env.cmd from cmd.exe, or just call the venv python directly:
#   E:\X光\.venv\Scripts\python.exe src\check_env.py

$Root = Split-Path -Parent $PSScriptRoot

$env:VIRTUAL_ENV = Join-Path $Root '.venv'
$env:PATH        = "$(Join-Path $Root '.venv\Scripts');$env:PATH"
$env:TMP         = Join-Path $Root '.tmp'
$env:TEMP        = $env:TMP
$env:PYTHONUTF8  = '1'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
# keep ultralytics' settings files inside the project
$env:YOLO_CONFIG_DIR = Join-Path $Root '.ultralytics'

foreach ($d in @($env:TMP, (Join-Path $Root '.cache\pip'), $env:YOLO_CONFIG_DIR)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

Write-Host "venv   : $env:VIRTUAL_ENV"
Write-Host "python : $(Join-Path $Root '.venv\Scripts\python.exe')"
