# High-resolution fine-tune: the dataset's objects are tiny, so this is the
# highest-leverage single change available.
#
# glassbottle has a median box area of 0.76% of the image; at 640 px many bottles
# are only ~20 px across.  Training/evaluating at 1024 px is the standard fix.
#
# This is a FINE-TUNE, not a fresh 50-epoch run: it starts from an already
# trained checkpoint and uses a lower LR for a short budget, so it must be
# reported as "high-resolution fine-tune", never compared head-to-head with the
# 50-epoch A/B/C runs at 640.
#
#   powershell -ExecutionPolicy Bypass -File envutils\train_d1024.ps1 -Init <best.pt> [-Epochs 25] [-Batch 6]
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

param(
    [Parameter(Mandatory = $true)][string]$Init,
    [int]$Epochs = 25,
    [int]$Batch = 6,
    [int]$Imgsz = 1024,
    [string]$Name = 'D_ft1024'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'

$yolo   = Join-Path $Root '.venv\Scripts\yolo.exe'
$py     = Join-Path $Root '.venv\Scripts\python.exe'
$data   = Join-Path $Root 'data\yolo\data.yaml'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

if (-not (Test-Path $Init)) { Write-Output "init weights not found: $Init"; exit 1 }
Write-Output ("[{0:HH:mm:ss}] fine-tune {1} at {2}px, batch {3}, {4} epochs" -f (Get-Date), $Init, $Imgsz, $Batch, $Epochs)

$log = Join-Path $logDir "$Name-train.log"
& $yolo detect train `
    model=$Init data=$data `
    epochs=$Epochs imgsz=$Imgsz batch=$Batch `
    device=0 workers=8 seed=0 `
    optimizer=AdamW lr0=0.0005 lrf=0.01 weight_decay=0.0005 cos_lr=True `
    warmup_epochs=1 `
    amp=True cache=False plots=True `
    hsv_h=0.0 hsv_s=0.0 hsv_v=0.15 `
    translate=0.1 scale=0.3 fliplr=0.5 flipud=0.0 `
    mosaic=0.2 mixup=0.0 copy_paste=0.0 close_mosaic=5 `
    project=$proj name=$Name exist_ok=True 2>&1 | Tee-Object -FilePath $log | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Output ("[{0:HH:mm:ss}] TRAINING FAILED (exit {1}), see {2}" -f (Get-Date), $LASTEXITCODE, $log)
    Get-Content $log -Tail 25
    exit 1
}
Write-Output ("[{0:HH:mm:ss}] training done, evaluating at {1}px" -f (Get-Date), $Imgsz)

$best = Join-Path (Join-Path (Join-Path $proj $Name) 'weights') 'best.pt'
& $py (Join-Path $Root 'src\evaluate.py') --weights $best --tag $Name --imgsz $Imgsz --batch $Batch `
    --notes "high-res fine-tune at ${Imgsz}px from $Init" 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$Name-eval.log") | Out-Null

Write-Output ("[{0:HH:mm:ss}] D DONE" -f (Get-Date))
