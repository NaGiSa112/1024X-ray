# Run the remaining experiment stages back to back, without leaving the GPU idle.
#
#   1. wait for the A run to reach 50 epochs, then evaluate it
#   2. sanity-check the EDS-1 domain weights on 5% of the data (1 epoch)
#   3. train + evaluate B (EDS-1 initialisation), then C (EDS-1 + stronger aug)
#
# Keep this file ASCII-only: Windows PowerShell 5.1 reads a .ps1 without a BOM
# using the ANSI code page, so Chinese literals get mangled here (and can even
# swallow a closing quote and break parsing).
#
# NOTE ON COMPARABILITY: the EDS-1 weight has 43.64M parameters, i.e. it is a
# YOLOv8l, while A is YOLOv8m (25.90M).  The Hugging Face repo publishes one
# size per dataset, so there is no v8m domain weight.  A vs B therefore differs
# in BOTH model size and initialisation; B vs C (both v8l, both EDS-1, only the
# augmentation differs) is the clean single-variable comparison.
#
# Training failures stop the chain (later stages depend on the same init); an
# evaluation failure is logged and skipped, because it does not invalidate the
# trained weights.
#
#   powershell -ExecutionPolicy Bypass -File envutils\train_chain.ps1

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'

$py     = Join-Path $Root '.venv\Scripts\python.exe'
$yolo   = Join-Path $Root '.venv\Scripts\yolo.exe'
$data   = Join-Path $Root 'data\yolo\data.yaml'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Stamp([string]$msg) {
    Write-Output ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg)
}

function Wait-ForEpochs([string]$csvPath, [int]$target, [int]$timeoutMin) {
    $deadline = (Get-Date).AddMinutes($timeoutMin)
    while ((Get-Date) -lt $deadline) {
        try {
            if ((Import-Csv $csvPath).Count -ge $target) { return $true }
        } catch { }
        Start-Sleep -Seconds 20
    }
    return $false
}

function Invoke-Stage([string]$tag, [string]$notes, [string[]]$trainArgs) {
    Stamp "=== ${tag}: training ==="
    $out = Join-Path $logDir "$tag-train.log"
    & $yolo @trainArgs 2>&1 | Tee-Object -FilePath $out | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Stamp "${tag} TRAINING FAILED (exit $LASTEXITCODE), see $out"
        Get-Content $out -Tail 25
        return $false
    }
    Stamp "${tag} training done"

    Stamp "=== ${tag}: evaluation ==="
    $best = Join-Path (Join-Path (Join-Path $proj $tag) 'weights') 'best.pt'
    & $py (Join-Path $Root 'src\evaluate.py') --weights $best --tag $tag --notes $notes 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$tag-eval.log") | Out-Null
    if ($LASTEXITCODE -ne 0) { Stamp "${tag} evaluation failed (weights are still fine)" }
    return $true
}

# ---------------------------------------------------------------- 1. wait for A
Stamp "waiting for A_coco640 to reach 50 epochs"
$csvA = Join-Path (Join-Path $proj 'A_coco640') 'results.csv'
if (-not (Wait-ForEpochs $csvA 50 90)) {
    Stamp "A did not reach 50 epochs in time; stopping"
    exit 1
}
Stamp "=== A: evaluation ==="
& $py (Join-Path $Root 'src\evaluate.py') `
    --weights (Join-Path (Join-Path (Join-Path $proj 'A_coco640') 'weights') 'best.pt') `
    --tag A --notes 'yolov8m COCO init, 640/50ep' 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir 'A-eval.log') | Out-Null

# ------------------------------------------------------- 2. EDS-1 sanity check
Stamp "=== B: EDS-1 sanity check (1 epoch, 5% data) ==="
$sanity = Join-Path $logDir 'B-sanity.log'
& $yolo detect train `
    model='weights/yolov8_eds1.pt' data=$data `
    epochs=1 fraction=0.05 imgsz=640 batch=8 `
    device=0 workers=4 seed=0 amp=True cache=False plots=False `
    hsv_h=0.0 hsv_s=0.0 hsv_v=0.15 mosaic=0.2 mixup=0.0 copy_paste=0.0 close_mosaic=0 `
    project=$proj name='B_sanity' exist_ok=True 2>&1 | Tee-Object -FilePath $sanity | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stamp "EDS-1 sanity check FAILED, stopping before B/C"
    Get-Content $sanity -Tail 30
    exit 1
}
Stamp "EDS-1 sanity check passed"

# ------------------------------------------------------------------- 3. B, C
$common = @(
    'detect', 'train', ('data=' + $data),
    'epochs=50', 'patience=0', 'imgsz=640', 'batch=16',
    'device=0', 'workers=8', 'seed=0', 'deterministic=True',
    'optimizer=AdamW', 'lr0=0.001', 'weight_decay=0.0005', 'cos_lr=True',
    'amp=True', 'cache=False', 'plots=True',
    'hsv_h=0.0', 'hsv_s=0.0', 'hsv_v=0.15',
    'degrees=0.0', 'translate=0.1', 'scale=0.3', 'fliplr=0.5', 'flipud=0.0',
    ('project=' + $proj)
)

$bArgs = @('model=weights/yolov8_eds1.pt') + $common + @(
    'mosaic=0.2', 'mixup=0.0', 'copy_paste=0.0', 'close_mosaic=10', 'name=B_eds640'
)
if (-not (Invoke-Stage -tag 'B_eds640' -notes 'yolov8l EDS-1 init, same args as A' -trainArgs $bArgs)) { exit 1 }

$cArgs = @('model=weights/yolov8_eds1.pt') + $common + @(
    'mosaic=0.5', 'mixup=0.05', 'copy_paste=0.0', 'close_mosaic=10', 'flipud=0.5',
    'name=C_eds_aug640'
)
if (-not (Invoke-Stage -tag 'C_eds_aug640' -notes 'yolov8l EDS-1 init + mosaic0.5/mixup0.05/flipud0.5' -trainArgs $cArgs)) { exit 1 }

Stamp "ALL STAGES DONE"
