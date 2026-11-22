# Overnight long run: the last two levers with real measured support.
#
# F: full 50-epoch training at 1024 px from the EDS-1 domain weights, with a
#    wider scale range.  D was only a 24-epoch *fine-tune* at 1024, so this is
#    the first properly trained 1024 px model; scale 0.3 -> 0.4 targets the
#    bimodal object sizes found in the audit.
# G: fine-tune F at 1280 px for 10 epochs -- the same "train at the size you are
#    limited by" logic that produced the +3.5 from 640 -> 1024.
#
# Both are evaluated on the leak-free 512-image validation split
# (data/yolo_cleanval), not the leaky 555-image one, and submissions are exported
# with TTA at each model's native resolution.
#
#   powershell -ExecutionPolicy Bypass -File envutils\overnight.ps1
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'

$yolo   = Join-Path $Root '.venv\Scripts\yolo.exe'
$py     = Join-Path $Root '.venv\Scripts\python.exe'
$data   = Join-Path $Root 'data\yolo\data.yaml'
$clean  = Join-Path $Root 'data\yolo_cleanval\data.yaml'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Stamp([string]$m) { Write-Output ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $m) }

function Train([string]$name, [string]$init, [int]$epochs, [int]$imgsz, [int]$batch, [double]$lr, [string]$extra) {
    Stamp "=== $name : train from $init, ${imgsz}px, batch $batch, $epochs epochs ==="
    $log = Join-Path $logDir "$name-train.log"
    $args = @(
        'detect', 'train', ('model=' + $init), ('data=' + $data),
        ('epochs=' + $epochs), ('imgsz=' + $imgsz), ('batch=' + $batch),
        'device=0', 'workers=8', 'seed=0', 'deterministic=True',
        'optimizer=AdamW', ('lr0=' + $lr), 'lrf=0.01', 'weight_decay=0.0005', 'cos_lr=True',
        'amp=True', 'cache=False', 'plots=True',
        'hsv_h=0.0', 'hsv_s=0.0', 'hsv_v=0.15',
        'translate=0.1', 'scale=0.4', 'fliplr=0.5', 'flipud=0.5',
        'mosaic=0.5', 'mixup=0.05', 'copy_paste=0.0', 'close_mosaic=10',
        ('project=' + $proj), ('name=' + $name)
    )
    if ($extra) { $args += $extra.Split(' ') }
    & $yolo @args 2>&1 | Tee-Object -FilePath $log | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Stamp "$name TRAINING FAILED (exit $LASTEXITCODE); tail:"
        Get-Content $log -Tail 25
        return $false
    }
    Stamp "$name training done"
    return $true
}

function Eval([string]$name, [int]$imgsz, [int]$batch) {
    $best = Join-Path (Join-Path (Join-Path $proj $name) 'weights') 'best.pt'
    if (-not (Test-Path $best)) { Stamp "${name} has no best.pt"; return $null }
    Stamp "=== $name : evaluated on clean val (512) at ${imgsz}px ==="
    & $py (Join-Path $Root 'src\evaluate.py') --weights $best --data $clean `
        --tag "${name}_clean" --imgsz $imgsz --batch $batch --skip-cm 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$name-eval.log") | Out-Null
    $json = Join-Path (Join-Path $proj $name) "metrics_${name}_clean.json"
    if (Test-Path $json) {
        $j = Get-Content $json -Raw -Encoding utf8 | ConvertFrom-Json
        Stamp ("$name clean mAP50 = {0}" -f $j.mAP50)
        return [double]$j.mAP50
    }
    return $null
}

function Export([string]$name, [string]$suffix, [int]$imgsz, [int]$batch) {
    $best = Join-Path (Join-Path (Join-Path $proj $name) 'weights') 'best.pt'
    Stamp "=== export submission $suffix from $name (${imgsz}px + TTA) ==="
    & $py (Join-Path $Root 'src\export_submission.py') --weights $best `
        --imgsz $imgsz --augment --conf 0.01 --iou 0.7 --max-det 300 --batch $batch `
        --out (Join-Path $Root "results_test2_submit_$suffix.json") 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$suffix-export.log") | Out-Null
    if ($LASTEXITCODE -ne 0) { Stamp "$suffix export FAILED"; return }
    & $py (Join-Path $Root 'src\validate_submission.py') `
        (Join-Path $Root "results_test2_submit_$suffix.json") 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$suffix-validate.log") | Out-Null
    if ($LASTEXITCODE -eq 0) { Stamp "$suffix submission valid" } else { Stamp "$suffix submission INVALID" }
}

# ------------------------------------------------------------------ F
$okF = Train -name 'F_full1024' -init (Join-Path $Root 'weights\yolov8_eds1.pt') `
    -epochs 50 -imgsz 1024 -batch 6 -lr 0.0008 -extra ''
if (-not $okF) { Stamp 'F failed, stopping'; exit 1 }
$mapF = Eval -name 'F_full1024' -imgsz 1024 -batch 6
Export -name 'F_full1024' -suffix 'F' -imgsz 1024 -batch 4

# ------------------------------------------------------------------ G
$bestF = Join-Path (Join-Path (Join-Path $proj 'F_full1024') 'weights') 'best.pt'
$okG = Train -name 'G_ft1280' -init $bestF -epochs 10 -imgsz 1280 -batch 3 -lr 0.0004 -extra ''
if ($okG) {
    $mapG = Eval -name 'G_ft1280' -imgsz 1280 -batch 3
    Export -name 'G_ft1280' -suffix 'G' -imgsz 1280 -batch 2
}

Stamp ("DONE. clean-val mAP50: F={0}  (D reference 0.7406 full-run / 0.7645 with TTA)" -f $mapF)
