# Midnight long run: progressive resizing, 1024 -> 1280 -> 1536.
#
# Why progressive resizing instead of a bigger backbone:
#   the only measured, unambiguous lever left is "train at the resolution you are
#   limited by" (640 -> 1024 was worth +3.5 mAP50).  Going to v8x would mean
#   starting from COCO again, because no domain-pretrained v8x exists -- that
#   throws away the EDS-1 initialisation which was worth +4.5.  So we keep v8l +
#   EDS-1 and push the resolution instead.
#
#   F: v8l EDS-1 @1024, full 50 epochs (D was only a 24-epoch fine-tune), scale 0.4
#   G: F -> @1280, 12 epochs
#   H: G -> @1536, 8 epochs (skipped if we are already past the time guard)
#
# Every stage evaluates on the leak-free 512-image split and exports a TTA
# submission, so each stage leaves usable artefacts even if the next one fails.
#
#   powershell -ExecutionPolicy Bypass -File envutils\midnight.ps1
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

function Stamp([string]$m) { Write-Output ("[{0:MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) }

function Train([string]$name, [string]$init, [int]$epochs, [int]$imgsz, [int]$batch, [double]$lr) {
    Stamp "=== $name : from $init, ${imgsz}px, batch $batch, $epochs epochs, lr $lr ==="
    $log = Join-Path $logDir "$name-train.log"
    # Augmentation follows the 2023 same-competition winning write-ups: the 2nd
    # place team called augmentation "the main source of gains", and all three
    # podium teams used heavy geometric augmentation plus ensembling.  Colour is
    # still kept almost untouched (hsv_h/s = 0) because X-ray pseudo-colour
    # encodes material, but the geometric range is much wider than our earlier
    # runs: degrees/shear/perspective added, scale 0.3 -> 0.5, mosaic 0.5 -> 1.0,
    # translate 0.1 -> 0.2, and mosaic closed earlier so the last epochs see real
    # composite images.
    $a = @(
        'detect', 'train', ('model=' + $init), ('data=' + $data),
        ('epochs=' + $epochs), ('imgsz=' + $imgsz), ('batch=' + $batch),
        'device=0', 'workers=8', 'seed=0', 'deterministic=True',
        'optimizer=AdamW', ('lr0=' + $lr), 'lrf=0.01', 'weight_decay=0.0005', 'cos_lr=True',
        'amp=True', 'cache=False', 'plots=True',
        'hsv_h=0.0', 'hsv_s=0.0', 'hsv_v=0.2',
        'translate=0.2', 'scale=0.5', 'shear=2.0', 'perspective=0.0005',
        'degrees=10.0', 'fliplr=0.5', 'flipud=0.5',
        'mosaic=1.0', 'mixup=0.1', 'copy_paste=0.0', 'close_mosaic=15',
        ('project=' + $proj), ('name=' + $name)
    )
    & $yolo @a 2>&1 | Tee-Object -FilePath $log | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Stamp "${name} TRAINING FAILED (exit $LASTEXITCODE); tail:"
        Get-Content $log -Tail 25
        return $false
    }
    Stamp "${name} training done"
    return $true
}

function EvalAndExport([string]$name, [string]$suffix, [int]$imgsz, [int]$evalBatch, [int]$exportBatch) {
    $best = Join-Path (Join-Path (Join-Path $proj $name) 'weights') 'best.pt'
    if (-not (Test-Path $best)) { Stamp "${name} has no best.pt"; return $null }

    Stamp "=== $name : clean-val eval at ${imgsz}px ==="
    & $py (Join-Path $Root 'src\evaluate.py') --weights $best --data $clean `
        --tag "${name}_clean" --imgsz $imgsz --batch $evalBatch --skip-cm 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$name-eval.log") | Out-Null
    $json = Join-Path (Join-Path $proj $name) "metrics_${name}_clean.json"
    $map = $null
    if (Test-Path $json) {
        $j = Get-Content $json -Raw -Encoding utf8 | ConvertFrom-Json
        $map = [double]$j.mAP50
        Stamp ("{0} clean mAP50 = {1}" -f $name, $j.mAP50)
    }

    Stamp "=== $name : export submission $suffix (${imgsz}px + TTA) ==="
    $out = Join-Path $Root "results_test2_submit_$suffix.json"
    & $py (Join-Path $Root 'src\export_submission.py') --weights $best `
        --imgsz $imgsz --augment --conf 0.01 --iou 0.7 --max-det 300 --batch $exportBatch `
        --out $out 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$suffix-export.log") | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & $py (Join-Path $Root 'src\validate_submission.py') $out 2>&1 |
            Tee-Object -FilePath (Join-Path $logDir "$suffix-validate.log") | Out-Null
        if ($LASTEXITCODE -eq 0) { Stamp "$suffix submission valid" } else { Stamp "$suffix submission INVALID" }
    } else {
        Stamp "$suffix export FAILED"
    }
    return $map
}

Stamp "MIDNIGHT RUN START"
$results = @{}

# ------------------------------------------------ F: full 50 ep at 1024
$ok = Train -name 'F_full1024' -init (Join-Path $Root 'weights\yolov8_eds1.pt') `
    -epochs 50 -imgsz 1024 -batch 6 -lr 0.0008
if (-not $ok) { Stamp 'F failed, stopping'; exit 1 }
$results['F'] = EvalAndExport -name 'F_full1024' -suffix 'F' -imgsz 1024 -evalBatch 6 -exportBatch 4

# ------------------------------------------------ G: 1280 fine-tune
$bestF = Join-Path (Join-Path (Join-Path $proj 'F_full1024') 'weights') 'best.pt'
$ok = Train -name 'G_ft1280' -init $bestF -epochs 12 -imgsz 1280 -batch 3 -lr 0.0004
if ($ok) {
    $results['G'] = EvalAndExport -name 'G_ft1280' -suffix 'G' -imgsz 1280 -evalBatch 3 -exportBatch 2
} else {
    Stamp 'G failed, skipping H'; exit 1
}

# ------------------------------------------------ H: 1536 fine-tune (time guarded)
if ((Get-Date).Hour -ge 5) {
    Stamp 'past 05:00, skipping H to protect the morning window'
} else {
    $bestG = Join-Path (Join-Path (Join-Path $proj 'G_ft1280') 'weights') 'best.pt'
    $ok = Train -name 'H_ft1536' -init $bestG -epochs 8 -imgsz 1536 -batch 2 -lr 0.0003
    if ($ok) {
        $results['H'] = EvalAndExport -name 'H_ft1536' -suffix 'H' -imgsz 1536 -evalBatch 2 -exportBatch 1
    } else {
        Stamp 'H failed; F and G artefacts are still valid'
    }
}

Stamp "MIDNIGHT RUN DONE"
foreach ($k in $results.Keys) { Stamp ("  {0}: clean mAP50 = {1}" -f $k, $results[$k]) }

# -------------------------------------------- ensemble of everything we have
# All three podium teams in the 2023 edition of this competition used box-level
# ensembling (NMS / WBF / SoftNMS), so it is worth one submission slot even
# though our offline AP harness is not yet trustworthy enough to pre-select it.
# Each model keeps its own inference size, which matters because a 640 model run
# at 1280 loses several points.
$cands = @(
    @('C_eds_aug640', 640),
    @('D_ft1024', 1024),
    @('F_full1024', 1024),
    @('G_ft1280', 1280),
    @('H_ft1536', 1536)
)
$paths = @()
$sizes = @()
foreach ($c in $cands) {
    $p = Join-Path (Join-Path (Join-Path $proj $c[0]) 'weights') 'best.pt'
    if (Test-Path $p) { $paths += $p; $sizes += $c[1] }
}
if ($paths.Count -ge 2) {
    Stamp ("=== ensemble of {0} models: {1} ===" -f $paths.Count, ($sizes -join ','))
    & $py (Join-Path $Root 'src\fuse_run.py') --weights @paths --imgsz-per-model ($sizes -join ',') `
        --conf 0.01 --iou 0.7 --max-det 300 --batch 4 --tta `
        --out (Join-Path $Root 'results_test2_submit_ENS.json') 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir 'ENS-export.log') | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & $py (Join-Path $Root 'src\validate_submission.py') `
            (Join-Path $Root 'results_test2_submit_ENS.json') 2>&1 |
            Tee-Object -FilePath (Join-Path $logDir 'ENS-validate.log') | Out-Null
        if ($LASTEXITCODE -eq 0) { Stamp 'ENS submission valid' } else { Stamp 'ENS submission INVALID' }
    } else {
        Stamp 'ENS export FAILED'
    }
} else {
    Stamp 'not enough models for an ensemble'
}

Stamp "submit the highest of F/G/H first, then try ENS; online = clean mAP50 x 0.919 x 100"
