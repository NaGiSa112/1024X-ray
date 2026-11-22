# Night run: semi-supervised student, then fused export. Deliverable by ~05:30.
#
# Goal: beat the current 75.335 online.  Calibration is 0.915-0.917 online per
# local mAP50, so 78 needs ~0.851 local (currently 0.8215, so +2.9).
#
# Lever chosen: transductive self-training.  The ensemble's own confident
# predictions on the 2325 test images become labels for those images, which are
# then added to training.  2263 test images are added (7836 seed boxes, ~3.5 boxes
# per image, above the 2.5 boxes/image density of the human labels, so the usual
# "unlabelled objects become background" failure mode does not apply).
#
# Stage order is deliberate: produce a valid deliverable EARLY (stage B), then use
# any remaining clock on a second architecture.  If anything after stage B fails,
# the submission from stage B still exists.
#
# Everything is logged to runs/chain_logs/night2-*.log so the run can be audited
# or continued by hand.
#
#   powershell -ExecutionPolicy Bypass -File envutils\night2.ps1
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

param(
    [int]$S1Epochs = 40,
    [switch]$SkipStageA
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'
$env:YOLO_CONFIG_DIR = Join-Path $Root '.ultralytics'
$env:TMP = Join-Path $Root '.tmp'
$env:TEMP = $env:TMP

$py     = Join-Path $Root '.venv\Scripts\python.exe'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
$semi   = Join-Path $Root 'data\yolo_semi\data.yaml'
$val489 = Join-Path $Root 'data\yolo_night75b\data.yaml'
New-Item -ItemType Directory -Force -Path $logDir, $env:TMP | Out-Null

function Stamp([string]$m) { Write-Output ("[{0:MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) }
function Log([string]$m) { Stamp $m; Add-Content -Path (Join-Path $logDir 'night2.log') -Value ("[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) -Encoding utf8 }

# ---------------------------------------------------------------- stage A
$S1 = 'S1_semi_yolov8l'
if (-not $SkipStageA) {
    Log "STAGE A: training $S1 on semi-supervised data ($S1Epochs epochs, 1024, batch 6)"
    $ErrorActionPreference = 'Continue'
    & $py (Join-Path $Root 'src\train_map50.py') `
        --weights (Join-Path $Root 'weights\yolov8_eds1.pt') `
        --data $semi --name $S1 --project $proj `
        --epochs $S1Epochs --imgsz 1024 --batch 6 --lr0 0.0008 --workers 8 --seed 0 `
        --cls 0.625 --box 7.5 --scale 0.3 --mosaic 0.5 --mixup 0.05 --flipud 0.5 `
        --close-mosaic 10 --warmup-epochs 3 --save-period 10 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$S1-train.log") | Out-Null
    $rc = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($rc -ne 0) {
        Log "STAGE A FAILED (exit $rc); tail:"
        Get-Content (Join-Path $logDir "$S1-train.log") -Tail 25 | ForEach-Object { Log "  $_" }
        exit 1
    }
    Log "STAGE A done"
}

$w = Join-Path $proj "$S1\weights\best_map50.pt"
if (-not (Test-Path $w)) { $w = Join-Path $proj "$S1\weights\best.pt" }

# ---------------------------------------------------------------- stage B1
Log "STAGE B1: single-model evaluation of $S1 on the 489 split"
$ErrorActionPreference = 'Continue'
& $py (Join-Path $Root 'src\evaluate.py') --weights $w --data $val489 `
    --tag "${S1}_single" --imgsz 1024 --batch 4 --skip-cm --square --augment 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$S1-eval.log") | Out-Null
$ErrorActionPreference = 'Stop'
$single = $null
$jf = Join-Path $proj "$S1\metrics_${S1}_single.json"
if (Test-Path $jf) {
    $single = (Get-Content $jf -Raw -Encoding utf8 | ConvertFrom-Json).mAP50
    Log "  $S1 single mAP50 = $single"
}

# ---------------------------------------------------------------- stage B2
Log "STAGE B2: fusion evaluation (F,B,H1,D,G,S1)"
$ErrorActionPreference = 'Continue'
& $py (Join-Path $Root 'src\ensemble_eval.py') --models "F,B,H1,D,G,S1" `
    --iou-grid 0.5,0.55 2>&1 | Tee-Object -FilePath (Join-Path $logDir 'night2-fusion.log') | Out-Null
$ErrorActionPreference = 'Stop'
$runDir = Get-ChildItem (Join-Path $Root 'runs\audit\ensemble') -Directory |
          Where-Object { $_.Name -match 'S1' } | Sort-Object LastWriteTime -Descending |
          Select-Object -First 1
if (-not $runDir) { Log 'no fusion run directory produced'; exit 1 }
$cfg = Join-Path $runDir.FullName 'fusion_config.json'
$best = (Get-Content $cfg -Raw -Encoding utf8 | ConvertFrom-Json)
Log ("  fusion best mAP50 = {0} (iou_thr {1})" -f $best.expected_mAP50, $best.iou_thr)

# ---------------------------------------------------------------- stage B3
Log "STAGE B3: exporting submission from $($runDir.Name)"
$out = Join-Path $Root 'results_test2_submit_night2.json'
$ErrorActionPreference = 'Continue'
& $py (Join-Path $Root 'src\export_fused.py') --config $cfg --out $out 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir 'night2-export.log') | Out-Null
$ErrorActionPreference = 'Stop'
if ($LASTEXITCODE -ne 0) { Log 'EXPORT FAILED'; exit 1 }
& $py (Join-Path $Root 'src\validate_submission.py') $out 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir 'night2-validate.log') | Out-Null
if ($LASTEXITCODE -eq 0) { Log 'DELIVERABLE READY: results_test2_submit_night2.json (valid)' }
else { Log 'submission INVALID -- do not upload' }

# ---------------------------------------------------------------- stage C
# Only if there is comfortable clock left: a second, different-architecture member.
$now = Get-Date
$cutoff = $now.Date.AddHours(5).AddMinutes(40)
if ($now -lt $cutoff) {
    $minutes = ($cutoff - $now).TotalMinutes
    $ep = [Math]::Floor(($minutes - 45) * 60 / 300)
    if ($ep -ge 8) {
        $S2 = 'S2_semi_hgnetv2'
        Log "STAGE C: training $S2 for $ep epochs (clock-limited)"
        $ErrorActionPreference = 'Continue'
        & $py (Join-Path $Root 'src\train_map50.py') `
            --weights (Join-Path $Root 'runs\first_test\H1_hgnetv2\weights\best_map50.pt') `
            --data $semi --name $S2 --project $proj `
            --epochs $ep --imgsz 1024 --batch 6 --lr0 0.0004 --workers 8 --seed 0 `
            --cls 0.625 --box 7.5 --scale 0.3 --mosaic 0.5 --mixup 0.05 --flipud 0.5 `
            --close-mosaic 8 --warmup-epochs 2 --save-period 10 2>&1 |
            Tee-Object -FilePath (Join-Path $logDir "$S2-train.log") | Out-Null
        $ErrorActionPreference = 'Stop'
        if ($LASTEXITCODE -eq 0) {
            $w2 = Join-Path $proj "$S2\weights\best_map50.pt"
            & $py (Join-Path $Root 'src\evaluate.py') --weights $w2 --data $val489 `
                --tag "${S2}_single" --imgsz 1024 --batch 4 --skip-cm --square --augment 2>&1 |
                Tee-Object -FilePath (Join-Path $logDir "$S2-eval.log") | Out-Null
            $j2 = Join-Path $proj "$S2\metrics_${S2}_single.json"
            if (Test-Path $j2) {
                Log ("  $S2 single mAP50 = " + (Get-Content $j2 -Raw -Encoding utf8 | ConvertFrom-Json).mAP50)
            }
            Log "STAGE C done -- rerun the fusion with S2 added to compare before exporting"
        } else { Log "STAGE C failed; stage B deliverable is unaffected" }
    } else { Log "STAGE C skipped (only $ep epochs would fit)" }
} else { Log 'STAGE C skipped (clock)' }

Log 'NIGHT2 FINISHED'
