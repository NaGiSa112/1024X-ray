# RT-DETR run: architecture change for ensemble diversity.
#
# Steps
#   1. wait until trial B has finished, so the two never share the 24 GB card
#   2. 2-epoch smoke test on 15% of the data to measure seconds/epoch
#      (RT-DETR's transformer decoder at 1024 px is much slower per iteration than
#      YOLOv8, and guessing the epoch budget wrong is how a run ends up useless)
#   3. pick the epoch count from that measurement against a hard clock
#   4. train, evaluate (best_map50 and best, with/without TTA), export, validate
#
# The deadline is wall-clock, not a hope: if the full run cannot fit, the script
# reduces epochs rather than overrunning.
#
#   powershell -ExecutionPolicy Bypass -File envutils\rtdetr_run.ps1
#   powershell -ExecutionPolicy Bypass -File envutils\rtdetr_run.ps1 -SkipWait
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

param(
    [switch]$SkipWait,
    [string]$Deadline = '21:00',
    [int]$MaxEpochs = 45,
    [int]$Imgsz = 640,
    [int]$Batch = 8
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'

$py     = Join-Path $Root '.venv\Scripts\python.exe'
$data   = Join-Path $Root 'data\yolo_night75b\data.yaml'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Stamp([string]$m) { Write-Output ("[{0:MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) }

# ------------------------------------------------------- 1. wait for the GPU
if (-not $SkipWait) {
    Stamp "waiting for the YOLO runs to finish (RT-DETR must not share the GPU)"
    # NOTE: this loop variable must NOT be called $deadline.  PowerShell variable
    # names are case-insensitive, so a lower-case $deadline silently overwrites
    # the -Deadline parameter and the clock arithmetic below then tries to parse
    # a DateTime as "HH:mm".
    $waitUntil = (Get-Date).AddMinutes(600)
    while ((Get-Date) -lt $waitUntil) {
        $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                  Where-Object { $_.CommandLine -match 'N75b_cls0625|V2C_cls0625|H1_hgnetv2|trial_[bch]|R0_rtdetr|R1_rtdetr' })
        if ($busy.Count -eq 0) { break }
        Start-Sleep -Seconds 60
    }
    Stamp "GPU is free"
}

# RT-DETR writes warnings to stderr (AMP checks, deterministic kernels).  Under
# $ErrorActionPreference='Stop' PowerShell 5.1 turns a native command's stderr
# line into a terminating NativeCommandError, which killed the first attempt at
# the very first backward pass.  Local Continue keeps $LASTEXITCODE meaningful.
$ErrorActionPreference = 'Continue'

# ------------------------------------------------------------ 2. smoke test
$smokeName = 'R0_rtdetr_smoke'
Stamp "smoke test: 2 epochs, 15% of data, 1024px, batch 4"
& $py (Join-Path $Root 'src\train_rtdetr.py') --data $data --name $smokeName `
    --epochs 2 --imgsz $Imgsz --batch $Batch --lr0 0.0008 --fraction 0.15 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$smokeName.log") | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stamp "SMOKE FAILED; tail:"
    Get-Content (Join-Path $logDir "$smokeName.log") -Tail 30
    exit 1
}

$smokeCsv = Join-Path (Join-Path $proj $smokeName) 'results.csv'
$perEpochSmoke = $null
if (Test-Path $smokeCsv) {
    $rows = Import-Csv $smokeCsv
    if ($rows.Count -ge 2) {
        $perEpochSmoke = ([double]$rows[-1].time - [double]$rows[0].time) / ($rows.Count - 1)
    }
}
if (-not $perEpochSmoke) { $perEpochSmoke = 600 }
$perEpochFull = $perEpochSmoke / 0.15
Stamp ("smoke: {0:N0}s/epoch at 15%  =>  est. {1:N0}s/epoch full ({2:N0} min)" -f $perEpochSmoke, $perEpochFull, ($perEpochFull/60))

# ------------------------------------------------------- 3. decide epochs
$now = Get-Date
$parts = $Deadline.Split(':')
$target = $now.Date.AddHours([int]$parts[0]).AddMinutes([int]$parts[1])
if ($target -le $now) { $target = $target.AddDays(1) }
$budgetMin = ($target - $now).TotalMinutes
# leave 40 min for evaluation + export + validation
$usable = [Math]::Max(0, $budgetMin - 40)
$fit = [Math]::Floor($usable * 60 / $perEpochFull)
$epochs = [Math]::Min($MaxEpochs, $fit)
Stamp ("clock: {0:N0} min to {1}  =>  fits {2} epochs, using {3}" -f $budgetMin, $Deadline, $fit, $epochs)
if ($epochs -lt 8) {
    Stamp "not enough clock for a meaningful RT-DETR run; stopping"
    exit 1
}

# ------------------------------------------------------------- 4. full run
$name = 'R1_rtdetr1024'
Stamp "training $name for $epochs epochs"
& $py (Join-Path $Root 'src\train_rtdetr.py') --data $data --name $name `
    --epochs $epochs --imgsz $Imgsz --batch $Batch --lr0 0.0008 --fraction 1.0 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$name-train.log") | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stamp "TRAINING FAILED; tail:"
    Get-Content (Join-Path $logDir "$name-train.log") -Tail 30
    exit 1
}
Stamp "training done"

# --------------------------------------------------------- 5. eval + export
$runDir = Join-Path $proj $name
$results = @()
foreach ($ckpt in @(@('best_map50', $true), @('best', $true), @('best', $false))) {
    $w = Join-Path (Join-Path $runDir 'weights') ($ckpt[0] + '.pt')
    if (-not (Test-Path $w)) { continue }
    $tag = "${name}_$($ckpt[0])"
    if ($ckpt[1]) { $tag = "${tag}_tta" }
    $a = @((Join-Path $Root 'src\evaluate.py'), '--weights', $w, '--data', $data,
           '--tag', $tag, '--imgsz', "$Imgsz", '--batch', '2', '--skip-cm', '--square')
    if ($ckpt[1]) { $a += '--augment' }
    Stamp "evaluating $tag"
    & $py @a 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$tag-eval.log") | Out-Null
    $json = Join-Path $runDir "metrics_$tag.json"
    if (Test-Path $json) {
        $j = Get-Content $json -Raw -Encoding utf8 | ConvertFrom-Json
        $results += [pscustomobject]@{ tag = $tag; weights = $w; tta = $ckpt[1]; mAP50 = [double]$j.mAP50 }
        Stamp ("  {0}: mAP50 = {1}" -f $tag, $j.mAP50)
    }
}
if ($results.Count -eq 0) { Stamp 'no evaluation results'; exit 1 }
$best = $results | Sort-Object -Property mAP50 -Descending | Select-Object -First 1
Stamp ("selected {0} (mAP50 {1})" -f $best.tag, $best.mAP50)

$out = Join-Path $Root 'results_test2_submit_R1_rtdetr.json'
$exp = @((Join-Path $Root 'src\export_submission.py'), '--weights', $best.weights,
         '--imgsz', "$Imgsz", '--conf', '0.001', '--iou', '0.7', '--max-det', '300',
         '--batch', '2', '--square', '--out', $out)
if ($best.tta) { $exp += '--augment' }
Stamp "exporting submission"
& $py @exp 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$name-export.log") | Out-Null
if ($LASTEXITCODE -ne 0) { Stamp 'export FAILED'; exit 1 }
& $py (Join-Path $Root 'src\validate_submission.py') $out 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$name-validate.log") | Out-Null
if ($LASTEXITCODE -eq 0) { Stamp 'submission valid' } else { Stamp 'submission INVALID' }

Stamp "RT-DETR DONE. local AP50 = $($best.mAP50)"

