# Trial B: corrected oversampling + a higher classification-loss gain.
#
# Two changes against last night's F, both grounded in what the data said:
#
#   1. oversampling: last night gave tool x3, pressuretank x2, drinkbottle x2 and
#      knife x2.  On the leak-free 489 split that lifted tool +10.8 and
#      pressuretank +10.0 AP50 but cost knife -6.0 and drinkbottle -2.1.  Those
#      two classes did not need the extra exposure, so this run gives only
#      tool x3 and pressuretank x2.
#   2. cls loss gain 0.5 -> 0.625.  The leaderboard metric is AP50, which only
#      rewards ranking, not box precision; the 2021 winner of this competition
#      reported raising the classification term by 25% helped.  0.625 is exactly
#      that 25% step, not an arbitrary bigger number.
#
# Everything else is held at last night's F configuration, so the comparison is
# F vs B on the same 489-image split.  Two variables change at once, which is
# recorded here on purpose: with ~5 h per run there is no budget for a clean
# A/B tonight, so if B underperforms the next run bisects.
#
#   powershell -ExecutionPolicy Bypass -File envutils\trial_b.ps1
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'

$py     = Join-Path $Root '.venv\Scripts\python.exe'
$data   = Join-Path $Root 'data\yolo_night75b\data.yaml'
$valMan = Join-Path $Root 'runs\audit\E_content_disjoint_val.txt'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$Name = 'N75b_cls0625'
function Stamp([string]$m) { Write-Output ("[{0:MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) }

# ------------------------------------------------------------------ 1. train
Stamp "training $Name (80 ep, 1024, batch 6, cls=0.625)"
& $py (Join-Path $Root 'src\train_map50.py') `
    --weights (Join-Path $Root 'weights\yolov8_eds1.pt') `
    --data $data --name $Name --project $proj `
    --epochs 80 --imgsz 1024 --batch 6 --lr0 0.0008 --workers 8 --seed 0 `
    --cls 0.625 --box 7.5 `
    --scale 0.3 --mosaic 0.5 --mixup 0.05 --flipud 0.5 `
    --close-mosaic 10 --warmup-epochs 3 --save-period 10 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$Name-train.log") | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stamp "TRAINING FAILED (exit $LASTEXITCODE); tail:"
    Get-Content (Join-Path $logDir "$Name-train.log") -Tail 25
    exit 1
}
Stamp "training done"

# --------------------------------------------------------------- 2. evaluate
$runDir = Join-Path $proj $Name
$results = @()
foreach ($ckpt in @(@('best_map50', $true), @('best_map50', $false), @('best', $true))) {
    $w = Join-Path (Join-Path $runDir 'weights') ($ckpt[0] + '.pt')
    if (-not (Test-Path $w)) { Stamp ("missing {0}" -f $w); continue }
    $tag = "${Name}_$($ckpt[0])"
    if ($ckpt[1]) { $tag = "${tag}_tta" }
    $args = @((Join-Path $Root 'src\evaluate.py'), '--weights', $w, '--data', $data,
              '--tag', $tag, '--imgsz', '1024', '--batch', '4', '--skip-cm', '--square')
    if ($ckpt[1]) { $args += '--augment' }
    Stamp "evaluating $tag"
    & $py @args 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$tag-eval.log") | Out-Null
    $json = Join-Path $runDir "metrics_$tag.json"
    if (Test-Path $json) {
        $j = Get-Content $json -Raw -Encoding utf8 | ConvertFrom-Json
        $results += [pscustomobject]@{ tag = $tag; weights = $w; tta = $ckpt[1]; mAP50 = [double]$j.mAP50 }
        Stamp ("  {0}: mAP50 = {1}" -f $tag, $j.mAP50)
    }
}

# ----------------------------------------------------------------- 3. export
if ($results.Count -eq 0) { Stamp 'no evaluation results; cannot choose'; exit 1 }
$best = $results | Sort-Object -Property mAP50 -Descending | Select-Object -First 1
Stamp ("selected {0} (mAP50 {1})" -f $best.tag, $best.mAP50)

$exportArgs = @((Join-Path $Root 'src\export_submission.py'), '--weights', $best.weights,
                '--imgsz', '1024', '--conf', '0.001', '--iou', '0.7', '--max-det', '300',
                '--batch', '4', '--square',
                '--out', (Join-Path $Root "results_test2_submit_$Name.json"))
if ($best.tta) { $exportArgs += '--augment' }
Stamp "exporting submission"
& $py @exportArgs 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$Name-export.log") | Out-Null
if ($LASTEXITCODE -ne 0) { Stamp 'export FAILED'; exit 1 }

& $py (Join-Path $Root 'src\validate_submission.py') `
    (Join-Path $Root "results_test2_submit_$Name.json") 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$Name-validate.log") | Out-Null
if ($LASTEXITCODE -eq 0) { Stamp 'submission valid' } else { Stamp 'submission INVALID' }

Stamp "TRIAL B DONE. local AP50 = $($best.mAP50), file results_test2_submit_$Name.json"
