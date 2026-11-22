# Resume the power-cut V2C run, then evaluate and export.
#
# The run reached epoch 24/80 before the outage.  last.pt holds optimizer, EMA,
# AMP scaler and the original train_args, so this is a true continuation (LR
# schedule and augmentation state included), not a fresh start with warm weights.
#
# --map50-floor is passed with the best AP50 already achieved (0.7115 at epoch 15)
# because a resumed run restarts the internal best-tracker from scratch; without
# the floor, an early resumed epoch scoring 0.68 would overwrite the better
# best_map50.pt checkpoint.
#
#   powershell -ExecutionPolicy Bypass -File envutils\resume_v2c.ps1
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

param(
    [double]$Map50Floor = 0.7115,
    [int]$Epochs = 80
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'

$py     = Join-Path $Root '.venv\Scripts\python.exe'
$data   = Join-Path $Root 'data\yolo_v2c_os\data.yaml'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
$Name   = 'V2C_cls0625'
$runDir = Join-Path $proj $Name
$last   = Join-Path (Join-Path $runDir 'weights') 'last.pt'

function Stamp([string]$m) { Write-Output ("[{0:MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) }

if (-not (Test-Path $last)) { Stamp "no last.pt at $last"; exit 1 }

Stamp "resuming $Name from epoch 24, floor $Map50Floor"
$ErrorActionPreference = 'Continue'   # native stderr must not become fatal
& $py (Join-Path $Root 'src\train_map50.py') `
    --weights $last --data $data --name $Name --project $proj --resume `
    --map50-floor $Map50Floor --epochs $Epochs `
    --imgsz 1024 --batch 6 --lr0 0.0008 --workers 8 --seed 0 `
    --cls 0.625 --box 7.5 --scale 0.3 --mosaic 0.5 --mixup 0.05 --flipud 0.5 `
    --close-mosaic 10 --warmup-epochs 3 --save-period 10 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$Name-resume.log") | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stamp "RESUME FAILED (exit $LASTEXITCODE); tail:"
    Get-Content (Join-Path $logDir "$Name-resume.log") -Tail 25
    exit 1
}
Stamp "resume finished"

$results = @()
foreach ($ckpt in @(@('best_map50', $true), @('best', $true), @('best', $false))) {
    $w = Join-Path (Join-Path $runDir 'weights') ($ckpt[0] + '.pt')
    if (-not (Test-Path $w)) { continue }
    $tag = "${Name}_$($ckpt[0])"
    if ($ckpt[1]) { $tag = "${tag}_tta" }
    $a = @((Join-Path $Root 'src\evaluate.py'), '--weights', $w, '--data', $data,
           '--tag', $tag, '--imgsz', '1024', '--batch', '4', '--skip-cm', '--square')
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

$out = Join-Path $Root "results_test2_submit_$Name.json"
$exp = @((Join-Path $Root 'src\export_submission.py'), '--weights', $best.weights,
         '--imgsz', '1024', '--conf', '0.001', '--iou', '0.7', '--max-det', '300',
         '--batch', '4', '--square', '--out', $out)
if ($best.tta) { $exp += '--augment' }
Stamp "exporting submission"
& $py @exp 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$Name-export.log") | Out-Null
if ($LASTEXITCODE -ne 0) { Stamp 'export FAILED'; exit 1 }
& $py (Join-Path $Root 'src\validate_submission.py') $out 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$Name-validate.log") | Out-Null
if ($LASTEXITCODE -eq 0) { Stamp 'submission valid' } else { Stamp 'submission INVALID' }

Stamp "V2C RESUME DONE. local AP50 = $($best.mAP50)"
