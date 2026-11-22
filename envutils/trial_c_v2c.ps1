# Trial C: the same recipe as trial B, but on de-duplicated, conflict-free data.
#
# This is the single-variable experiment the data work exists for:
#
#   trial B  data/yolo_night75b   source 5023 -> 5628 effective, oversampling only
#   trial C  data/yolo_v2c_os     source 4720 -> 5253 effective, oversampling only
#
# Both then train identically (YOLOv8l / EDS-1, 1024 px, 80 epochs, batch 6,
# cls 0.625), so the only difference is the data cleaning:
#   - 42 training copies of validation images removed (leakage)
#   - 91 redundant duplicate copies removed
#   - 170 images whose duplicate copies carried CONTRADICTORY labels removed
#     (both copies, because no automatic rule can say which one is right)
#
# If cleaned data helps, trial C beats trial B on the same leak-free 552/489-style
# split.  If it does not, the label conflicts were not the bottleneck and the next
# lever is architecture or distillation.
#
#   powershell -ExecutionPolicy Bypass -File envutils\trial_c_v2c.ps1
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

param(
    [switch]$SkipWait,
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
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$Name = 'V2C_cls0625'
function Stamp([string]$m) { Write-Output ("[{0:MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) }

if (-not $SkipWait) {
    Stamp "waiting for the RT-DETR run to finish before taking the GPU"
    $deadline = (Get-Date).AddMinutes(600)
    while ((Get-Date) -lt $deadline) {
        $busy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
                  Where-Object { $_.CommandLine -match 'R0_rtdetr|R1_rtdetr|rtdetr_run|N75b_cls0625' })
        if ($busy.Count -eq 0) { break }
        Start-Sleep -Seconds 60
    }
    Stamp "GPU is free"
}

Stamp "training $Name on $data ($Epochs epochs)"
& $py (Join-Path $Root 'src\train_map50.py') `
    --weights (Join-Path $Root 'weights\yolov8_eds1.pt') `
    --data $data --name $Name --project $proj `
    --epochs $Epochs --imgsz 1024 --batch 6 --lr0 0.0008 --workers 8 --seed 0 `
    --cls 0.625 --box 7.5 --scale 0.3 --mosaic 0.5 --mixup 0.05 --flipud 0.5 `
    --close-mosaic 10 --warmup-epochs 3 --save-period 10 2>&1 |
    Tee-Object -FilePath (Join-Path $logDir "$Name-train.log") | Out-Null
if ($LASTEXITCODE -ne 0) {
    Stamp "TRAINING FAILED; tail:"
    Get-Content (Join-Path $logDir "$Name-train.log") -Tail 25
    exit 1
}
Stamp "training done"

$runDir = Join-Path $proj $Name
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

Stamp "TRIAL C DONE. local AP50 = $($best.mAP50)"
