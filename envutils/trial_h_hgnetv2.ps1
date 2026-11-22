# Trial H: HGNetV2 backbone with EDS-1 domain pretraining.
#
# This is a single-variable experiment against trial B:
#
#   trial B   YOLOv8l  (CSPDarknet, 43.6M)  + EDS-1, 1024, 80 ep, cls 0.625 -> 0.7835
#   trial H   HGNetV2  (PP-YOLOE-style, 54.7M) + EDS-1, same everything else
#
# Everything except the backbone is held identical (same dataset yolo_night75b,
# same oversampling, same lr/aug/cls), so a difference is attributable to the
# architecture plus its pretraining.
#
# Why this variant: of the five alternative checkpoints released alongside the
# EDS weights, only HGNetV2 loads in this ultralytics version -- nextvit_v8, HR,
# LIM and DOAM all reference custom modules from the authors' fork
# (ultralytics.nn.modules.*).  HGNetV2 is a backbone ultralytics implements
# natively, and it is genuinely different from CSPDarknet, which is the point:
# every model we own is currently a YOLOv8, so their errors are correlated.
#
# Epoch budget is set to land before the 18:00 review checkpoint; the full run
# would want more, and this is recorded rather than hidden.
#
#   powershell -ExecutionPolicy Bypass -File envutils\trial_h_hgnetv2.ps1
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

param(
    [int]$Epochs = 40,
    [string]$Weights = 'weights\hgnetv2_eds1.pt',
    [string]$Name = 'H1_hgnetv2'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'
$env:YOLO_CONFIG_DIR = Join-Path $Root '.ultralytics'
$env:TMP = Join-Path $Root '.tmp'
$env:TEMP = $env:TMP

$py     = Join-Path $Root '.venv\Scripts\python.exe'
$data   = Join-Path $Root 'data\yolo_night75b\data.yaml'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
New-Item -ItemType Directory -Force -Path $logDir, $env:TMP | Out-Null

function Stamp([string]$m) { Write-Output ("[{0:MM-dd HH:mm:ss}] {1}" -f (Get-Date), $m) }

Stamp "training $Name from $Weights for $Epochs epochs"
$ErrorActionPreference = 'Continue'
& $py (Join-Path $Root 'src\train_map50.py') `
    --weights (Join-Path $Root $Weights) `
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

Stamp "TRIAL H DONE. local AP50 = $($best.mAP50)"
