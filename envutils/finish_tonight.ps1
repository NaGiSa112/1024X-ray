# Tonight's pipeline: turn the finished B/C runs into validated submissions, then
# spend the remaining GPU time on the highest-leverage experiment (1024 px).
#
#   1. wait for C to finish training AND evaluation (signal: metrics_<tag>.json,
#      which src/evaluate.py writes as its very last step)
#   2. export + validate submissions for B and C at 640
#   3. fine-tune the better of B/C at 1024 px (D)
#   4. export + validate the 1024 submission, plus a TTA variant of it
#
# GPU work is strictly serialised: two trainings on a 24 GB card OOM, and
# exporting while a training run is active risks killing it.  Earlier version of
# this script keyed off the *log file existing*, which is true the instant the
# evaluation starts -- that raced the evaluation for VRAM.  The metrics JSON is
# only written after the evaluation completes, so it is the correct signal.
#
#   powershell -ExecutionPolicy Bypass -File envutils\finish_tonight.ps1
#
# ASCII-only on purpose (PowerShell 5.1 reads BOM-less .ps1 as the ANSI code page).

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'

$py     = Join-Path $Root '.venv\Scripts\python.exe'
$proj   = Join-Path $Root 'runs\first_test'
$logDir = Join-Path $Root 'runs\chain_logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Stamp([string]$msg) {
    Write-Output ("[{0:HH:mm:ss}] {1}" -f (Get-Date), $msg)
}

function Wait-ForFile([string]$path, [int]$timeoutMin, [string]$what) {
    $deadline = (Get-Date).AddMinutes($timeoutMin)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $path) { return $true }
        Start-Sleep -Seconds 20
    }
    Stamp "TIMEOUT waiting for $what"
    return $false
}

function Export-Submission([string]$tag, [string]$weights, [int]$imgsz, [int]$batch, [string]$suffix, [bool]$tta) {
    $label = if ($tta) { "$tag-tTA" } else { $tag }
    Stamp "exporting submission for $label at ${imgsz}px (tta=$tta)"
    $out = Join-Path $Root "results_test2_submit_$suffix.json"
    $extra = @()
    if ($tta) { $extra += '--augment' }
    & $py (Join-Path $Root 'src\export_submission.py') `
        --weights $weights --imgsz $imgsz --conf 0.01 --batch $batch @extra `
        --out $out 2>&1 | Tee-Object -FilePath (Join-Path $logDir "$suffix-export.log") | Out-Null
    if ($LASTEXITCODE -ne 0) { Stamp "$label export FAILED"; return $false }

    & $py (Join-Path $Root 'src\validate_submission.py') $out 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$suffix-validate.log") | Out-Null
    if ($LASTEXITCODE -ne 0) { Stamp "$label submission INVALID -- see $logDir"; return $false }
    Stamp "$label submission OK -> $out"
    return $true
}

$exported = @()

# ------------------------------------------------ 1. wait for B and C metrics
$metricsB = Join-Path (Join-Path $proj 'B_eds640') 'metrics_B_eds640.json'
$metricsC = Join-Path (Join-Path $proj 'C_eds_aug640') 'metrics_C_eds_aug640.json'

if (-not (Wait-ForFile $metricsB 10 'B metrics')) { Stamp "B metrics missing"; exit 1 }
Stamp "B metrics ready"

Stamp "waiting for C training + evaluation (up to 150 min)"
$cDone = Wait-ForFile $metricsC 150 'C metrics'
if ($cDone) { Stamp "C metrics ready" } else { Stamp "C not ready; continuing with B only" }

# ------------------------------------------------- 2. submissions for B / C
$bestB = Join-Path (Join-Path (Join-Path $proj 'B_eds640') 'weights') 'best.pt'
if (Export-Submission -tag 'B' -weights $bestB -imgsz 640 -batch 8 -suffix 'B' -tta $false) { $exported += 'B' }

$bestC = Join-Path (Join-Path (Join-Path $proj 'C_eds_aug640') 'weights') 'best.pt'
if ($cDone -and (Test-Path $bestC)) {
    if (Export-Submission -tag 'C' -weights $bestC -imgsz 640 -batch 8 -suffix 'C' -tta $false) { $exported += 'C' }
}

# ------------------------------------ 3. pick the better base, fine-tune at 1024
function Get-Map50([string]$jsonPath) {
    if (-not (Test-Path $jsonPath)) { return -1.0 }
    $j = Get-Content $jsonPath -Raw -Encoding utf8 | ConvertFrom-Json
    return [double]$j.mAP50
}

$mapB = Get-Map50 $metricsB
$mapC = if ($cDone) { Get-Map50 $metricsC } else { -1.0 }
Stamp ("val mAP50: B={0:N4}  C={1:N4}" -f $mapB, $mapC)

if ($mapC -gt $mapB -and $cDone -and (Test-Path $bestC)) {
    $baseW = $bestC; $baseTag = 'C'; $baseMap = $mapC
} else {
    $baseW = $bestB; $baseTag = 'B'; $baseMap = $mapB
}
Stamp "1024 px fine-tune base: $baseTag (mAP50 $baseMap)"

$dName = 'D_ft1024'
# 24 epochs, not 40: at 1024 px a v8l epoch costs ~250 s (2.5x the 640 px cost),
# so 40 epochs would land at ~22:45 and leave no room for export before 23:00.
$dEpochs = 24
$dOk = $false
foreach ($batch in @(6, 4)) {
    Stamp "=== D: 1024 px fine-tune, $dEpochs epochs, batch $batch ==="
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Root 'envutils\train_d1024.ps1') `
        -Init $baseW -Epochs $dEpochs -Batch $batch -Imgsz 1024 -Name $dName 2>&1 |
        Tee-Object -FilePath (Join-Path $logDir "$dName-run-batch$batch.log") | Out-Null
    if ($LASTEXITCODE -eq 0) { $dOk = $true; break }
    Stamp "D failed with batch $batch (likely OOM); retrying smaller"
}

# ------------------------------------------------------ 4. 1024 px submissions
if ($dOk) {
    $bestD = Join-Path (Join-Path (Join-Path $proj $dName) 'weights') 'best.pt'
    if (Export-Submission -tag 'D' -weights $bestD -imgsz 1024 -batch 4 -suffix 'D' -tta $false) { $exported += 'D' }
    if (Export-Submission -tag 'D' -weights $bestD -imgsz 1024 -batch 4 -suffix 'D_tta' -tta $true) { $exported += 'D_tta' }
} else {
    Stamp "D training failed; no 1024 px submission"
}

Stamp ("ALL DONE. submissions: {0}" -f ($exported -join ', '))
