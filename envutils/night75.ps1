# User-started overnight training. ASCII-only for Windows PowerShell 5.1.
[CmdletBinding()]
param(
    [string]$Data = 'data/yolo_night75/data.yaml',
    [string]$RunPrefix = 'N75_20261120',
    [int]$Epochs = 80,
    [int]$Batch = 6,
    [string]$Deadline = '2026-11-21T08:00:00+08:00',
    [switch]$Enable1280,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$env:PYTHONIOENCODING = 'utf-8'
$py = Join-Path $Root '.venv\Scripts\python.exe'
$project = Join-Path $Root 'runs\first_test'
$logRoot = Join-Path $Root ('runs\night75\' + $RunPrefix)
$deadlineTime = [DateTimeOffset]::Parse($Deadline, [Globalization.CultureInfo]::InvariantCulture)
$fName = $RunPrefix + '_F1024'
$gName = $RunPrefix + '_G1280'
$script:Candidates = @()
$script:Exports = @()

function Stamp([string]$Message) {
    Write-Host ('[{0:HH:mm:ss}] {1}' -f (Get-Date), $Message)
}

function Invoke-Python([string[]]$Arguments, [string]$LogName) {
    # Windows PowerShell 5.1 wraps ordinary native stderr as ErrorRecords.
    # Continue logging those records; use the process exit code for failure.
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $py @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $logRoot $LogName) | Out-Host
        $processExit = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    if ($processExit -ne 0) {
        throw ('Python failed ({0}); see {1}' -f $processExit, (Join-Path $logRoot $LogName))
    }
}

function Evaluate-Candidate([string]$Weights, [string]$Tag, [int]$Size, [bool]$Tta) {
    $evalBatch = if ($Size -gt 1024) { 2 } else { 4 }
    $evalArguments = @(
        (Join-Path $Root 'src\evaluate.py'), '--weights', $Weights,
        '--data', $Data, '--tag', $Tag, '--imgsz', [string]$Size,
        '--batch', [string]$evalBatch, '--conf', '0.001', '--square', '--skip-cm'
    )
    if ($Tta) { $evalArguments += '--augment' }
    Invoke-Python -Arguments $evalArguments -LogName ($Tag + '-eval.log')
    $runDir = Split-Path -Parent (Split-Path -Parent $Weights)
    $metricsPath = Join-Path $runDir ('metrics_' + $Tag + '.json')
    $metrics = Get-Content -LiteralPath $metricsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $script:Candidates += [PSCustomObject]@{
        tag = $Tag; weights = $Weights; imgsz = $Size; augment = $Tta
        batch = $evalBatch; conf = 0.001; rect = $false; mAP50 = [double]$metrics.mAP50
        metrics_file = $metricsPath
    }
    Stamp ('{0}: mAP50={1:F4}, TTA={2}' -f $Tag, $metrics.mAP50, $Tta)
}

function Evaluate-Run([string]$Name, [int]$Size) {
    foreach ($checkpoint in @('best_map50', 'best')) {
        $weights = Join-Path (Join-Path (Join-Path $project $Name) 'weights') ($checkpoint + '.pt')
        if (-not (Test-Path -LiteralPath $weights)) { throw ('Missing checkpoint: ' + $weights) }
        Evaluate-Candidate -Weights $weights -Tag ($Name + '_' + $checkpoint) -Size $Size -Tta $false
        Evaluate-Candidate -Weights $weights -Tag ($Name + '_' + $checkpoint + '_tta') -Size $Size -Tta $true
    }
}

function Export-Candidate($Candidate, [string]$Suffix) {
    $outputPath = Join-Path $Root ('results_test2_submit_' + $RunPrefix + '_' + $Suffix + '.json')
    if (Test-Path -LiteralPath $outputPath) { throw ('Refusing to replace submission: ' + $outputPath) }
    $exportArguments = @(
        (Join-Path $Root 'src\export_submission.py'), '--weights', $Candidate.weights,
        '--imgsz', [string]$Candidate.imgsz, '--batch', [string]$Candidate.batch,
        '--conf', '0.001', '--iou', '0.7', '--max-det', '300', '--square', '--out', $outputPath
    )
    if ($Candidate.augment) { $exportArguments += '--augment' }
    Invoke-Python -Arguments $exportArguments -LogName ($Suffix + '-export.log')
    Invoke-Python -Arguments @((Join-Path $Root 'src\validate_submission.py'), $outputPath) -LogName ($Suffix + '-validate.log')
    $script:Exports += [PSCustomObject]@{
        file = $outputPath; sha256 = (Get-FileHash -LiteralPath $outputPath -Algorithm SHA256).Hash
        candidate = $Candidate; structural_validation = 'passed'
    }
}

$plan = [PSCustomObject]@{
    data = $Data; initialization = 'weights/yolov8_eds1.pt'
    stage_f = [PSCustomObject]@{ name = $fName; epochs = $Epochs; imgsz = 1024; batch = $Batch; lr0 = 0.0008 }
    stage_g = [PSCustomObject]@{ enabled = [bool]$Enable1280; name = $gName; epochs = 12; imgsz = 1280; batch = 3; lr0 = 0.0001 }
    deadline = $deadlineTime.ToString('o'); validation_conf = 0.001; square_validation = $true
    pseudo_labels = $false; automatic_online_upload = $false
}
if ($PlanOnly) { $plan | ConvertTo-Json -Depth 6; exit 0 }

if ($RunPrefix -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunPrefix must contain only ASCII letters, digits, underscore or hyphen.' }
if ($Epochs -lt 1 -or $Batch -lt 1) { throw 'Epochs and Batch must be positive.' }
foreach ($required in @($py, $Data, (Join-Path $Root 'weights\yolov8_eds1.pt'), (Join-Path $Root 'src\train_map50.py'))) {
    if (-not (Test-Path -LiteralPath $required)) { throw ('Missing required file: ' + $required) }
}
foreach ($name in @($fName, $gName)) {
    if (Test-Path -LiteralPath (Join-Path $project $name)) { throw ('Run already exists; choose a new RunPrefix: ' + $name) }
}
if (Test-Path -LiteralPath $logRoot) { throw ('Log directory already exists; choose a new RunPrefix: ' + $logRoot) }

# This also catches the old midnight waiter before it starts a second GPU job.
$conflicts = @(Get-CimInstance Win32_Process | Where-Object {
    $_.ProcessId -ne $PID -and $_.Name -match '^(python|yolo|powershell|pwsh)\.exe$' -and
    $_.CommandLine -match '(detect\s+train|train_map50\.py|[\\/](midnight|overnight)\.ps1)'
})
if ($conflicts.Count -gt 0) {
    $conflicts | Format-List ProcessId, Name, CommandLine | Out-Host
    throw 'Another training job or old midnight queue exists. Resolve that queue before starting this run.'
}
$hoursLeft = ($deadlineTime - [DateTimeOffset]::Now).TotalHours
$estimatedHours = $Epochs * 260.0 / 3600.0 + 0.75
if ($hoursLeft -lt $estimatedHours) {
    throw ('Only {0:F2} h until deadline; estimated F + evaluation/export is {1:F2} h. Reduce Epochs or extend Deadline.' -f $hoursLeft, $estimatedHours)
}
New-Item -ItemType Directory -Path $logRoot | Out-Null
$plan | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $logRoot 'plan.json') -Encoding UTF8

$dWeights = Join-Path $project 'D_ft1024\weights\best.pt'
Evaluate-Candidate -Weights $dWeights -Tag ($RunPrefix + '_D_ref') -Size 1024 -Tta $false
Evaluate-Candidate -Weights $dWeights -Tag ($RunPrefix + '_D_ref_tta') -Size 1024 -Tta $true
$bestD = $script:Candidates | Sort-Object mAP50 -Descending | Select-Object -First 1

Stamp 'F: public EDS initialization, real training labels, 1024px.'
Invoke-Python -Arguments @(
    (Join-Path $Root 'src\train_map50.py'), '--weights', (Join-Path $Root 'weights\yolov8_eds1.pt'),
    '--data', $Data, '--name', $fName, '--epochs', [string]$Epochs, '--imgsz', '1024',
    '--batch', [string]$Batch, '--lr0', '0.0008', '--close-mosaic', '10'
) -LogName 'F-train.log'
Evaluate-Run -Name $fName -Size 1024
$bestF = $script:Candidates | Where-Object { $_.tag.StartsWith($fName) } | Sort-Object mAP50 -Descending | Select-Object -First 1

if ($Enable1280 -and $bestF.mAP50 -ge ($bestD.mAP50 + 0.005) -and ($deadlineTime - [DateTimeOffset]::Now).TotalHours -ge 2.25) {
    Stamp 'G: 12-epoch 1280px fine-tune; enough time remains and F improved locally.'
    try {
        Invoke-Python -Arguments @(
            (Join-Path $Root 'src\train_map50.py'), '--weights', $bestF.weights, '--data', $Data,
            '--name', $gName, '--epochs', '12', '--imgsz', '1280', '--batch', '3', '--lr0', '0.0001',
            '--warmup-epochs', '1', '--mosaic', '0.2', '--mixup', '0', '--flipud', '0', '--close-mosaic', '5'
        ) -LogName 'G-train.log'
        Evaluate-Run -Name $gName -Size 1280
    } catch {
        Stamp ('G failed; keeping completed F and D candidates. ' + $_.Exception.Message)
    }
} else {
    Stamp 'Skipping G: disabled, insufficient time, or F did not exceed D by 0.5 local percentage points.'
}

$ranked = @($script:Candidates | Sort-Object mAP50 -Descending)
$best = $ranked[0]
$bestNew = $ranked | Where-Object { $_.tag.StartsWith($fName) -or $_.tag.StartsWith($gName) } | Select-Object -First 1
$summary = [PSCustomObject]@{
    plan = $plan; candidates = $ranked; selected = $best; best_new = $bestNew
    exports = @(); note = 'Local AP50 selects candidates; no fixed local-to-online conversion or guaranteed online score.'
}
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $logRoot 'summary.json') -Encoding UTF8
Export-Candidate -Candidate $best -Suffix 'selected'
if ($best.tag -ne $bestNew.tag) { Export-Candidate -Candidate $bestNew -Suffix 'new' }
$summary.exports = $script:Exports
$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $logRoot 'summary.json') -Encoding UTF8
Stamp ('DONE. Selected local AP50={0:F4}. See {1}. Upload remains manual.' -f $best.mAP50, (Join-Path $logRoot 'summary.json'))
