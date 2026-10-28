#Requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('classical', 'patchcore', 'efficientad', 'dinov3', 'superadd')][string]$Profile = 'classical',
    [ValidateSet('serve', 'infer')][string]$Action = 'serve',
    [string]$Image,
    [string]$PythonPath,
    [string]$BundlePath,
    [string]$ArtifactRoot,
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [ValidateRange(1, 64)][int]$Threads = 2,
    [switch]$ShowPlan
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$profiles = @{
    classical = @{ name = 'classical_diff'; family = 'classical_cv'; bundle = '.tmp\cv1-release-20261023\point2-qualification-20261023\models\classical-diff-bottle-qualified-v1' }
    patchcore = @{ name = 'patchcore'; family = 'anomalib_patchcore'; bundle = '.tmp\cv1-release-20261023\point2-qualification-20261023\models\patchcore-bottle-qualified-v1' }
    efficientad = @{ name = 'efficientad'; family = 'anomalib_efficientad'; bundle = '.tmp\cv1-release-20261023\efficientad-qualified\models\efficientad-bottle-torch213-full20-v1' }
    dinov3 = @{ name = 'dino_memory'; family = 'dinov3_timm_memory'; bundle = '.tmp\cv1-release-20261023\p10-dinov3-20261024\demo-core\models\dinov3-small-bottle-demo-v1' }
    superadd = @{ name = 'superadd'; family = 'anomalib_superadd'; bundle = '.tmp\superadd-20261025-loco-splicing-r04' }
}
$selected = $profiles[$Profile]
if (-not $PythonPath) {
    $PythonPath = if ($Profile -eq 'superadd') {
        Join-Path $projectRoot '.venv-py312-psf-test\Scripts\python.exe'
    } else {
        Join-Path $env:USERPROFILE 'AppData\Local\Temp\cv213\Scripts\python.exe'
    }
}
if (-not $BundlePath) {
    $BundlePath = Join-Path $projectRoot $selected.bundle
}
if (-not $ArtifactRoot) {
    $ArtifactRoot = Join-Path $projectRoot ".tmp\local-demo\$Profile\evidence"
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw "Python not found: $PythonPath. Supply -PythonPath for the bundle-compatible environment."
}
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
$BundlePath = (Resolve-Path -LiteralPath $BundlePath).Path
$ArtifactRoot = [System.IO.Path]::GetFullPath($ArtifactRoot)
$requiredFiles = if ($Profile -eq 'superadd') {
    @('model.pt', 'fit_metadata.json', 'run_status.json')
} else {
    @('model.json', 'calibration.json', 'bundle_commit.json')
}
foreach ($file in $requiredFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $BundlePath $file) -PathType Leaf)) {
        throw "Incomplete model bundle: missing $file in $BundlePath"
    }
}
if ($Profile -eq 'superadd') {
    $fit = Get-Content -LiteralPath (Join-Path $BundlePath 'fit_metadata.json') -Raw | ConvertFrom-Json -AsHashtable
    $runStatus = Get-Content -LiteralPath (Join-Path $BundlePath 'run_status.json') -Raw | ConvertFrom-Json -AsHashtable
    if ($runStatus.status -ne 'complete') { throw 'SuperADD fit is incomplete.' }
    $modelName = 'superadd'
    $modelId = Split-Path -Leaf $BundlePath
    $category = $fit.category
    $dependencies = $fit.runtime
} else {
    $model = Get-Content -LiteralPath (Join-Path $BundlePath 'model.json') -Raw | ConvertFrom-Json -AsHashtable
    if ($model.family -ne $selected.family -or $model.category -ne 'bottle') {
        throw "The $Profile profile requires a calibrated $($selected.family) bottle bundle."
    }
    if ($model.model_id -ne (Split-Path -Leaf $BundlePath)) {
        throw 'The bundle folder name must match model.json model_id for API loading.'
    }
    $modelName = $selected.name
    $modelId = $model.model_id
    $category = $model.category
    $dependencies = $model.runtime_identity.dependencies
}
$configPath = Join-Path $projectRoot 'configs\deployment\local.yaml'
$arguments = @('-m', 'industrial_inspection.cli', '--config', $configPath)
if ($Action -eq 'infer') {
    if (-not $Image -or -not (Test-Path -LiteralPath $Image -PathType Leaf)) {
        throw 'For -Action infer supply -Image with an existing image file.'
    }
    $Image = (Resolve-Path -LiteralPath $Image).Path
    $arguments += @('infer', '--image', $Image, '--model-name', $modelName,
        '--model-dir', $BundlePath, '--category', $category)
} else {
    if ($Image) { throw '-Image is only used with -Action infer.' }
    $arguments += @('serve', '--host', '127.0.0.1', '--port', "$Port")
}
$settings = @{
    II_PROJECT_ROOT = $projectRoot
    II_MODEL_ROOT = (Split-Path -Parent $BundlePath)
    II_MODEL_NAME = $modelName
    II_MODEL_ID = $modelId
    II_CATEGORY = $category
    II_DEVICE = 'cpu'
    II_ARTIFACT_ROOT = $ArtifactRoot
    PYTHONPATH = (Join-Path $projectRoot 'src')
    OMP_NUM_THREADS = "$Threads"
    MKL_NUM_THREADS = "$Threads"
}
if ($ShowPlan) {
    [ordered]@{
        python = $PythonPath
        arguments = $arguments
        environment = $settings
        url = "http://127.0.0.1:$Port"
        profile = $Profile
        action = $Action
        expected_dependencies = $dependencies
    } | ConvertTo-Json -Depth 8
    return
}

$previous = @{}
foreach ($name in $settings.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location -LiteralPath $projectRoot
try {
    foreach ($name in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($name, $settings[$name], 'Process')
    }
    Write-Host "$Profile demo ($Action): $category"
    Write-Host "Evidence: $ArtifactRoot"
    if ($Action -eq 'serve') {
        Write-Host "Open http://127.0.0.1:$Port after application startup; Ctrl+C stops the service."
    }
    & $PythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Demo service exited with code $LASTEXITCODE."
    }
}
finally {
    foreach ($name in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
    Pop-Location
}
