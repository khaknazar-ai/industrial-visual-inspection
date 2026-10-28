#Requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('ron88', 'pcad')][string]$Dataset = 'ron88',
    [ValidateSet('serve', 'infer')][string]$Action = 'serve',
    [string]$Image,
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [switch]$ShowPlan
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv-py312-psf-test\Scripts\python.exe'
$run = if ($Dataset -eq 'ron88') { 'ron88-dinov3-20261026-r01' } else { 'pcad-dinov3-20261026-r01' }
$category = if ($Dataset -eq 'ron88') { 'bottle' } else { 'switch_8_port' }
$runRoot = Join-Path $root ".tmp\$run"
$model = Join-Path $runRoot 'model'
$status = Get-Content (Join-Path $runRoot 'experiment\run_status.json') -Raw | ConvertFrom-Json
if ($status.status -ne 'complete') { throw 'The external experiment is incomplete.' }
foreach ($file in @('model.json', 'thresholds.json', 'encoder.safetensors')) {
    if (-not (Test-Path -LiteralPath (Join-Path $model $file))) { throw "Missing model asset: $file" }
}
$settings = @{
    PYTHONPATH = (Join-Path $root 'src'); II_PROJECT_ROOT = $root
    II_MODEL_ROOT = $runRoot; II_MODEL_ID = 'model'; II_MODEL_NAME = 'dino_memory'
    II_CATEGORY = $category; II_DEVICE = 'cuda'
    II_ARTIFACT_ROOT = (Join-Path $root ".tmp\external-demo\$Dataset")
    OMP_NUM_THREADS = '2'; MKL_NUM_THREADS = '2'; HF_HUB_OFFLINE = '1'
}
$arguments = @('-m', 'industrial_inspection.cli', '--config', (Join-Path $root 'configs\deployment\local.yaml'))
if ($Action -eq 'infer') {
    if (-not $Image -or -not (Test-Path -LiteralPath $Image -PathType Leaf)) { throw 'Supply an existing -Image.' }
    $arguments += @('infer', '--image', (Resolve-Path -LiteralPath $Image).Path, '--model-name', 'dino_memory', '--model-dir', $model, '--category', $category)
} else {
    if ($Image) { throw '-Image applies only to infer.' }
    $arguments += @('serve', '--host', '127.0.0.1', '--port', "$Port")
}
if ($ShowPlan) { @{ python = $python; arguments = $arguments; environment = $settings } | ConvertTo-Json -Depth 5; return }
$previous = @{}
try {
    foreach ($key in $settings.Keys) {
        $previous[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $settings[$key], 'Process')
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "External demo exited with $LASTEXITCODE" }
} finally {
    foreach ($key in $previous.Keys) { [Environment]::SetEnvironmentVariable($key, $previous[$key], 'Process') }
}
