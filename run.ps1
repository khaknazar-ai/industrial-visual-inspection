#Requires -Version 7.0
[CmdletBinding()]
param(
    [ValidateSet('classical', 'patchcore', 'efficientad', 'dinov3', 'superadd', 'ron88', 'pcad')]
    [string]$Scenario = 'classical',
    [string]$Image,
    [switch]$Example,
    [switch]$List,
    [switch]$ShowPlan,
    [ValidateRange(1024, 65535)][int]$Port = 8000
)
$ErrorActionPreference = 'Stop'
if ($List) {
    @(
        'classical   MVTec bottle / Classical Diff / CPU (default)',
        'patchcore   MVTec bottle / PatchCore / CPU',
        'efficientad MVTec bottle / EfficientAD / CPU',
        'dinov3      MVTec bottle / DINOv3 memory / CPU',
        'superadd    LOCO splicing connectors / SuperADD / CPU',
        'ron88       External Ron88 bottles / DINOv3 memory / CUDA',
        'pcad        External PCAD switch assembly / DINOv3 memory / CUDA'
    )
    return
}
if ($Example -and $Image) { throw 'Choose either -Example or -Image.' }
if ($Example) {
    $Image = switch ($Scenario) {
        'superadd' { Join-Path $PSScriptRoot 'data\external\mvtec_loco\splicing_connectors\validation\good\057.png' }
        'pcad' { Join-Path $PSScriptRoot 'data\external\pcad_256\switch_8_port\train\good\000.png' }
        'ron88' {
            $sample = Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'data\external\ron88\images\val') -Filter 'good*.jpg' |
                Sort-Object Name | Select-Object -First 1
            if (-not $sample) { throw 'Ron88 example image is unavailable.' }
            $sample.FullName
        }
        default { Join-Path $PSScriptRoot 'data\external\p09\mvtec_ad\bottle\test\good\000.png' }
    }
}
$options = @{ Port = $Port; ShowPlan = $ShowPlan; Action = 'serve' }
if ($Image) {
    # Resolve user-relative paths before switching into the project directory.
    $options.Image = (Resolve-Path -LiteralPath $Image).Path
    $options.Action = 'infer'
}
Push-Location $PSScriptRoot
try {
    if ($Scenario -in @('ron88', 'pcad')) {
        & (Join-Path $PSScriptRoot 'scripts\start_external_demo.ps1') -Dataset $Scenario @options
    } else {
        & (Join-Path $PSScriptRoot 'scripts\start_local_demo.ps1') -Profile $Scenario @options
    }
} finally {
    Pop-Location
}
