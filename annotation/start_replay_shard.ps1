param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [int]$CurrentProcessId,

    [int]$Port = 7860,
    [string]$HostAddress = "127.0.0.1",
    [ValidateRange(0, 5000)]
    [int]$DelayMilliseconds = 0,
    [switch]$SkipConstraintEnrichment,
    [switch]$SkipFullCatalog,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$benchmarkRoot = Split-Path -Parent $repoRoot
$pythonPath = Join-Path $benchmarkRoot ".venv38-webshop\Scripts\python.exe"
$serverPath = Join-Path $PSScriptRoot "replay_server.py"
$resolvedDataset = (Resolve-Path -LiteralPath $DatasetPath).Path
$resolvedOutputParent = (Resolve-Path -LiteralPath (Split-Path -Parent $OutputPath)).Path
$resolvedOutput = Join-Path $resolvedOutputParent (Split-Path -Leaf $OutputPath)

foreach ($requiredPath in @($resolvedDataset, $pythonPath, $serverPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

if ($DelayMilliseconds -gt 0) {
    Start-Sleep -Milliseconds $DelayMilliseconds
}

$currentProcess = Get-Process -Id $CurrentProcessId -ErrorAction SilentlyContinue
if ($currentProcess) {
    if ($currentProcess.ProcessName -notmatch '^python') {
        throw "Refusing to stop non-Python process $CurrentProcessId"
    }
    Stop-Process -Id $CurrentProcessId -Force
}

$launchArguments = @(
    $serverPath,
    "--dataset", $resolvedDataset,
    "--output", $resolvedOutput,
    "--host", $HostAddress,
    "--port", [string]$Port
)
if ($SkipConstraintEnrichment) { $launchArguments += "--skip_constraint_enrichment" }
if ($SkipFullCatalog) { $launchArguments += "--skip_full_catalog" }

Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $launchArguments `
    -WorkingDirectory $repoRoot `
    -WindowStyle Hidden

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $status = Invoke-RestMethod -Uri "http://${HostAddress}:$Port/api/shard_status" -TimeoutSec 2
        if ($status.ok) {
            $ready = $true
            break
        }
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $ready) {
    throw "Replay shard did not become ready on ${HostAddress}:$Port"
}

if (-not $NoBrowser) {
    Start-Process "http://${HostAddress}:$Port/" -WindowStyle Hidden
}
