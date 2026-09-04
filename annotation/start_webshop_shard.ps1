param(
    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 35)]
    [int]$Shard,

    [ValidateRange(0, 5000)]
    [int]$DelayMilliseconds = 0,

    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$benchmarkRoot = Split-Path -Parent $repoRoot
$shardTag = "{0:D3}" -f $Shard
$shardDir = Join-Path $repoRoot "data\simulation\webshop_v2_350_formal_priority_classified_shards"
$datasetPath = Join-Path $shardDir "shard_$shardTag.json"
$outputPath = Join-Path $shardDir "shard_${shardTag}_human_annotated.json"
$pythonPath = Join-Path $benchmarkRoot ".venv38-webshop\Scripts\python.exe"
$serverPath = Join-Path $PSScriptRoot "replay_server.py"
$logDir = Join-Path $PSScriptRoot "data"
$stdoutPath = Join-Path $logDir "replay_server_shard_$shardTag.stdout.log"
$stderrPath = Join-Path $logDir "replay_server_shard_$shardTag.stderr.log"

foreach ($requiredPath in @($datasetPath, $outputPath, $pythonPath, $serverPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
        throw "Required file not found: $requiredPath"
    }
}

if ($DelayMilliseconds -gt 0) {
    Start-Sleep -Milliseconds $DelayMilliseconds
}

$listener = Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    $listenerPid = @($listener | Select-Object -ExpandProperty OwningProcess -Unique)[0]
    $listenerProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$listenerPid"
    if ($listenerProcess.CommandLine -notlike "*annotation\replay_server.py*") {
        throw "Port 7861 belongs to another process: $($listenerProcess.CommandLine)"
    }
}
$replayProcesses = @(
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*annotation\replay_server.py*" -and
        $_.CommandLine -match "--port\s+7861(?:\s|$)"
    }
)
if ($replayProcesses.Count) {
    $replayProcessIds = @($replayProcesses | Select-Object -ExpandProperty ProcessId -Unique)
    Stop-Process -Id $replayProcessIds -Force -ErrorAction SilentlyContinue
}

$launchArguments = @(
    $serverPath,
    "--dataset", $datasetPath,
    "--output", $outputPath,
    "--skip_constraint_enrichment",
    "--port", "7861"
)
$serverProcess = Start-Process `
    -FilePath $pythonPath `
    -ArgumentList $launchArguments `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$ready = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $request = [System.Net.WebRequest]::Create("http://127.0.0.1:7861/api/candidates/0/0")
        $request.Proxy = $null
        $request.Timeout = 2000
        $response = $request.GetResponse()
        $response.Dispose()
        $ready = $true
        break
    }
    catch {
        Start-Sleep -Milliseconds 500
    }
}
if (-not $ready) {
    throw "Shard $Shard replay server did not become ready. See $stderrPath"
}

$pageUrl = "http://127.0.0.1:7861/?shard=$shardTag"
if (-not $NoBrowser) {
    Start-Process $pageUrl -WindowStyle Hidden
}
Write-Host "Opened WebShop replay Shard $Shard / 35"
Write-Host "Annotation output: $outputPath"
