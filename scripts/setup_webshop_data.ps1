[CmdletBinding()]
param(
    [ValidateSet("small", "all")]
    [string]$Dataset = "all",

    [string]$Python = "python",

    [switch]$BuildIndex,

    [switch]$Force
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$webShopRoot = Join-Path $repoRoot "WebShop"
$dataDir = Join-Path $webShopRoot "data"
$searchDir = Join-Path $webShopRoot "search_engine"

New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

function Get-WebShopDataFile {
    param(
        [Parameter(Mandatory)]
        [string]$FileId,

        [Parameter(Mandatory)]
        [string]$FileName
    )

    $destination = Join-Path $dataDir $FileName
    if ((Test-Path -LiteralPath $destination) -and -not $Force) {
        Write-Host "Keeping existing $destination"
        return
    }

    $url = "https://drive.google.com/uc?id=$FileId"
    Write-Host "Downloading $FileName"
    & $Python -m gdown --continue $url -O $destination
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to download $FileName"
    }
}

if ($Dataset -eq "all") {
    Get-WebShopDataFile -FileId "1A2whVgOO0euk5O13n2iYDM0bQRkkRduB" -FileName "items_shuffle.json"
    Get-WebShopDataFile -FileId "1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi" -FileName "items_ins_v2.json"
}
else {
    Get-WebShopDataFile -FileId "1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib" -FileName "items_shuffle_1000.json"
    Get-WebShopDataFile -FileId "1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu" -FileName "items_ins_v2_1000.json"
}

Get-WebShopDataFile -FileId "14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O" -FileName "items_human_ins.json"

if ($Dataset -eq "all") {
    $productFile = Join-Path $dataDir "items_shuffle.json"
    $productHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $productFile).Hash
    Write-Host "items_shuffle.json SHA-256: $productHash"
}

if ($BuildIndex) {
    if ($Dataset -ne "all") {
        throw "-BuildIndex currently builds the full index and requires -Dataset all."
    }

    $previousDataset = $env:WEBSHOP_DATASET
    $previousJavaOptions = $env:JAVA_TOOL_OPTIONS
    try {
        $env:WEBSHOP_DATASET = "all"
        if (-not $env:JAVA_TOOL_OPTIONS) {
            $env:JAVA_TOOL_OPTIONS = "-Xmx12g"
        }

        Push-Location $searchDir
        try {
            & $Python .\convert_product_file_format.py --output resources
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to generate WebShop search resources."
            }

            & $Python -m pyserini.index.lucene `
                --collection JsonCollection `
                --input resources `
                --index indexes `
                --generator DefaultLuceneDocumentGenerator `
                --threads 1 `
                --storePositions `
                --storeDocvectors `
                --storeRaw
            if ($LASTEXITCODE -ne 0) {
                throw "Failed to build the WebShop Lucene index."
            }
        }
        finally {
            Pop-Location
        }
    }
    finally {
        $env:WEBSHOP_DATASET = $previousDataset
        $env:JAVA_TOOL_OPTIONS = $previousJavaOptions
    }
}

Write-Host "WebShop data setup complete."
