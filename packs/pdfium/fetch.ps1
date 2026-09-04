# Stage OUR prebuilt pdfium without building it. Windows twin of fetch.sh.
#
#   powershell -File packs\pdfium\fetch.ps1                    # win-x64, the host
#   powershell -File packs\pdfium\fetch.ps1 -Platform linux-x64
#
# Reads packs\pdfium\prebuilt.json, downloads the library for the platform,
# checks it against the manifest's sha256 and places it where
# crates\paddock-pdfium\build.rs looks: packs\pdfium\<platform>\. The recipe in
# build\ is the provenance; this only saves the 15-minute Chromium build.
#
# Idempotent: a library that is already staged and matches the manifest is
# left alone. The download lands in a .part file and is renamed only after the
# hash matches, so a killed transfer never leaves a wrong-sized library for
# the linker to find.
param(
    [string]$Platform = 'win-x64'
)
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 renders a progress bar per received chunk, which turns
# a 30 MB download into minutes. Silence it; the transfer is the same.
$ProgressPreference = 'SilentlyContinue'

$Here = $PSScriptRoot
$manifest = Get-Content (Join-Path $Here 'prebuilt.json') -Raw | ConvertFrom-Json
$entry = $manifest.files | Where-Object { $_.platform -eq $Platform }
if (-not $entry) {
    throw "no entry for platform '$Platform' in packs\pdfium\prebuilt.json"
}

# VERSION is the source of truth for the pin; the manifest carries the same
# string. A mismatch means someone bumped one and not the other.
$pin = (Get-Content (Join-Path $Here 'VERSION') -Raw).Trim()
if ($pin -ne $manifest.version) {
    Write-Warning "packs\pdfium\VERSION is $pin but prebuilt.json is $($manifest.version)"
}

$destDir = Join-Path $Here $Platform
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$dest = Join-Path $destDir $entry.name

function Sha256Of([string]$Path) {
    (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if ((Test-Path $dest) -and ((Sha256Of $dest) -eq $entry.sha256)) {
    Write-Host "pdfium already staged: $dest ($($manifest.version))"
    exit 0
}

Write-Host "==> pdfium $($manifest.version) for $Platform"
Write-Host "    $($entry.url)"
$part = "$dest.part"
Invoke-WebRequest -Uri $entry.url -OutFile $part -UseBasicParsing

$got = Sha256Of $part
if ($got -ne $entry.sha256) {
    Remove-Item -Force $part
    throw "sha256 mismatch for $($entry.name): expected $($entry.sha256), got $got"
}
Move-Item -Force $part $dest
Write-Host "    staged $dest"
