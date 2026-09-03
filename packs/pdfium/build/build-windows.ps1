# Build pdfium from source as one static library for win-x64, and stage it plus
# its licences into packs\pdfium\.
#
#   powershell -File packs\pdfium\build\build-windows.ps1 [-Root DIR] [-Version chromium/NNNN]
#                                                         [-SkipSync] [-SkipBuild]
#
# Why we build it instead of downloading bblanchon's prebuilt: paddock ships two
# binaries and nothing else. A static pdfium links into
# paddock-runner.exe - no sidecar DLL, nothing written to disk at first use, and
# the linker drops the ~95% of pdfium we never call. lector already builds
# pdfium this way for WASM (c:\dev\lector packages/pdfium-wasm/build); this is
# the same recipe pointed at the native toolchain instead of emscripten.
#
# Windows cannot use the Docker leg - Chromium's Windows build needs depot_tools
# plus a real MSVC install, so this one runs natively on the dev box. Linux is
# built by build-linux.sh in a container.
#
# The source tree lives outside the repo (default: a pdfium-build directory
# beside the checkout): it is several GB after gclient sync.
param(
    [string]$Root = '',
    # The DEFAULT here is not the SOURCE of TRUTH - packs/pdfium/VERSION is, and
    # prebuilt.json carries the same string. Keep all three in step when bumping.
    #
    # The pin is free to be whatever pdfium release we want now. paddock-pdfium
    # bindgens the FFI from the headers of the tree this script builds, so the
    # bindings are that vintage; there is no separate declaration list to
    # outrun.
    #
    # It used to be capped, and the history is worth keeping because the failure
    # was invisible: we used to bind pdfium through pdfium-render, whose
    # hand-written per-vintage declarations are a vtable under a STATIC
    # link - every FPDF_* must resolve at link time, so we could only build a
    # vintage that crate declared. The pin sat at 7763 for months purely because
    # Cargo.lock sat at pdfium-render 0.9.2 while 0.9.3 (lifting the cap to
    # 7881) had already shipped, and nothing anywhere said so. That is how the
    # version of a memory-unsafe C++ parser fed arbitrary user PDFs ended up set
    # by a third party's release cadence. Do not reintroduce a binding crate
    # without noticing you have taken that ceiling back.
    #
    # Bumping is: change VERSION, run this script and build-linux.sh, publish
    # both, replace prebuilt.json with the manifest that run emits.
    [string]$Version = 'chromium/8009',
    [switch]$SkipSync,
    [switch]$SkipBuild
)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path "$PSScriptRoot\..\..\..").Path
if (-not $Root) { $Root = Join-Path (Split-Path $Repo -Parent) 'pdfium-build' }

$DepotTools = Join-Path $Root 'depot_tools'
$Src = Join-Path $Root 'pdfium'
$OutDir = 'out/static'   # gn wants forward slashes

if (-not (Test-Path $DepotTools)) {
    throw "depot_tools missing at $DepotTools - run: git clone --depth 1 https://chromium.googlesource.com/chromium/tools/depot_tools.git `"$DepotTools`""
}

# Use the LOCAL Visual Studio rather than Google's internal toolchain package
# (which we have no access to). vs_toolchain.py then finds VS through vswhere.
$env:DEPOT_TOOLS_WIN_TOOLCHAIN = '0'

# ...and PIN which Visual Studio, because vswhere answers with the NEWEST
# install and a machine can easily have two. Unpinned, pdfium compiled against
# VS18 Insiders' STL headers (toolset 14.51.36231) while cargo linked against
# VS2022's libraries (14.44.35207), and the link died on four STL helpers that
# exist in one and not the other:
#
#   unresolved external symbol __std_rotate
#   unresolved external symbol __std_min_element_f_ / __std_max_element_f_
#   unresolved external symbol __std_find_last_not_ch_pos_1
#
# Those are the out-of-line halves of <algorithm> and <string>: header-inlined
# in the newer STL, real functions in libcpmt.lib - so the mismatch cannot show
# up until the final link, long after pdfium.lib built cleanly. This is the
# same VS18-shadows-VS2022 hazard records for the CUDA pack, reaching
# in through vswhere instead of through PATH.
#
# Override it to the same Visual Studio cargo will link with. GYP_MSVS_VERSION
# has to agree or vs_toolchain.py re-derives the path and ignores the override.
$vs2022 = 'C:\Program Files\Microsoft Visual Studio\2022\Professional'
if (Test-Path $vs2022) {
    $env:GYP_MSVS_OVERRIDE_PATH = $vs2022
    $env:GYP_MSVS_VERSION = '2022'
    Write-Host "  toolchain: pinned to $vs2022"
} else {
    Write-Warning "VS2022 Professional not at $vs2022 - pdfium will build against whatever vswhere returns, which may not be what cargo links against"
}
# depot_tools must come first: it ships the python3/git/gn/ninja the build
# expects, and a system python on PATH ahead of it breaks the hooks.
$env:PATH = "$DepotTools;$env:PATH"

Write-Host "pdfium build (win-x64)"
Write-Host "  version: $Version"
Write-Host "  source:  $Src"
Write-Host "  repo:    $Repo"

if (-not $SkipSync) {
    New-Item -ItemType Directory -Force $Root | Out-Null

    # gclient needs a solution file beside the checkout. `checkout_configuration:
    # small` skips the test corpora and reference binaries - hundreds of MB of
    # things a build does not read.
    $gclient = Join-Path $Root '.gclient'
    if (-not (Test-Path $gclient)) {
        @(
            'solutions = ['
            '  {'
            '    "name": "pdfium",'
            '    "url": "https://pdfium.googlesource.com/pdfium.git",'
            '    "managed": False,'
            '    "custom_deps": {},'
            '    "custom_vars": {'
            '      "checkout_configuration": "small",'
            '    },'
            '  },'
            ']'
        ) | Out-File $gclient -Encoding ascii
        Write-Host "wrote $gclient"
    }

    if (-not (Test-Path $Src)) {
        Write-Host "cloning pdfium ..."
        git clone https://pdfium.googlesource.com/pdfium.git $Src
        if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
    }
    Write-Host "checking out $Version ..."
    git -C $Src fetch origin $Version
    if ($LASTEXITCODE -ne 0) { throw "git fetch $Version failed" }
    git -C $Src checkout FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }

    Push-Location $Root
    try {
        # --nohooks first so a sync failure is separable from a hook failure;
        # the hooks are what fetch clang, gn and ninja.
        Write-Host "gclient sync ..."
        & gclient.bat sync --no-history --shallow --nohooks
        if ($LASTEXITCODE -ne 0) { throw "gclient sync failed" }
        Write-Host "gclient runhooks ..."
        & gclient.bat runhooks
        if ($LASTEXITCODE -ne 0) { throw "gclient runhooks failed" }
    } finally { Pop-Location }
}

if (-not $SkipBuild) {
    $argsDst = Join-Path $Src "out\static"
    New-Item -ItemType Directory -Force $argsDst | Out-Null
    Copy-Item -Force (Join-Path $PSScriptRoot 'args-win.gn') (Join-Path $argsDst 'args.gn')

    Push-Location $Src
    try {
        Write-Host "gn gen $OutDir ..."
        & gn.bat gen $OutDir
        if ($LASTEXITCODE -ne 0) { throw "gn gen failed" }
        Write-Host "ninja (this takes a while) ..."
        & ninja.bat -C $OutDir pdfium
        if ($LASTEXITCODE -ne 0) { throw "ninja failed" }
    } finally { Pop-Location }
}

if ($SkipBuild) {
    Write-Host "`n-SkipBuild: source is synced, nothing staged."
    exit 0
}

# --- stage the library ------------------------------------------------------
# pdf_is_complete_lib folds every dependent object into this one archive, so it
# is the only file the Rust link needs.
$lib = Join-Path $Src 'out\static\obj\pdfium.lib'
if (-not (Test-Path $lib)) {
    # Older/newer layouts have moved this; fail with what we did find rather
    # than a bare "not found".
    $found = Get-ChildItem -Recurse -Filter 'pdfium.lib' (Join-Path $Src 'out\static') -ErrorAction SilentlyContinue
    throw "pdfium.lib not at $lib`nfound instead: $($found.FullName -join ', ')"
}
$dst = Join-Path $Repo 'packs\pdfium\win-x64'
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item -Force $lib (Join-Path $dst 'pdfium.lib')

# --- stage the licences -----------------------------------------------------
# These used to arrive inside bblanchon's archive; building ourselves means the
# obligation is ours to meet from our checkout. The notices generator reads
# this directory. Platform-independent (both OS legs build the same pin), so it
# sits beside the per-platform dirs rather than inside one.
#
# Over-collect deliberately: crediting a component we did not end up linking is
# harmless, omitting one we did is a licence breach. The set is every LICENSE
# file in the checkout's third_party tree, plus pdfium's own.
$licDst = Join-Path $Repo 'packs\pdfium\licenses'
if (Test-Path $licDst) { Remove-Item -Recurse -Force $licDst }
New-Item -ItemType Directory -Force $licDst | Out-Null
Copy-Item -Force (Join-Path $Src 'LICENSE') (Join-Path $licDst 'pdfium-LICENSE.txt')
$n = 1
Get-ChildItem -Path (Join-Path $Src 'third_party') -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $component = $_.Name
    Get-ChildItem -Path $_.FullName -File -Filter 'LICENSE*' -ErrorAction SilentlyContinue | Select-Object -First 1 | ForEach-Object {
        Copy-Item -Force $_.FullName (Join-Path $licDst "$component-LICENSE.txt")
        $script:n++
    }
}
Set-Content -Path (Join-Path $Repo 'packs\pdfium\VERSION') -Value $Version -Encoding ascii

$size = [math]::Round((Get-Item (Join-Path $dst 'pdfium.lib')).Length / 1MB, 1)
$commit = (git -C $Src log -1 --format=%H)
Write-Host ""
Write-Host "pdfium.lib   $size MB   ->  $dst"
Write-Host "licences     $n files   ->  $licDst"
Write-Host "pinned       $Version  ($commit)"
