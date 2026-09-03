# Builds the CUDA kernel pack as a MULTI-ARCH fatbin. Requires the CUDA toolkit
# (nvcc) and an MSVC developer environment (nvcc drives cl.exe on Windows).
#
#   powershell -File build.ps1                 # all default arches
#   powershell -File build.ps1 -Arches 86,90   # a subset (faster local iterate)
#
# Output: build/pd-cuda-sm86.dll  (name kept for compatibility; it is a fatbin
# holding SASS for every arch below plus forward-compat PTX, so one pack runs on
# Ampere/Ada/Hopper/Blackwell, not sm_86-only). PTX is forward-compatible only, so
# each GPU generation needs its own SASS here; the trailing compute_* PTX lets a
# NEWER GPU than we compiled for JIT the kernels.
#
# CI note: packs are not built by cargo (GPU-less CI skips them); pack-enabled
# runners and release pipelines call this script.

param(
    # SM targets to emit SASS for. Keep in sync with the GPUs we test on:
    #   80 = A100, 86 = A6000/RTX30, 89 = Ada/RTX40, 90 = Hopper H100,
    #   100 = GB200/B200, 103 = Blackwell Ultra, 110 = Jetson T-series,
    #   120 = Blackwell/RTX50, 121 = DGX Spark GB10. Trim for faster local
    #   builds.
    #
    # Must match build.sh's list. The fatbin carries no PTX, so a die absent
    # from here cannot load the pack at all - and this script once lagged its
    # Linux twin's list (no 103/110/121), so a Windows-built pack silently
    # could not serve three generations the Linux one could. Same drift that
    # left cutgemm uncompiled here for weeks.
    #
    # [string[]], not [int[]] - normalised below. `powershell -File` hands every
    # argument over as a STRING, so `-Arches 86,100,120` (the form this file's
    # own header documents, and the form the release script prints when it
    # tells you to rebuild) arrived as the single string "86,100,120", which
    # PowerShell then cast to an int by reading the commas as THOUSANDS
    # SEPARATORS: one arch, sm_86100120. That is the
    # second time this parameter has silently eaten an arch list; the first
    # was the space-separated `-Arches 86 100 120` spelling. Taking strings and
    # splitting them ourselves makes every spelling mean the same thing, and
    # matches build.sh, which has always taken one comma-separated token.
    [string[]] $Arches = @('80', '86', '89', '90', '100', '103', '110', '120', '121'),
    # Emit a STATIC library (build/pd-cuda.lib) instead of the DLL, for
    # `cargo build --features static-pack` to link into paddock-runner
    # The.lib is a build input and is gitignored like the DLL -
    # there is no download for it, this script is the delivery.
    #
    # Orthogonal to -Arches deliberately. A RELEASE should pass the validated
    # set (`-Static -Arches 86,100,120`) so the binary and the engine's
    # gpu_support allowlist agree by construction; leaving that to the caller
    # keeps the choice visible in the release script instead of hidden behind
    # a linkage switch.
    [switch] $Static,
    # Arch whose PTX to embed for forward-compat onto future/unknown GPUs.
    # Parallel per-arch compilations. nvcc builds every -gencode target
    # SERIALLY by default, so the 8 targets below ran one cicc at a time -
    # measured at 99.8% of one core on a 32-core machine (3% of it) for
    # ~14 minutes. 0 = one thread per target, capped by cores.
    # Same generated code, just not one-at-a-time.
    [int] $Threads = 0
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force (Join-Path $here 'build') | Out-Null

# One spelling, whichever way the caller wrote it: `-Arches 86,100,120` through
# -File (one string, commas inside), `-Arches @(86,100,120)` through -Command
# (three elements), or `-Arches 86` (one). Split every element, then parse.
# A non-numeric token is a typo, and a typo that silently drops an arch is the
# failure this parameter keeps having - so it stops rather than skips.
$Arches = @($Arches | ForEach-Object { $_ -split '[,\s]+' } | Where-Object { $_ } | ForEach-Object {
    if ($_ -notmatch '^\d+$') { throw "not an SM number: '$_' (expected e.g. -Arches 86,100,120)" }
    [int]$_
})
if ($Arches.Count -eq 0) { throw "-Arches parsed to nothing" }

# Only emit arches this toolkit actually supports (older nvcc lacks sm_100/120).
$supportedRaw = (& nvcc --list-gpu-arch) | ForEach-Object { ($_ -replace 'compute_', '') } | Where-Object { $_ }
$supported = $supportedRaw | Where-Object { $_ -match '^\d+$' } | ForEach-Object { [int]$_ }
$gencode = @()
$defines = @()
$bsHost = $false
foreach ($a in $Arches) {
    if ($supported -contains $a) {
        $gencode += "-gencode=arch=compute_$a,code=sm_$a"
        # --list-gpu-arch omits feature targets; any toolkit that knows
        # compute_120 (CUDA >= 12.8) accepts the 120a feature target too
        if ($a -eq 120) {
            # the 'a' feature target carries the block-scale (mxf8f6f4) MMA;
            # without it + PD_BS_HOST the pack exports NULL for those entries
            # and the engine keeps the s8 mmq path (see pack.cu).
            $gencode += "-gencode=arch=compute_120a,code=sm_120a"
            $bsHost = $true
        }
        # sm_100 (B200): mirrors build.sh (an audit caught this script
        # shipping SM100 without its own TCGEN05 route).
        # The 100a feature target carries tcgen05 (tensor-memory MMA) for the
        # rowwise-e4m3 GEMM; PD_TC5_HOST turns its launcher route on, and
        # PD_BS_HOST makes the f8w8-family launchers real (per-device nulling
        # in paddock_pack_kernels_v1 keeps sm_120a-only families honest).
        if ($a -eq 100) {
            $gencode += "-gencode=arch=compute_100a,code=sm_100a"
            $defines += "-DPD_TC5_HOST=1"
            $bsHost = $true
        }
    } else {
        Write-Warning "toolkit does not support sm_$a, skipping"
    }
}
if ($bsHost) { $defines += "-DPD_BS_HOST=1" }
# No trailing PTX: JIT-limping onto a GPU generation we
# never validated is exactly the unknown-performance serve the engine's
# validated-arch allowlist exists to refuse. A new generation gets a proper
# bring-up and an explicit SASS target, not a fallback.
if ($gencode.Count -eq 0) { throw "no supported arches to build" }

# cutgemm: the vendored CUTLASS sm100 fp8 GEMM is its own TU so the
# CUTLASS headers never touch the multi-arch pack.cu compile. Built for real
# only when an sm_100 target is asked for AND the headers are here; otherwise
# it compiles to `cudaErrorNotSupported` stubs, which pack.cu still has to LINK
# against - leaving it out is three unresolved externals and no pack at all
# (build.sh has done this for longer; this script had not, so every Windows
# build was broken until it turned up while chasing a missing kernel).
# /MT for the static lib, and only for it. Rust's Windows binaries link the
# static CRT (+crt-static in .cargo/config.toml) because pdfium is built /MT
# and mixing CRTs gives the process two heaps - the same trap, so the kernels
# have to agree. nvcc defaults to /MD, which would link but corrupt. The DLL
# keeps the default: it is self-contained and never shares a heap with us.
# /Zc:preprocessor is not optional and not about the CRT: CUDA 13's
# <cooperative_groups.h> - pulled into the fatbin by gemm/f32_qkv.cuh since the
# qwen4exp lane landed - includes CCCL, and CCCL hard-#errors on
# MSVC's traditional preprocessor. It aborts the pack.cu compile outright, so
# there is no .dll and no .lib at the end, only a throw. gcc has no such check,
# which is why build.sh never needed this and why a Linux-side push cannot see
# the break - it is Windows-only by construction. Satisfy the check rather than
# define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING: that silences the
# diagnostic and keeps the non-conforming preprocessor CCCL is warning about.
$crt = @('-Xcompiler', '/Zc:preprocessor')
if ($Static) { $crt += @('-Xcompiler', '/MT') }
# PD_STATIC drops __declspec(dllexport) from every launcher - see abi.cuh. An
# archive is resolved by address at link time, so exporting 430 kernel names
# from the consuming exe buys nothing.
if ($Static) { $defines += '-DPD_STATIC=1' }

# Its own object per linkage: the static lane compiles it /MT, and leaving a
# /MT object under the shared name would quietly get linked into the next DLL
# build somebody else runs.
$cutleaf = if ($Static) { 'build/cutgemm-mt' } else { 'build/cutgemm' }
$cutobj = Join-Path $here "$cutleaf.obj"
$cutFlags = @()
$cutInc = $env:PD_CUTLASS_INC
if (($Arches -contains 100) -and $cutInc -and (Test-Path "$cutInc\include")) {
    $cutFlags += @(
        "-DPD_CUTGEMM=1", "-I$cutInc\include", "-I$cutInc\tools\util\include",
        "-gencode=arch=compute_100a,code=sm_100a", "--expt-relaxed-constexpr"
    )
    Write-Host "building cutgemm TU: $($cutFlags -join ' ')"
} else {
    # The STUB still needs our arch list. Without any -gencode nvcc emits its
    # default target, which on CUDA 13 is sm_75 - so every pack ever built has
    # carried a dead Turing cubin from this one TU (found by cuobjdump'ing an
    # sm_86-ONLY build and finding sm_75 in it). Harmless, in
    # that preflight refuses pre-Ampere long before anything could launch it,
    # but it made the fatbin's arch list a lie.
    $cutFlags += $gencode
    Write-Host "building cutgemm TU: (stub - no CUTLASS headers or no sm_100 target)"
}
& nvcc -O3 -std=c++17 @cutFlags @crt -c -o $cutobj (Join-Path $here 'src/gemm/cutgemm.cu')
if ($LASTEXITCODE -ne 0) { throw "nvcc failed on cutgemm.cu with exit code $LASTEXITCODE" }

# nv4cut (the checkpoint-native NVFP4 decode GEMM) is a SECOND CUTLASS TU, on
# the same terms as cutgemm: sm_100a only, CUTLASS headers kept out of the
# multi-arch pack.cu compile, NotSupported stubs when the tree is absent. Its
# own object per linkage/flavour, same reason. Ported from build.sh, which
# grew this TU first - the exact sh/ps1 drift these scripts' own comments warn
# about, found as unresolved symbols in the static link:
# pack.cu declared the entry points, no TU defined them.
$nv4obj = $cutobj -replace 'cutgemm', 'nv4cut'
Write-Host "building nv4cut TU: $($cutFlags -join ' ')"
& nvcc -O3 -std=c++17 @cutFlags @crt -c -o $nv4obj (Join-Path $here 'src/gemm/nv4cut.cu')
if ($LASTEXITCODE -ne 0) { throw "nvcc failed on nv4cut.cu with exit code $LASTEXITCODE" }

# -lib archives the objects; the DLL name is kept for compatibility (it is a
# fatbin either way, not sm_86-only). Both carry the same fatbin and the same
# two exports - only how the engine reaches them differs.
$link = '--shared'
$leaf = 'build/pd-cuda-sm86.dll'
if ($Static) { $link = '-lib'; $leaf = 'build/pd-cuda.lib' }
$out = Join-Path $here $leaf
Write-Host "building fatbin:" ($gencode -join ' ') ($defines -join ' ') "$link --threads $Threads"
# -std=c++17 matches the two object TUs above, which have always had it. It was
# absent here and did not matter until CCCL arrived: MSVC defaults to C++14, so
# _MSVC_LANG read 201402L and CCCL's dialect gate (#error "libcu++ requires at
# least C++ 17") aborted the fatbin. Passing it also stops pack.cu linking
# against cutgemm/nv4cut objects built in a different dialect.
& nvcc -O3 -std=c++17 --threads $Threads @gencode @defines @crt $link -o $out (Join-Path $here 'pack.cu') $cutobj $nv4obj

if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Write-Host "built (multi-arch): $out"

# Record what this archive can run on, beside it. With no PTX in the fatbin the
# arch list is the hardware list, and once the kernels are welded into a binary
# nobody can inspect them without the CUDA tools - so the packaging step reads
# this to stamp VERSION.txt, and whoever holds the folder can answer "will this
# serve my card" without owning nvcc.
if ($Static) {
    (($gencode | ForEach-Object { ($_ -split ',')[1] -replace 'code=', '' }) -join ' ') |
        Out-File (Join-Path $here 'build/pd-cuda.arches') -Encoding ascii
    # ...and which nvcc emitted them. The toolkit version is a deliberate,
    # separately-tracked decision from the driver floor (nvcc 13.3
    # for its codegen on the newest instruction families, while cudarc's
    # cuda-13000 pin holds every user's minimum driver at r580), so when a
    # codegen change is ever suspected, the shipped folder should be able to
    # answer "built by what" on its own.
    #
    # Written here and not read at packaging time, for the reason the whole
    # arches file exists: asking `nvcc --version` while packaging describes the
    # CHECKOUT's toolkit, which may have moved since this archive was built.
    # (A staged VERSION.txt once carried "nvcc 13.3.1" because a human typed
    # it in, and the next restage silently dropped it.)
    $ver = ((& nvcc --version) | Select-String -Pattern 'release\s+[\d.]+,\s*V([\d.]+)').Matches.Groups[1].Value
    if (-not $ver) { $ver = 'unknown' }
    $ver | Out-File (Join-Path $here 'build/pd-cuda.nvcc') -Encoding ascii
}
