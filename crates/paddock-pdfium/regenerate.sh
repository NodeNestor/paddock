#!/usr/bin/env bash
# Regenerate src/ffi.rs from the headers of the pdfium we build.
#
#   bash crates/paddock-pdfium/regenerate.sh [/path/to/pdfium/public]
#
# RUN this in the same COMMIT that MOVES packs/pdfium/VERSION. The bindings and
# the library are one thing: the whole reason this crate exists is that a
# hand-maintained binding could disagree with the library, which is what capped
# us at an old pdfium for months. Generating from the headers we actually build
# makes that disagreement impossible - but only if the two move together.
#
# The output is COMMITTED, so nobody needs bindgen or libclang to build paddock.
# The cost is this script, run by whoever bumps the pin.
#
#   cargo install bindgen-cli --locked
#   Windows also needs LLVM for libclang: LIBCLANG_PATH=C:/Program Files/LLVM/bin
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default matches build-windows.ps1's -Root: a pdfium-build directory beside
# the repo. The Linux leg builds in a container and leaves no tree behind, so
# pass the path if yours is elsewhere.
HEADERS="${1:-$HERE/../../../pdfium-build/pdfium/public}"

[ -f "$HEADERS/fpdfview.h" ] || {
    echo "no pdfium headers at $HEADERS" >&2
    echo "sync a tree first: powershell -File packs/pdfium/build/build-windows.ps1 -SkipBuild" >&2
    exit 1
}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
cat > "$tmp/wrapper.h" <<'EOF'
#include "fpdfview.h"
#include "fpdf_formfill.h"
EOF

# ALLOWLISTED, not everything. pdfium's public API is ~500 functions; we call
# 18. Generating the lot would bury the six operations this crate actually
# performs in noise and make every pin bump an unreviewable diff - the point of
# committing the output is that a human can see what the C ABI did.
bindgen "$tmp/wrapper.h" -o "$HERE/src/ffi.rs" \
    --allowlist-function 'FPDF_InitLibrary|FPDF_LoadMemDocument64|FPDF_CloseDocument|FPDF_GetPageCount|FPDF_LoadPage|FPDF_ClosePage|FPDF_GetPageWidthF|FPDF_GetPageHeightF|FPDF_RenderPageBitmap|FPDF_GetLastError|FPDFBitmap_CreateEx|FPDFBitmap_FillRect|FPDFBitmap_GetBuffer|FPDFBitmap_GetStride|FPDFBitmap_Destroy|FPDFDOC_InitFormFillEnvironment|FPDFDOC_ExitFormFillEnvironment|FPDF_FFLDraw' \
    --allowlist-var 'FPDFBitmap_BGRA|FPDF_ANNOT' \
    --allowlist-type 'FPDF_FORMFILLINFO' \
    --no-layout-tests --merge-extern-blocks \
    -- -I"$HEADERS" -x c++

echo "wrote $HERE/src/ffi.rs"
echo "  functions: $(grep -c 'pub fn FPDF' "$HERE/src/ffi.rs")"
echo
echo "Now: cargo test -p paddock-pdfium, and read the diff - a changed signature"
echo "is pdfium changing its ABI, which is exactly what you want to see."
