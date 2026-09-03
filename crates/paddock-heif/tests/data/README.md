# Where these two files come from

Both are from **libheif's own repository** (`fuzzing/data/corpus/`, tag
v1.23.1, LGPL-3.0-or-later - the licence text ships in
`packs/heif/licenses/libheif-COPYING.txt`). They are 32x32 synthetic test
patterns of a few hundred bytes each, not photographs, and carry no metadata
about anybody.

| file | codec | why it is here |
|---|---|---|
| `avif32.heif` | AV1, decoded by dav1d | proves the AVIF half of the pack is wired up |
| `hevc32.heif` | HEVC, decoded by libde265 | proves the HEIC half is |

**One fixture per decoder, deliberately.** libheif is one library over two
independent codec backends, and it does not fail to build when one of them is
missing - it prints `Not compiling 'dav1d' backend` and produces a library that
opens the file and refuses it at decode time. A single fixture would test one
backend and leave the other free to go missing silently on some future pin bump.

They are also the smallest real answer to "does our build actually decode
anything", which no amount of sniffing coverage can give: everything above the
`ftyp` box is the native library's business.
