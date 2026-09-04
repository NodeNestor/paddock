# Contributing to Paddock

Thanks for looking. Paddock is young and moves fast, so this page is short and
practical. The README says what the engine is and why; this says how to get a
change in.

## Where to start

- **Bugs and model support** are the most useful things to bring. Open an
  issue with the template; `paddock --version`, the GPU and driver, and the
  model file name are what make a report actionable.
- **Kernels** live under `packs/cuda/src/`, hand-written CUDA, no Python in
  the build. `pack.cu`'s include list is the one true order. A kernel change
  is judged by measurement on a named board, never by reasoning alone; see
  the GPU offer below if you do not have one.
- **The engine** is `crates/paddock-engine` (scheduler, paged KV, memory,
  model families), the serving edge is `crates/paddock-runner`, and the
  manager plus Studio is `crates/paddock-manager` with the Vue app in
  `studio/`. Each crate's `Cargo.toml` carries a one-line description.

Not sure where something belongs, or planning something large? Open an issue
first and say what you intend to do. It is cheap and it saves a rewrite.

## Building

Follow the README's Building section. Two things it is easy to miss:

- On Linux you also need `cmake` and a C/C++ compiler. The Opus codec used
  for audio is compiled from source at build time, and that build is cmake.
- The Rust build never needs a CUDA toolkit or a GPU. Only the kernel pack
  does, and the pack is loaded at runtime over a C ABI.

## Before you open a pull request

Run what CI runs. All of it works on a machine without a GPU.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets
cargo test --workspace
cd studio && npm ci && npm run build && npm test
```

Clippy must come back clean. The workspace denies `unwrap` in library code,
along with `dbg!`, `todo!` and `unimplemented!`; a new one is a build error,
not a warning. `expect` with the invariant named in the message is the
sanctioned form. Tests may `unwrap` freely.

Tests that need a GPU skip themselves with a notice when no kernel pack or
device is present. On a box that has both, set `PADDOCK_STRICT_GATES=1` to turn
every skip into a failure, and `PADDOCK_HEAVY_TESTS=1` to opt into the gates
that load a whole model. `crates/paddock-engine/tests/common/mod.rs` documents
how a pack and models are found.

If you touched a kernel or anything on the serving path, say in the pull
request what you measured, on which GPU and driver, and against what.

## Rules of the tree

- **No silent failures.** An error is reported, not swallowed. Truncation is an
  error, not a trim. This is the first principle in the README and it is the
  one reviewers hold hardest.
- **Line endings are correctness.** `.gitattributes` pins `.sh`, `.py` and
  Dockerfiles to LF and `.ps1` to CRLF. Do not add a global `text=auto`; the
  file explains why.
- **Dependencies stay current**, and the `cudarc` feature pin is
  load-bearing. `cuda-13000` is what holds the minimum driver at the 580
  branch; raising it is a release-note decision, never a routine bump. The
  comment in the root `Cargo.toml` has the reasoning. `siftx` and `scriptor`
  are git dependencies pinned by revision, so a bump is a deliberate change of
  the rev.
- **Windows and Linux, x64, CUDA.** There is no CPU path and no macOS build.
  Contributions for other backends are a conversation to have in an issue
  before code.

## Pull requests

- One change per pull request. Small ones merge fast; a mixed one waits for
  its slowest part.
- The subject line names the area and what changed, in the style of the
  history: `scheduler: ...`, `qwen3.8: ...`, `studio: ...`. The body says why,
  and what you verified.
- Rebase on `main` rather than merging `main` into your branch.

## GPUs for contributors

Contributors can get a private GPU from us: an RTX 5090, an RTX PRO 6000 or a
B200, for 24 hours to start and up to 31 days for regular contributors, with
full access including clocks and Nsight. Capacity is limited and first come,
first served. Open an issue titled `GPU access: <what you plan to work on>`
and say roughly how long you need.

## Licence

Paddock is dual-licensed under MIT and Apache-2.0. Unless you state otherwise,
anything you intentionally submit for inclusion is licensed the same way,
without additional terms. Third-party code you bring in must be compatible
and gets its notice in `THIRD-PARTY-NOTICES`.
