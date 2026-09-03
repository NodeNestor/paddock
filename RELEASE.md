# Paddock 0.1.3

A feature release. Windows x64 and Linux x64, NVIDIA GPUs, driver 580 or newer.

The theme is memory: this release gives a large amount of video memory back to
the cache, and adds a place for the cache to overflow to when it still runs out.

## New

- **KV cache offloading to RAM and disk.** When a conversation is evicted from
  video memory its cache is now demoted to system RAM, and optionally to a
  disk store, instead of being thrown away - so returning to an older chat
  re-uses what was already computed rather than re-reading the whole prompt.
  The disk tier survives a restart, retires its oldest entries under quota
  pressure, and holds itself to a daily write budget so it cannot wear an SSD.
  Off by default; a switch in the Advanced tab, and the fit estimate prices
  what it costs the box.

- **fp8 KV cache on every supported GPU.** Halving the bytes each cached token
  costs was previously limited to cards with fp8 tensor cores. It needs fp8
  *storage*, not fp8 math, so it is now available everywhere - including
  Ampere - roughly doubling the context that fits in a given cache.

- **Gemma QAT checkpoints serve natively.** Google's quantization-aware-trained
  Gemma files store their weights at Q4_0, which is what the model was trained
  at rather than a lossy conversion of it. Paddock now serves those tensors
  directly instead of refusing the file.

## Improved

- **A large amount of video memory came back.** Several model families were
  keeping two copies of the same weights resident - one for each of two
  execution lanes. On Qwen 3.8 27B the memory held outside the cache fell from
  **31.8 GiB to 6.9 GiB**; Gemma 4's attention weights fell from 4.95 GB to
  0.82 GB. On a 96 GB card that moved Qwen 3.8's planned cache from the
  smallest of the three major serving engines to the largest. Everything freed
  becomes context.

- **Speculative decoding is more selective.** The controller now pools its
  acceptance measurements across speculation depths instead of learning each
  one separately, and speculation is switched off in the cases where it
  measurably loses to not speculating at all.

- **The fit estimate stopped overstating.** "This model" was counting memory
  that belongs to other things on the card; a reserved ceiling was being
  reported as memory currently in use; and when something does not fit, the
  estimate now says what is short and against what, instead of blaming the
  card.

- Dependencies swept to current across the tree - Rust 1.98, MCP revision
  2026-07-28, and a newer SQLite binding.

## Fixed

- Embedding requests failed with an out-of-memory error at around 4,000 tokens
  on Qwen3-Embedding-8B. The internal size buckets were 64x coarser than
  intended, so a modest request reserved an enormous one.
- On GPUs older than Ada, two attention paths were selected for the fp8 cache
  that have no implementation on that hardware, and they produced wrong output
  rather than refusing. They are no longer selected there.
- Asking for native fp8 weights from a directory that cannot be read now fails
  with an error naming the problem, instead of silently serving the 8-bit
  fallback and reporting success.
- On servers configured with a large maximum context, working memory was sized
  from that maximum rather than from what a step actually uses, which left no
  room for the prefix cache and silently disabled it.
- Qwen 3.5 produced corrupt output on short prompts under one of the projection
  paths (a numeric bug in the small-batch epilogue).
