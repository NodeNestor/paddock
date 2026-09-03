# paddock-bench

A throughput and latency harness. It reports prefill tokens/s, decode tokens/s
and time-to-first-token, either in-process against the engine or over HTTP
against any OpenAI-compatible endpoint.

```sh
# in-process, against a local model and kernel pack
cargo run -p paddock-bench --release -- \
  <model>.gguf --device cuda --pack packs/cuda/build/pd-cuda-sm86.dll \
  --prompt-tokens 128 --decode-tokens 64

# over HTTP, against a running server
cargo run -p paddock-bench --release -- \
  --endpoint http://127.0.0.1:11550/v1 --name paddock --endpoint-model <id>
```

The HTTP mode is deliberately the only way to measure another engine. Other
inference servers stay external black boxes: their frameworks are never linked
into this codebase, and the harness just speaks to their endpoint like any
client would. The same invocation therefore works against anything that serves
an OpenAI-compatible API.

Two things to know when reading its output. In-process mode uses a synthetic
prompt, token 0 repeated, which is content-independent for timing purposes, and
it discards warmup iterations. HTTP mode counts streamed token events, so
time-to-first-token and decode rate are real but prefill rate is not observable
from a client and is shown as `-`.
