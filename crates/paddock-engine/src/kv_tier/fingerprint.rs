//! Content identity for the cache namespace: which weights and which
//! tokenizer these cached activations came from.
//!
//! Why this exists at all is written in `digest.rs`'s `model_tensors` field
//! and in the bug it cites: SGLang shipped silent cross-run corruption by
//! leaving a field out of its cache key (#33268). Before the durable tier
//! the cache was per-process and died with the runner, so geometry alone was
//! a sufficient
//! key - the comment in every family's `build_tier` said so, and said it
//! would stop being true "with T2 persistence, where a stale cache
//! could cross runs". That is now: the store survives restarts, and eviction
//! publishes the durable copy as readable *within* a run.
//!
//! The failure it prevents is concrete. Weight quantization does not change
//! KV geometry, so `Qwen3.6-27B` at Q8_0 and the same model at UD-Q4_K_XL
//! produce an identical architecture string, an identical namespace, and the
//! same directory on disk. Point a port at the other file and the new run
//! adopts the old run's activations - different numbers, no error, no way for
//! the user to see it. Same for a fine-tune at identical shape, or a
//! re-exported tokenizer.
//!
//! **This is a fingerprint, not a commitment.** Hashing a 130 GB GGUF at load
//! would cost a minute of startup per serve, so we hash what identifies the
//! file cheaply: the full tensor DIRECTORY (every name, dtype, shape and
//! offset - which alone separates every quant variant) plus bounded samples
//! of the tensor data itself, which separates checkpoints that differ only in
//! their values. The threat model is a user swapping files, not an adversary
//! crafting collisions; a deliberate second-preimage against a keyed blake3
//! is not the risk being managed here. Being wrong in the safe direction is
//! free: a fingerprint that changes when it did not need to costs one cold
//! cache, while one that fails to change costs correctness.

use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

/// Domain separators - a weights digest and a tokenizer digest must never be
/// able to collide with each other, or with any other blake3 use in the tree.
const CTX_WEIGHTS: &str = "paddock kv-tier weights fingerprint v1";
const CTX_TOKENIZER: &str = "paddock kv-tier tokenizer fingerprint v1";

/// Tensors sampled for their CONTENT. Spread across the sorted directory so
/// the sample covers early, middle and late layers rather than whatever the
/// writer happened to put first.
const SAMPLE_TENSORS: usize = 16;
/// Bytes read from each sampled tensor. 64 KiB is ~1 MB of mmap reads total -
/// microseconds warm, and far beyond any plausible accidental match.
const SAMPLE_BYTES: usize = 64 * 1024;

/// Hash one metadata value canonically. The encoding is explicit rather than
/// derived from `Debug` so a formatting change in the models crate can never
/// silently re-key every cache on disk.
fn hash_value(h: &mut blake3::Hasher, v: &Value) {
    match v {
        Value::U8(x) => {
            h.update(b"u8");
            h.update(&x.to_le_bytes());
        }
        Value::I8(x) => {
            h.update(b"i8");
            h.update(&x.to_le_bytes());
        }
        Value::U16(x) => {
            h.update(b"u16");
            h.update(&x.to_le_bytes());
        }
        Value::I16(x) => {
            h.update(b"i16");
            h.update(&x.to_le_bytes());
        }
        Value::U32(x) => {
            h.update(b"u32");
            h.update(&x.to_le_bytes());
        }
        Value::I32(x) => {
            h.update(b"i32");
            h.update(&x.to_le_bytes());
        }
        Value::F32(x) => {
            h.update(b"f32");
            h.update(&x.to_le_bytes());
        }
        Value::F64(x) => {
            h.update(b"f64");
            h.update(&x.to_le_bytes());
        }
        Value::U64(x) => {
            h.update(b"u64");
            h.update(&x.to_le_bytes());
        }
        Value::I64(x) => {
            h.update(b"i64");
            h.update(&x.to_le_bytes());
        }
        Value::Bool(x) => {
            h.update(b"bool");
            h.update(&[u8::from(*x)]);
        }
        Value::Str(s) => {
            h.update(b"str");
            h.update(&(s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        }
        Value::Array(items) => {
            h.update(b"arr");
            h.update(&(items.len() as u64).to_le_bytes());
            for it in items {
                hash_value(h, it);
            }
        }
    }
}

/// Content identity of the loaded weights. Cheap, deterministic, and stable
/// across moves and renames (the path is deliberately not an input - a file
/// that moved is the same weights).
pub fn weights(map: &MappedGguf) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key(CTX_WEIGHTS);
    h.update(&(map.shard_count() as u64).to_le_bytes());
    h.update(&map.total_len().to_le_bytes());

    // The tensor directory, in a stable order. Two files that differ in any
    // tensor's dtype, shape, name or placement diverge here - which covers
    // every requant, every architecture change, and every split-family
    // reshard, without reading a byte of tensor data.
    let mut names: Vec<&str> = map.tensor_infos().map(|t| t.name.as_str()).collect();
    names.sort_unstable();
    h.update(&(names.len() as u64).to_le_bytes());
    for n in &names {
        let Some(t) = map.tensor_info(n) else {
            continue;
        };
        h.update(&(n.len() as u64).to_le_bytes());
        h.update(n.as_bytes());
        h.update(&t.raw_type.to_le_bytes());
        h.update(&t.offset.to_le_bytes());
        h.update(&(t.dims.len() as u64).to_le_bytes());
        for d in &t.dims {
            h.update(&d.to_le_bytes());
        }
    }

    // Sampled content, so two checkpoints that differ only in their VALUES -
    // a fine-tune of the same base at the same quant - do not share a cache.
    if !names.is_empty() {
        let stride = names.len().div_ceil(SAMPLE_TENSORS).max(1);
        for n in names.iter().step_by(stride).take(SAMPLE_TENSORS) {
            let Ok((_info, bytes)) = map.tensor_bytes(n) else {
                continue;
            };
            let take = bytes.len().min(SAMPLE_BYTES);
            h.update(&(take as u64).to_le_bytes());
            h.update(&bytes[..take]);
        }
    }
    *h.finalize().as_bytes()
}

/// Identity of the tokenizer and chat template the prompt bytes went through.
/// Token ids are meaningless across tokenizer revisions, and a template
/// change reshapes the prompt without touching a single weight - either one
/// makes a cached prefix mean something different.
pub fn tokenizer(map: &MappedGguf) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key(CTX_TOKENIZER);
    let md = &map.gguf().metadata;
    let mut keys: Vec<&String> = md
        .keys()
        .filter(|k| k.starts_with("tokenizer.") || k.as_str() == "general.chat_template")
        .collect();
    keys.sort_unstable();
    h.update(&(keys.len() as u64).to_le_bytes());
    for k in keys {
        h.update(&(k.len() as u64).to_le_bytes());
        h.update(k.as_bytes());
        if let Some(v) = md.get(k) {
            hash_value(&mut h, v);
        }
    }
    *h.finalize().as_bytes()
}

/// The safetensors analogue, for families whose checkpoint is a directory of
/// shards rather than a GGUF (nemotron's NVFP4 lane). Same shape of answer:
/// the tensor directory, which separates every dtype and layout variant, plus
/// bounded content samples for checkpoints that differ only in their values.
pub fn weights_safetensors(st: &paddock_models::safetensors::ShardedSafetensors) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key(CTX_WEIGHTS);
    h.update(b"safetensors");
    let mut names: Vec<&String> = st.names().collect();
    names.sort_unstable();
    h.update(&(names.len() as u64).to_le_bytes());
    for n in &names {
        let Some((t, _)) = st.bytes(n) else { continue };
        h.update(&(n.len() as u64).to_le_bytes());
        h.update(n.as_bytes());
        h.update(&(t.dtype as u32).to_le_bytes());
        h.update(&(t.begin as u64).to_le_bytes());
        h.update(&(t.end as u64).to_le_bytes());
        h.update(&(t.shape.len() as u64).to_le_bytes());
        for d in &t.shape {
            h.update(&(*d as u64).to_le_bytes());
        }
    }
    if !names.is_empty() {
        let stride = names.len().div_ceil(SAMPLE_TENSORS).max(1);
        for n in names.iter().step_by(stride).take(SAMPLE_TENSORS) {
            let Some((_t, bytes)) = st.bytes(n) else {
                continue;
            };
            let take = bytes.len().min(SAMPLE_BYTES);
            h.update(&(take as u64).to_le_bytes());
            h.update(&bytes[..take]);
        }
    }
    *h.finalize().as_bytes()
}

/// Tokenizer identity for a checkpoint DIRECTORY: the tokenizer files
/// themselves. `tokenizer.json` carries the whole vocabulary and merge table,
/// so its bytes are the identity; the config files beside it shape the
/// prompt the same way a GGUF chat template does.
pub fn tokenizer_dir(dir: &std::path::Path) -> [u8; 32] {
    let mut h = blake3::Hasher::new_derive_key(CTX_TOKENIZER);
    h.update(b"dir");
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
    ] {
        h.update(&(name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        match std::fs::read(dir.join(name)) {
            Ok(b) => {
                h.update(&(b.len() as u64).to_le_bytes());
                h.update(&b);
            }
            // absent is itself identity - a checkpoint that GAINS a chat
            // template is not the same prompt shape as one without it
            Err(_) => {
                h.update(&u64::MAX.to_le_bytes());
            }
        }
    }
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use paddock_models::gguf::Value;

    /// The canonical encoding must distinguish values a lazier one would
    /// flatten together - the whole point is that no two different metadata
    /// states hash alike.
    #[test]
    fn value_encoding_separates_look_alikes() {
        let mk = |v: &Value| {
            let mut h = blake3::Hasher::new_derive_key(CTX_TOKENIZER);
            hash_value(&mut h, v);
            *h.finalize().as_bytes()
        };
        // same digits, different types
        assert_ne!(mk(&Value::U32(7)), mk(&Value::I32(7)));
        assert_ne!(mk(&Value::U32(7)), mk(&Value::U64(7)));
        // string vs a one-element array of it
        assert_ne!(
            mk(&Value::Str("a".into())),
            mk(&Value::Array(vec![Value::Str("a".into())]))
        );
        // concatenation must not alias: ["ab"] vs ["a","b"]
        assert_ne!(
            mk(&Value::Array(vec![Value::Str("ab".into())])),
            mk(&Value::Array(vec![
                Value::Str("a".into()),
                Value::Str("b".into())
            ]))
        );
        // and equal values still agree with themselves
        assert_eq!(mk(&Value::Str("x".into())), mk(&Value::Str("x".into())));
    }

    #[test]
    fn nested_arrays_are_positional() {
        let mk = |v: &Value| {
            let mut h = blake3::Hasher::new_derive_key(CTX_WEIGHTS);
            hash_value(&mut h, v);
            *h.finalize().as_bytes()
        };
        let a = Value::Array(vec![Value::U8(1), Value::U8(2)]);
        let b = Value::Array(vec![Value::U8(2), Value::U8(1)]);
        assert_ne!(mk(&a), mk(&b), "order must matter");
    }
}
