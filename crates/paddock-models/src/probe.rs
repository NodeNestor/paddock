//! Model probing: turn a GGUF file into an honest, structured report.
//!
//! One report type serves every consumer - `paddock model inspect` (human and
//! --json), /v1/models enrichment, and the estimator's input. Reads a bounded
//! prefix, never the whole file.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::gguf::{GgufError, GgufFile, Value};

/// Header region cap. gpt-oss-20b's header (incl. 201k-token vocab) is ~13 MB;
/// 256 MB leaves room for far bigger vocabs without ever reading weight data.
const PROBE_PREFIX: u64 = 256 << 20;

#[derive(Debug, thiserror::Error)]
pub enum ProbeError {
    #[error("cannot read {0}: {1}")]
    Io(PathBuf, std::io::Error),
    #[error("{0}: {1}")]
    Parse(PathBuf, GgufError),
}

/// Per-quantization-type rollup.
#[derive(Debug, Clone, Serialize)]
pub struct QuantBucket {
    pub type_name: String,
    pub tensors: usize,
    /// None inside the sum means at least one tensor had an unverified layout;
    /// we report what we know and flag the rest, never guess.
    pub bytes: Option<u64>,
}

/// One transformer block's KV footprint, as read off the header.
///
/// The rule that makes this honest on hybrids: a block holds KV **iff it has a
/// K projection tensor**. qwen3.5/3.6 interleave Gated-DeltaNet blocks that
/// carry recurrent state instead of a growing cache - those blocks have no
/// `attn_k` and simply never land in this list, so a ctx-scaling estimate built
/// from it doesn't charge them. Nothing here is arch-specific beyond reading
/// the sliding-window pattern the arch declares.
#[derive(Debug, Clone, Serialize)]
pub struct KvLayer {
    /// Block index, as it appears in the `blk.N.` tensor prefix.
    pub layer: usize,
    /// Elements per token for K. V is tracked separately: they match in every
    /// architecture we serve today, but an asymmetric one shouldn't silently
    /// come out half-price.
    pub k_dim: u64,
    pub v_dim: u64,
    /// Sliding-window cap in tokens. `None` = this block holds full context.
    /// gemma4's 5:1 pattern means most blocks cap here and never grow with ctx.
    pub window: Option<u64>,
}

/// Per-slot state held by recurrent (Gated-DeltaNet / SSM) blocks.
///
/// This is the other half of a hybrid's memory story and it behaves the
/// opposite way to KV: fixed per sequence, *independent of context length*,
/// but paid for every concurrent slot. Leaving it out understates a wide
/// batch badly - qwen3.5-9B at batch 32 carries ~10 GB of it.
#[derive(Debug, Clone, Serialize)]
pub struct RecurrentShape {
    /// Backbone blocks carrying recurrent state instead of a KV cache.
    pub layers: u64,
    /// Recurrent matrix elements per block per slot (`n_v_heads * state^2`).
    pub state_elems: u64,
    /// Short-convolution window elements per block per slot.
    pub conv_elems: u64,
    /// Short-convolution channel width. Drives the prefill-span scratch, which
    /// is sized by span rather than by batch.
    pub conv_dim: u64,
    /// Element width - the engine keeps this state in f32; precision here is
    /// load-bearing, so it is not a knob.
    pub elem_bytes: u64,
}

/// Static cross-attention K/V held per slot by an encoder-decoder (whisper).
///
/// The THIRD memory class, after the growing KV cache and recurrent state,
/// and it behaves like the recurrent one: fixed per sequence, independent of
/// how long the transcript gets, paid for every concurrent slot. What makes
/// it big is that it is sized by the ENCODER WINDOW rather than by the
/// request - whisper's audio window is a constant 30 s = 1500 frames whether
/// the clip is four seconds or thirty, so a slot's cross cache is the same
/// size for every request and never shrinks.
///
/// At the large-v3 geometry this is 234 MiB per slot at f16 (117 at fp8) -
/// 77% of what a slot costs - so a will-it-fit that ignored it would be
/// wrong by a factor of four at any real concurrency.
#[derive(Debug, Clone, Serialize)]
pub struct CrossKv {
    /// Decoder blocks holding a cross-attention cache (all of them, for
    /// whisper - every decoder layer cross-attends to the encoder output).
    pub layers: u64,
    /// Encoder frames cached per layer: the fixed audio window.
    pub frames: u64,
    /// Elements per frame for K, and again for V - tracked separately for the
    /// same reason `KvLayer` does it.
    pub k_dim: u64,
    pub v_dim: u64,
}

/// Everything worth knowing about a model file before loading it.
#[derive(Debug, Clone, Serialize)]
pub struct ModelReport {
    pub path: PathBuf,
    /// Total bytes on disk - summed across shards for split families.
    pub file_size: u64,
    /// Files backing the model (1 unless it's a split family).
    pub shards: usize,
    pub gguf_version: u32,
    pub architecture: Option<String>,
    pub tensor_count: usize,
    pub quant_mix: Vec<QuantBucket>,
    // core geometry - None simply means the arch doesn't declare the key
    pub context_length: Option<u64>,
    pub block_count: Option<u64>,
    pub embedding_length: Option<u64>,
    pub head_count: Option<u64>,
    pub head_count_kv: Option<u64>,
    pub sliding_window: Option<u64>,
    pub expert_count: Option<u64>,
    pub expert_used_count: Option<u64>,
    pub tokenizer_model: Option<String>,
    pub token_count: Option<u64>,
    pub has_chat_template: bool,
    /// Blocks that hold a KV cache, in block order. Empty when the header
    /// declares no K projections at all (embedding/rerank exports) - which is
    /// a real answer, not a gap: those models cache nothing.
    pub kv_layers: Vec<KvLayer>,
    /// Trailing MTP/nextn draft blocks counted in `block_count` but not part of
    /// the backbone. Excluded from `kv_layers` - the drafter's cache is charged
    /// by the engine's separate speculative budget, not the serving envelope.
    pub nextn_blocks: u64,
    /// On-disk bytes of those blocks' tensors.
    ///
    /// They ship inside the weights file but are loaded only when the endpoint
    /// speculates (`qwen35/load.rs`: `if n_nextn > 0 && spec_wanted`), so file
    /// size alone cannot answer "what does this endpoint actually hold". This
    /// is the difference between spec on and spec off for an in-file-MTP model
    /// - the whole difference, since there is no separate drafter to charge.
    pub nextn_bytes: u64,
    /// Present only for hybrids that declare DeltaNet/SSM geometry.
    pub recurrent: Option<RecurrentShape>,
    /// Present only for encoder-decoders that cache the encoder output
    /// (whisper). See [`CrossKv`] - it is the dominant per-slot term there.
    pub cross_kv: Option<CrossKv>,
}

/// Read and parse one file's header region (never the weights).
fn parse_header(path: &Path) -> Result<(GgufFile, u64), ProbeError> {
    let io_err = |e| ProbeError::Io(path.to_path_buf(), e);
    let file = std::fs::File::open(path).map_err(io_err)?;
    let file_size = file.metadata().map_err(io_err)?.len();

    use std::io::Read;
    let mut head = vec![0u8; file_size.min(PROBE_PREFIX) as usize];
    std::io::BufReader::new(file)
        .read_exact(&mut head)
        .map_err(io_err)?;

    let f = GgufFile::parse_prefix(&head, file_size)
        .map_err(|e| ProbeError::Parse(path.to_path_buf(), e))?;
    Ok((f, file_size))
}

/// `blk.<n>.<rest>` -> `(n, rest)`. Anything else isn't a transformer block.
fn block_tensor(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, rest) = rest.split_once('.')?;
    Some((idx.parse().ok()?, rest))
}

/// Output width of a projection weight. GGUF stores 2-D weights as
/// `[in, out]` (ne0 is the fastest-moving = input row length), but writers
/// have disagreed, so key off the embedding width rather than trust the order:
/// whichever dim ISN'T n_embd is the projection's output. When both match
/// (MHA, where kv_dim == n_embd) either answer is the same one.
fn proj_out_dim(dims: &[u64], n_embd: Option<u64>) -> Option<u64> {
    match dims {
        [only] => Some(*only),
        [a, b] => match n_embd {
            Some(e) if *a == e => Some(*b),
            Some(e) if *b == e => Some(*a),
            _ => Some(*b),
        },
        _ => None,
    }
}

/// Is block `i` a sliding-window layer, for an architecture that declares a
/// window but no per-layer pattern?
///
/// Only architectures whose convention we can point at in our own loader belong
/// here - this table and the engine must agree, so each arm cites its source.
/// Anything unlisted returns None and the caller stays conservative.
fn swa_by_convention(arch: Option<&str>, i: usize) -> Option<bool> {
    match arch? {
        // gpt-oss alternates, even blocks sliding - see the engine's
        // `is_swa: i % 2 == 0` in gpu_model/gpt_oss.rs. Keep in step with it.
        "gpt-oss" => Some(i.is_multiple_of(2)),
        // laguna repeats [full, SWA, SWA, SWA] - full attention where
        // i % 4 == 0. See the engine's `is_swa: i % 4 != 0` in
        // gpu_model/laguna/load.rs. Keep in step with it.
        "laguna" => Some(!i.is_multiple_of(4)),
        _ => None,
    }
}

/// Per-block KV geometry from the header alone.
fn kv_layers_of(f: &GgufFile, n_embd: Option<u64>, backbone_blocks: u64) -> Vec<KvLayer> {
    // gemma4 declares which blocks are sliding-window as a per-layer bool
    // array; muse-glimmer declares a scalar PERIOD; gpt-oss declares only the
    // window and leaves the pattern to convention. A window we can't place on
    // specific blocks is deliberately not applied - for a will-it-fit, the safe
    // direction to be wrong in is over-estimating, never under.
    let swa_pattern: Option<Vec<bool>> = match f.arch_field("attention.sliding_window_pattern") {
        Some(Value::Array(a)) => Some(a.iter().map(|v| matches!(v, Value::Bool(true))).collect()),
        // Scalar period, muse-glimmer's spelling. Mirrors the engine's
        // `gpu_model/gemma4/load.rs::swa_pattern`, which is itself pinned
        // against llama-hparams.cpp `set_swa_pattern(n, dense_first=false)` and
        // muse's own config.json `layer_types` - the last block of each group
        // is the full-attention one, so `il % n < n - 1` is sliding. Reading it
        // here is data-driven rather than another `swa_by_convention` arch arm,
        // so any family shipping the scalar form is priced correctly.
        //
        // Inverting the phase would still "work" and just answer wrong, which
        // is why both sides cite the reference instead of inferring it.
        Some(v) => v.as_u64().map(|period| {
            let n = backbone_blocks as usize;
            match period as usize {
                // llama.cpp treats period 0 as "every layer sliding"
                0 => vec![true; n],
                p => (0..n).map(|il| il % p < p - 1).collect(),
            }
        }),
        _ => None,
    };
    let arch = f.architecture();
    let window = f
        .arch_field("attention.sliding_window")
        .and_then(Value::as_u64)
        .filter(|&w| w > 0);

    let v_dim_of = |i: usize| -> Option<u64> {
        f.tensors
            .iter()
            .find(|t| block_tensor(&t.name) == Some((i, "attn_v.weight")))
            .and_then(|t| proj_out_dim(&t.dims, n_embd))
    };

    let mut out: Vec<KvLayer> = f
        .tensors
        .iter()
        .filter_map(|t| {
            let (i, rest) = block_tensor(&t.name)?;
            if rest != "attn_k.weight" || (i as u64) >= backbone_blocks {
                return None;
            }
            let k_dim = proj_out_dim(&t.dims, n_embd)?;
            Some(KvLayer {
                layer: i,
                k_dim,
                v_dim: v_dim_of(i).unwrap_or(k_dim),
                window: window.filter(|_| {
                    swa_pattern
                        .as_ref()
                        .map(|p| p.get(i).copied().unwrap_or(false))
                        .or_else(|| swa_by_convention(arch, i))
                        .unwrap_or(false)
                }),
            })
        })
        .collect();
    out.sort_by_key(|l| l.layer);
    out
}

/// Whisper's geometry, which shares no metadata key with the llama-style
/// families: our converter writes `whisper.d_model`,
/// `whisper.decoder.layer_count`, `whisper.max_{source,target}_positions`
/// and so on (our whisper converter), and its tensors are named
/// `decoder.layers.0.self_attn.k_proj.weight` rather than `blk.0.attn_k`.
///
/// Without this the probe reported an all-null card with zero kv layers, and
/// the picker would have printed a will-it-fit verdict with no cache priced
/// at all - a silent failure, which the product principles ban outright.
///
/// Returns (context_length, block_count, embedding_length, head_count,
/// vocab, kv_layers, cross_kv) - all None/empty for a non-whisper file.
struct Whisper {
    ctx: u64,
    dec_layers: u64,
    d_model: u64,
    heads: u64,
    vocab: u64,
    audio_frames: u64,
}

fn whisper_geometry(f: &GgufFile) -> Option<Whisper> {
    if f.architecture()? != "whisper" {
        return None;
    }
    let u = |k: &str| f.arch_field(k).and_then(Value::as_u64);
    // Every one of these is required: a whisper file missing any of them is
    // not something to guess around, and a partial card is what this function
    // exists to prevent.
    Some(Whisper {
        ctx: u("max_target_positions")?,
        dec_layers: u("decoder.layer_count")?,
        d_model: u("d_model")?,
        heads: u("decoder.head_count")?,
        vocab: u("vocab_size")?,
        audio_frames: u("max_source_positions")?,
    })
}

pub fn probe_path(path: &Path) -> Result<ModelReport, ProbeError> {
    let (f, first_size) = parse_header(path)?;

    // split family probed via its first shard: pull in the siblings so the
    // report tells the truth about the whole model, not a third of it.
    // (Probing a lone/misnamed shard still reports that file as-is - the
    // probe describes what it was pointed at; the loader is where a broken
    // family becomes a hard error.)
    let mut shards = vec![(f, first_size)];
    let split_count = shards[0]
        .0
        .metadata
        .get("split.count")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if split_count > 1
        && let Some(name) = crate::split::parse_split_name(path)
        && name.no_1based == 1
        && u64::from(name.count) == split_count
    {
        for i in 2..=name.count {
            shards.push(parse_header(&name.sibling(i))?);
        }
    }
    let f = &shards[0].0;
    let file_size: u64 = shards.iter().map(|(_, size)| size).sum();

    // quant rollup in descending-bytes order - the story at a glance
    let mut by_type: BTreeMap<String, (usize, Option<u64>)> = BTreeMap::new();
    for t in shards.iter().flat_map(|(g, _)| g.tensors.iter()) {
        let e = by_type
            .entry(format!("{:?}", t.ggml_type))
            .or_insert((0, Some(0)));
        e.0 += 1;
        e.1 = match (e.1, t.byte_size()) {
            (Some(acc), Some(b)) => Some(acc + b),
            _ => None,
        };
    }
    let mut quant_mix: Vec<QuantBucket> = by_type
        .into_iter()
        .map(|(type_name, (tensors, bytes))| QuantBucket {
            type_name,
            tensors,
            bytes,
        })
        .collect();
    quant_mix.sort_by_key(|b| std::cmp::Reverse(b.bytes.unwrap_or(u64::MAX)));

    let arch_u64 = |suffix: &str| f.arch_field(suffix).and_then(Value::as_u64);
    let token_count = match f.metadata.get("tokenizer.ggml.tokens") {
        Some(Value::Array(a)) => Some(a.len() as u64),
        _ => None,
    };

    // block_count counts trailing MTP/nextn draft blocks too (27B: 64 backbone
    // + 1); the backbone is what the serving envelope is charged for.
    let nextn_blocks = arch_u64("nextn_predict_layers").unwrap_or(0);
    let embedding_length = arch_u64("embedding_length");
    let backbone_blocks = arch_u64("block_count")
        .unwrap_or(u64::MAX)
        .saturating_sub(nextn_blocks);
    // Bytes of the MTP block(s). Two spellings in the wild and both count:
    // a `nextn.*` prefix, and blocks indexed at or past the backbone's end
    // (`blk.<backbone..>.`) - the same trailing-block rule `backbone_blocks`
    // is derived from, so the two can't disagree about which block is which.
    let nextn_bytes: u64 = if nextn_blocks == 0 {
        0
    } else {
        shards
            .iter()
            .flat_map(|(g, _)| g.tensors.iter())
            .filter(|t| {
                t.name.starts_with("nextn.")
                    || t.name
                        .strip_prefix("blk.")
                        .and_then(|r| r.split_once('.'))
                        .and_then(|(i, _)| i.parse::<u64>().ok())
                        .is_some_and(|i| i >= backbone_blocks)
            })
            .filter_map(|t| t.byte_size())
            .sum()
    };
    // Tensor directories live per shard; KV geometry is read across all of them
    // so a split family reports every block, not just the first shard's.
    let kv_layers = {
        let mut all: Vec<KvLayer> = shards
            .iter()
            .flat_map(|(g, _)| kv_layers_of(g, embedding_length, backbone_blocks))
            .collect();
        all.sort_by_key(|l| l.layer);
        all.dedup_by_key(|l| l.layer);
        all
    };

    // Gated-DeltaNet geometry rides the overloaded ssm.* namespace, with two
    // keys used for something other than their name (see the qwen35 loader:
    // group_count = key heads, time_step_rank = value heads, not a dt rank).
    // Any backbone block without a KV cache is a recurrent one.
    let recurrent = match (
        arch_u64("ssm.state_size"),
        arch_u64("ssm.group_count"),
        arch_u64("ssm.time_step_rank"),
        arch_u64("ssm.conv_kernel"),
    ) {
        (Some(state), Some(n_k_heads), Some(n_v_heads), Some(conv_k))
            if backbone_blocks != u64::MAX =>
        {
            let conv_dim = 2 * state * n_k_heads + state * n_v_heads;
            Some(RecurrentShape {
                layers: backbone_blocks.saturating_sub(kv_layers.len() as u64),
                state_elems: n_v_heads * state * state,
                conv_elems: conv_k.saturating_sub(1) * conv_dim,
                conv_dim,
                elem_bytes: 4,
            })
        }
        _ => None,
    };

    // Whisper answers every geometry question from its own keys, and its two
    // caches are both derived here: the decoder's causal self-attention is
    // ordinary ctx-scaling KV (one d_model-wide K and V per decoder block,
    // capped by the trained 448), and the cross-attention over the encoder
    // window is the fixed per-slot term.
    let w = whisper_geometry(f);
    let (kv_layers, cross_kv) = match &w {
        Some(w) => (
            (0..w.dec_layers as usize)
                .map(|layer| KvLayer {
                    layer,
                    k_dim: w.d_model,
                    v_dim: w.d_model,
                    window: None,
                })
                .collect(),
            Some(CrossKv {
                layers: w.dec_layers,
                frames: w.audio_frames,
                k_dim: w.d_model,
                v_dim: w.d_model,
            }),
        ),
        None => (kv_layers, None),
    };

    Ok(ModelReport {
        path: path.to_path_buf(),
        file_size,
        shards: shards.len(),
        gguf_version: f.version,
        architecture: f.architecture().map(str::to_owned),
        tensor_count: shards.iter().map(|(g, _)| g.tensors.len()).sum(),
        quant_mix,
        context_length: w
            .as_ref()
            .map_or_else(|| arch_u64("context_length"), |w| Some(w.ctx)),
        block_count: w
            .as_ref()
            .map_or_else(|| arch_u64("block_count"), |w| Some(w.dec_layers)),
        embedding_length: w.as_ref().map_or(embedding_length, |w| Some(w.d_model)),
        head_count: w
            .as_ref()
            .map_or_else(|| arch_u64("attention.head_count"), |w| Some(w.heads)),
        // whisper is plain MHA everywhere - no GQA, so kv heads == heads
        head_count_kv: w
            .as_ref()
            .map_or_else(|| arch_u64("attention.head_count_kv"), |w| Some(w.heads)),
        sliding_window: arch_u64("attention.sliding_window"),
        expert_count: arch_u64("expert_count"),
        expert_used_count: arch_u64("expert_used_count"),
        tokenizer_model: f
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            .map(str::to_owned),
        // whisper embeds the whole HF tokenizer.json rather than a
        // tokenizer.ggml.tokens array, so the vocab comes off its own key
        token_count: w.as_ref().map_or(token_count, |w| Some(w.vocab)),
        has_chat_template: f.metadata.contains_key("tokenizer.chat_template"),
        kv_layers,
        nextn_blocks,
        nextn_bytes,
        recurrent,
        cross_kv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::Writer;

    /// Toy geometry: the rules under test are about which blocks cache and how
    /// wide, not about realistic sizes, and the parser bounds-checks every
    /// tensor against the data section - so keep the fixture small enough to
    /// actually back its own tensors.
    const N_EMBD: u64 = 64;
    const KV_DIM: u64 = 16;
    /// F32 tensor of [N_EMBD, KV_DIM], padded to the 32-byte alignment.
    const TENSOR_BYTES: u64 = N_EMBD * KV_DIM * 4;

    /// Build a small but self-consistent GGUF on disk and probe it.
    /// `attn_blocks` lists the blocks that get K/V projections; every other
    /// block stands in for a recurrent one with no cache. `extra` writes any
    /// additional metadata and returns how many pairs it wrote (the GGUF
    /// header needs the count up front).
    fn probe_synthetic(
        arch: &str,
        blocks: u64,
        attn_blocks: &[usize],
        extra: impl Fn(&mut Writer) -> u64,
    ) -> ModelReport {
        let names: Vec<String> = attn_blocks
            .iter()
            .flat_map(|i| {
                [
                    format!("blk.{i}.attn_k.weight"),
                    format!("blk.{i}.attn_v.weight"),
                ]
            })
            .collect();

        // write the metadata twice: once to learn `extra`'s count, once for real
        let mut probe = Writer::new(0, 0);
        let extra_count = extra(&mut probe);

        let mut w = Writer::new(names.len() as u64, 3 + extra_count);
        w.kv_str("general.architecture", arch);
        w.kv_u32(&format!("{arch}.block_count"), blocks as u32);
        w.kv_u32(&format!("{arch}.embedding_length"), N_EMBD as u32);
        extra(&mut w);
        for (n, name) in names.iter().enumerate() {
            w.tensor_f32(name, &[N_EMBD, KV_DIM], n as u64 * TENSOR_BYTES);
        }
        let bytes = w.finish_with_data(names.len() * TENSOR_BYTES as usize);

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("synthetic.gguf");
        std::fs::write(&path, &bytes).expect("write");
        probe_path(&path).expect("probe")
    }

    /// The hybrid rule: a block caches KV only if it has a K projection. This
    /// is what makes qwen3.5/3.6 come out at a quarter of their block count
    /// instead of charging DeltaNet blocks for a cache they never grow.
    #[test]
    fn only_blocks_with_a_k_projection_hold_kv() {
        let r = probe_synthetic("qwen35", 8, &[3, 7], |_| 0);
        assert_eq!(
            r.kv_layers.iter().map(|l| l.layer).collect::<Vec<_>>(),
            vec![3, 7]
        );
        assert!(
            r.kv_layers
                .iter()
                .all(|l| l.k_dim == KV_DIM && l.v_dim == KV_DIM)
        );
    }

    /// Trailing MTP/nextn blocks are counted in block_count but are not the
    /// backbone, and their cache is the drafter's, not the envelope's.
    #[test]
    fn nextn_blocks_are_excluded() {
        let r = probe_synthetic("qwen35", 4, &[1, 3], |w| {
            w.kv_u32("qwen35.nextn_predict_layers", 1);
            1
        });
        assert_eq!(r.nextn_blocks, 1);
        // block 3 is the nextn block here, so only block 1 counts
        assert_eq!(
            r.kv_layers.iter().map(|l| l.layer).collect::<Vec<_>>(),
            vec![1]
        );
    }

    /// A declared per-layer pattern wins, and window-capped blocks are marked.
    #[test]
    fn declared_sliding_window_pattern_is_applied() {
        let r = probe_synthetic("gemma4", 4, &[0, 1, 2, 3], |w| {
            w.kv_u32("gemma4.attention.sliding_window", 1024);
            w.kv_bool_array(
                "gemma4.attention.sliding_window_pattern",
                &[true, true, true, false],
            );
            2
        });
        let wins: Vec<_> = r.kv_layers.iter().map(|l| l.window).collect();
        assert_eq!(wins, vec![Some(1024), Some(1024), Some(1024), None]);
    }

    /// muse-glimmer ships the pattern as a scalar PERIOD rather than an array.
    /// Period 4 means the last block of each group is the full-attention one
    /// (`il % 4 < 3` is sliding) - the same phase the engine's
    /// `gemma4/load.rs::swa_pattern` uses, pinned against muse's config.json.
    /// Before this was read, every block priced as full ctx-scaling KV, which
    /// over-charged muse's cache by ~3.8x at its trained 131072 window.
    #[test]
    fn scalar_sliding_window_period_is_applied() {
        let r = probe_synthetic("muse-glimmer", 8, &[0, 1, 2, 3, 4, 5, 6, 7], |w| {
            w.kv_u32("muse-glimmer.attention.sliding_window", 2048);
            w.kv_u32("muse-glimmer.attention.sliding_window_pattern", 4);
            2
        });
        let wins: Vec<_> = r.kv_layers.iter().map(|l| l.window).collect();
        assert_eq!(
            wins,
            vec![
                Some(2048),
                Some(2048),
                Some(2048),
                None,
                Some(2048),
                Some(2048),
                Some(2048),
                None,
            ],
            "period 4 puts full attention on the LAST block of each group"
        );
    }

    /// gpt-oss declares a window but no pattern; the convention table places
    /// it on even blocks, matching the engine's `is_swa: i % 2 == 0`.
    #[test]
    fn gpt_oss_window_falls_back_to_convention() {
        let r = probe_synthetic("gpt-oss", 4, &[0, 1, 2, 3], |w| {
            w.kv_u32("gpt-oss.attention.sliding_window", 128);
            1
        });
        let wins: Vec<_> = r.kv_layers.iter().map(|l| l.window).collect();
        assert_eq!(wins, vec![Some(128), None, Some(128), None]);
    }

    /// laguna declares a window but no pattern; the convention table places
    /// it on every block EXCEPT the i%4==0 full-attention ones, matching the
    /// engine's `is_swa: i % 4 != 0`.
    #[test]
    fn laguna_window_falls_back_to_convention() {
        let r = probe_synthetic("laguna", 8, &[0, 1, 2, 3, 4, 5, 6, 7], |w| {
            w.kv_u32("laguna.attention.sliding_window", 512);
            1
        });
        let wins: Vec<_> = r.kv_layers.iter().map(|l| l.window).collect();
        let w = Some(512);
        assert_eq!(wins, vec![None, w, w, w, None, w, w, w]);
    }

    /// An unknown architecture that declares a window but no pattern must not
    /// guess where it applies - over-estimating is the safe direction.
    #[test]
    fn unknown_arch_window_without_pattern_is_not_applied() {
        let r = probe_synthetic("mystery", 2, &[0, 1], |w| {
            w.kv_u32("mystery.attention.sliding_window", 256);
            1
        });
        assert!(r.kv_layers.iter().all(|l| l.window.is_none()));
    }

    /// Projection width is keyed off the embedding length, not dim order, so a
    /// writer using the opposite convention still reports the right KV width.
    #[test]
    fn projection_width_is_the_non_embedding_dim() {
        assert_eq!(proj_out_dim(&[4096, 1024], Some(4096)), Some(1024));
        assert_eq!(proj_out_dim(&[1024, 4096], Some(4096)), Some(1024));
        // MHA, where both dims are the embedding width - either answer is same
        assert_eq!(proj_out_dim(&[4096, 4096], Some(4096)), Some(4096));
    }

    /// Encoder exports cache nothing, and that is an answer rather than a gap.
    #[test]
    fn a_model_with_no_k_projections_reports_no_kv() {
        let r = probe_synthetic("clip", 4, &[], |_| 0);
        assert!(r.kv_layers.is_empty());
        assert!(r.recurrent.is_none());
    }
}
