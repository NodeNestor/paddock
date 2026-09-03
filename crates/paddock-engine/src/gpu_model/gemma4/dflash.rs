//! Muse Glimmer's DFlash block-diffusion drafter.
//!
//! Meta ships a 2.6 B, 5-layer drafter next to the 30 B target. One drafter
//! forward drafts a whole block: rows = [last committed token, 15 x mask id
//! 201818] embedded through the TARGET's `token_embd`, five drafter layers
//! over a per-slot KV ring, the drafter's own final norm, then the TARGET's
//! `output.weight` and a device argmax. Rows 1.. are the drafts.
//!
//! What the drafter conditions on is the target's own middle: per layer it
//! reads a KV ring built from `z = enc_norm(fc(concat h_i))`, where `h_i` is
//! the target's residual stream ENTERING layers `dflash.target_layers` =
//! [2, 14, 26, 38, 50]. That index is the layer's INPUT (llama.cpp's
//! `t_layer_inp[il] = inpL`, muse-glimmer.cpp:85), so it equals the residual
//! LEAVING layers {1, 13, 25, 37, 49} - the same taps the model card lists.
//! Watch out: a +1 shift between the two spellings is silent corruption, but
//! that hazard belongs to laguna's safetensors drafter, whose
//! `aux_hidden_norms` this file's checkpoint does not have at all; here the
//! GGUF spelling and the card agree and there is nothing to shift.
//!
//! Reference for every line of the graph: llama.cpp `src/models/dflash.cpp`
//! (the non-DSpark arm) + `common/speculative.cpp`'s
//! `common_speculative_impl_draft_dflash`, read at the time. Study only -
//! this is our own implementation on our own kernels.
//!
//! Differences from `laguna/dflash.rs`, which is otherwise the template:
//!
//! - **Weights arrive as GGUF k-quant** (Q4_K bodies, Q6_K on `ffn_down` and
//!   `attn_v`), not bf16 safetensors, so there is no load-time quantize at
//!   all: every plane rides the existing W4A8 ladder exactly as the target's
//!   own k-quant models do. laguna had to host-quantize to Q8_0 first.
//! - **Split q/k/v planes** (the checkpoint ships them separately), qk
//!   RMSNorm, plain rope theta=500k, and **no per-head output gate** - the
//!   softplus gate is a laguna thing; this block is a plain Qwen3-style one.
//! - **The block is NON-CAUSAL**, which laguna's causal variant is not:
//!   `common/speculative.cpp` sets `llama_set_causal_attn(ctx_dft, false)`,
//!   so every noise row attends to every other row of its block. We get that
//!   from the attention bound the multimodal splice already uses - all 16
//!   rows carry bound `p + block - 1` while rope keeps true positions. Known
//!   deviation: our windowed kernels derive the SWA start from that same
//!   bound, so a row at p+i sees window start p+block-1-2047 instead of
//!   p+i-2047 - up to 15 of 2048 oldest keys, on drafts that are all
//!   target-verified anyway. Exact semantics need a second per-row position
//!   array in the prefill attention ABI; that is the SOTA target and is
//!   recorded rather than silently skipped.
//! - **No aux norms.** laguna norms each tap before the fusion; this
//!   checkpoint has no `aux_hidden_norms` and llama.cpp's loader has no
//!   mapping for them either - `fc` eats the raw concat.
//!
//! Fusion shape: `fc` is one [5*embd, embd] plane, and we split it at load
//! into five [embd, embd] bands (k-quant superblocks are position
//! independent, so a row-strip is a well-formed weight - `repack_kquant_bands`).
//! Each band is then consumed at its TAP, inside the layer walk, accumulating
//! into `zacc`. That is the whole reason the taps don't need a [rows,
//! 5*embd] staging plane: at 2 k rows that concat would be 545 MB of the
//! serving budget, and it would exist only to be read once.

use std::collections::HashMap;
use std::path::Path;

use cudarc::driver::CudaSlice;

use super::GpuGemma4;
use super::load::{LoadError, key_f32, key_u64, key_u64_array, plain_rope};
use crate::gpu::{GpuError, KvDtype, QuantW};
use crate::gpu_model::qwen35::{kq_mm_pre, mmq_pre_any, prefill_mm_pre_any};
use paddock_models::mapped::MappedGguf;

fn drv(e: cudarc::driver::DriverError) -> GpuError {
    crate::gpu::from_driver(e)
}

/// Draft blocks per round. One DFlash forward covers every warm slot, but
/// the rows are not free: the drafter's own head runs the TARGET's 202 k
/// vocab over all of them, so the logits plane alone is `blocks * 16 * vocab`
/// floats. 8 blocks (128 rows, ~103 MB) is the same cap laguna settled on
/// for the same reason; slots past it are verified pending-only that round.
fn max_blocks() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        paddock_models::dev_var!("PADDOCK_DFLASH_BLOCKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| (1..=32).contains(&n))
            .unwrap_or(8)
    })
}

/// PADDOCK_DFLASH_NOCONV=1 runs a DFlash2 checkpoint through the v1 forward,
/// conv skipped. Not a shipping mode - v2's weights were trained expecting the
/// conv, so this drafts worse. It exists to separate "the conv is wrong" from
/// "the conv is right and something else is missing", which acceptance alone
/// cannot tell apart.
fn conv_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NOCONV").is_some())
}

/// PADDOCK_DFLASH_CONV_TSIDE=1 reads `*_conv_base` as `[embd, side, tap]`
/// instead of `[embd, tap, side]`. The shipped geometry has taps == sides == 2,
/// so the tensor's SHAPE cannot distinguish the two and a transposed read is
/// silent - it costs acceptance and nothing else. This switch makes the
/// ambiguity measurable instead of assumed.
fn conv_tside() -> bool {
    static T: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *T.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_CONV_TSIDE").is_some())
}

/// PADDOCK_DFLASH_NOFUSE=1 skips the per-tick tap/fusion entirely. Not a
/// serving mode - the drafter's KV ring stops being fed, so every slot goes
/// cold and nothing drafts. It exists to price what having a drafter ATTACHED
/// costs a tick that does not draft: at high batch the Ladder already picks
/// k=0, yet spec-on still trails nospec by 4-10%, and fusion BANDWIDTH does
/// not explain it (~163 MB against a ~27.6 GB tick). This isolates it.
pub(crate) fn fuse_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NOFUSE").is_some())
}

/// PADDOCK_DFLASH_NOSELECT=1 drafts a v2 checkpoint by per-row argmax, the way
/// v1 does, with the candidate selector loaded but unused. The conv still runs.
/// Separates the selector's contribution from the conv's.
fn sel_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| paddock_models::dev_var_os!("PADDOCK_DFLASH_NOSELECT").is_some())
}

/// One drafter block - a plain Qwen3-style layer. Every projection is a
/// `QuantW`, so Q4_K/Q6_K/Q8_0 all serve off the same call sites.
/// DFlash2's grouped dynamic convolution, one per sublayer. It runs twice
/// around its sublayer - `prepare` before with base side 0, `finish` after
/// with side 1 - which is exactly what the trailing 2 of the base tensor
/// indexes. Both sides' dynamic deltas come out of one projection GEMM.
pub(crate) struct DflashConv {
    /// `[embd, taps, 2]` F32 in GGUF order, i.e. side-major last:
    /// element (side s, tap t, channel c) sits at `c + embd*(t + taps*s)`.
    pub base: CudaSlice<f32>,
    /// `[embd, 2*taps*num_groups]` - per-token, per-group deltas for both
    /// sides; the row splits into `[2][taps][num_groups]`.
    pub proj: QuantW,
}

/// DFlash2's candidate selector: a rank-r bilinear transition model that
/// scores `pred(prev) . (hidden * succ(next))` so a block of drafts can be
/// chosen as a coherent PATH instead of independently per position. This is
/// the piece that answers v1's conditional-independence problem.
pub(crate) struct DflashSelector {
    /// `[embd, rank]`
    pub hidden: QuantW,
    /// `[rank, vocab]`
    pub pred: QuantW,
    /// `[rank, vocab]`
    pub succ: QuantW,
}

pub(crate) struct DflashLayer {
    pub attn_norm: CudaSlice<f32>,
    pub wq: QuantW,
    pub wk: QuantW,
    pub wv: QuantW,
    pub wo: QuantW,
    pub q_norm: CudaSlice<f32>,
    pub k_norm: CudaSlice<f32>,
    pub ffn_norm: CudaSlice<f32>,
    pub w_gate: QuantW,
    pub w_up: QuantW,
    pub w_down: QuantW,
    /// DFlash2 only; `None` on a v1 checkpoint.
    pub attn_conv: Option<DflashConv>,
    pub ffn_conv: Option<DflashConv>,
}

/// Serving state: the feature KV rings plus the two row bands (fusion at
/// tick width, the draft round at block width). Built at `enable_batch`,
/// because the ring count needs the slot count.
pub(crate) struct DflashState {
    /// per-layer feature K/V rings, f16, [slots * ring * 16, kv_dim]
    pub kv: Vec<(CudaSlice<u8>, CudaSlice<u8>)>,
    /// static ring table [slots, bps]: s*ring + j%ring, drafter-sized ring
    pub d_bt: CudaSlice<u32>,
    /// blocks per sequence in the ring table - the paged-KV stride every
    /// append/attend call passes. (The ring depth itself and the slot count
    /// are baked into `d_bt` at build time and never re-read, so they live
    /// in the build log rather than here.)
    pub bps: usize,
    /// Fusion row capacity. Must cover a whole serving tick, else the tap
    /// cannot record every row the tick advanced and the affected slots go
    /// cold (`stale` below) rather than silently keeping a hole.
    pub cap: usize,
    /// Per slot, the TOKENS whose features currently sit in the ring, in
    /// coverage order - `cov[slot][i]` is the token at position
    /// `feat[slot].0 + i`. Kept so a prefix restore can prove how much of
    /// the ring still describes the incoming sequence instead of assuming
    /// none of it does; bounded by the same `window + block` the ring is.
    pub cov: Vec<Vec<u32>>,
    /// fc band accumulator + its scratch twin, [cap, embd]
    zacc: CudaSlice<f32>,
    ztmp: CudaSlice<f32>,
    /// quantized tap rows (one quantize serves all five bands' GEMMs)
    tq: CudaSlice<i8>,
    ts: CudaSlice<f32>,
    tyq: CudaSlice<u8>,
    txs: CudaSlice<f32>,
    tss: CudaSlice<f32>,
    /// feature K/V staging for the ring append, [cap, kv_dim]
    fk: CudaSlice<f32>,
    fkn: CudaSlice<f32>,
    fv: CudaSlice<f32>,
    /// draft-round row planes, [rows, ...] with rows = blocks * block
    x: CudaSlice<f32>,
    xn: CudaSlice<f32>,
    q: CudaSlice<f32>,
    qn: CudaSlice<f32>,
    k: CudaSlice<f32>,
    kn: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    proj: CudaSlice<f32>,
    /// DFlash2 conv output plane, [rows, embd]. A separate plane and not an
    /// in-place walk because the conv reads row-1 while writing row. A 1-elem
    /// stub on a v1 checkpoint.
    cvx: CudaSlice<f32>,
    /// DFlash2 conv coefficients, [rows, 2 * taps * num_groups] - One
    /// projection per sublayer feeds both wraps, so the side-1 half has to
    /// outlive the sublayer between `prepare` and `finish`. Attention's half
    /// is dead before the FFN's is written, so one plane serves both.
    cvc: CudaSlice<f32>,
    /// DFlash2 selector planes; 1-elem stubs without a selector.
    /// `sel_params` is a STATIC mode-4 row table - `topk_rows` is the host-head
    /// sampler's prefilter and gates on per-row mode, so handing it an
    /// all-mode-4 table is what lets the drafter reuse that tuned kernel
    /// instead of growing a second exact top-K.
    sel_params: CudaSlice<u32>,
    /// [rows, k, 2] - (id, raw-logit bits) pairs, `topk_rows`' own layout.
    sel_topk: CudaSlice<u32>,
    /// [rows*k + blocks] - candidate ids, block anchors appended at the tail.
    sel_ids: CudaSlice<u32>,
    /// [(rows*k + blocks), rank] and [rows*k, rank] - gathered codebook rows.
    sel_pred: CudaSlice<f32>,
    sel_succ: CudaSlice<f32>,
    /// [rows, rank] - the selector's hidden projection.
    sel_hs: CudaSlice<f32>,
    ffn_gate: CudaSlice<f32>,
    ffn_up: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    xq: CudaSlice<i8>,
    xs: CudaSlice<f32>,
    ssums: CudaSlice<f32>,
    part: CudaSlice<f32>,
    yq: CudaSlice<u8>,
    xsums: CudaSlice<f32>,
    /// K-split fixup plane for the Q8_0 arm of `prefill_mm_pre_any` AND the
    /// f8t head. A stub when every drafter plane is k-quant and no f8t head
    /// exists; the k-quant rungs never touch it.
    skfix: CudaSlice<f32>,
    /// e4m3 activation staging for the f8t head route (rows x embd bytes +
    /// one f32 row scale). Unused when the head has no tile plane.
    e4q: CudaSlice<i8>,
    e4rs: CudaSlice<f32>,
    /// no attention sinks in this arch - a -inf plane keeps the shared
    /// kernel signature honest instead of a special case
    sinks: CudaSlice<f32>,
    d_toks: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    /// attention BOUND per row (= block end, the non-causal block) - the
    /// same split the multimodal image splice uses
    d_apos: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    d_out: CudaSlice<u32>,
    /// per-slot contiguous feature coverage [start, end): drafting at p is
    /// legal iff end == p and start <= max(0, p - window)
    pub feat: Vec<(u32, u32)>,
    /// set by a tap that could not record the whole tick (r > cap): the
    /// append then clears every slot it touched instead of leaving a hole
    pub stale: bool,
    /// captured draft rounds keyed by (block count, rows per block) - both
    /// are shapes the body bakes in
    pub graphs: HashMap<(usize, usize), super::SendGraph>,
}

/// The attached drafter.
pub(crate) struct DflashDrafter {
    pub layers: Vec<DflashLayer>,
    /// `fc` split into `n_taps` [embd, embd] column bands, in
    /// `target_layers` order - band i multiplies the tap from
    /// `target_layers[i]`
    pub fc_bands: Vec<QuantW>,
    /// `enc.output_norm` - the encoder's norm after fc
    pub enc_norm: CudaSlice<f32>,
    /// the drafter's own final norm; the TARGET's head runs on top of it
    pub final_norm: CudaSlice<f32>,
    /// target-layer INPUT indices the walk taps
    pub target_layers: Vec<usize>,
    pub block: usize,
    pub mask_token: u32,
    pub n_heads: usize,
    pub n_kv: usize,
    pub hd: usize,
    pub window: usize,
    pub eps: f32,
    pub rope: (f32, f32, f32, f32, f32, f32),
    /// DFlash2 conv geometry: (taps, group_size, num_groups). `None` = v1.
    pub conv: Option<(usize, usize, usize)>,
    /// DFlash2 candidate selector plus its (rank, top_k). `None` = v1.
    pub selector: Option<(DflashSelector, usize, usize)>,
    /// The DRAFTER's own logit epilogue. v1 declares neither key and wants
    /// none; v2 ships both, and they do not commute - scale first, then cap,
    /// via `logit_epilogue_dev`. Getting this backwards is invisible to
    /// greedy output and shows up only as lost acceptance.
    pub logit_scale: f32,
    pub softcap: f32,
    pub state: Option<DflashState>,
}

/// Result of the synthetic end-to-end smoke (`dflash_selftest`).
pub struct DflashSelftest {
    pub drafts: Vec<u32>,
    /// same round twice -> identical picks (catches ring-append races)
    pub repeat_identical: bool,
    pub ms_per_round: f64,
}

impl GpuGemma4 {
    /// Sideload the DFlash drafter GGUF. Validates its geometry against the
    /// target it will draft for, splits `fc` into per-tap bands, and leaves
    /// serving untouched until `enable_batch` builds the rings.
    pub fn attach_dflash(&mut self, path: &Path) -> Result<(), LoadError> {
        let file = if path.is_dir() {
            path.join("dflash-kquant.gguf")
        } else {
            path.to_path_buf()
        };
        let map = MappedGguf::open(&file).map_err(|e| {
            LoadError::Tensor(file.display().to_string(), format!("open dflash gguf: {e}"))
        })?;
        let arch = map
            .gguf()
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str().map(|s| s.to_owned()))
            .unwrap_or_default();
        if arch != "dflash" {
            return Err(LoadError::BadKey(format!(
                "general.architecture = {arch:?}, want \"dflash\""
            )));
        }
        let k = |s: &str| format!("dflash.{s}");
        let n_layer = key_u64(&map, &k("block_count"))? as usize;
        let embd = key_u64(&map, &k("embedding_length"))? as usize;
        let ff = key_u64(&map, &k("feed_forward_length"))? as usize;
        let n_heads = key_u64(&map, &k("attention.head_count"))? as usize;
        let n_kv = key_u64(&map, &k("attention.head_count_kv"))? as usize;
        let hd = key_u64(&map, &k("attention.key_length"))? as usize;
        let hd_v = key_u64(&map, &k("attention.value_length"))? as usize;
        let window = key_u64(&map, &k("attention.sliding_window"))? as usize;
        let eps = key_f32(&map, &k("attention.layer_norm_rms_epsilon"), Some(1e-5))?;
        let theta = key_f32(&map, &k("rope.freq_base"), Some(500_000.0))?;
        let block = key_u64(&map, &k("block_size"))? as usize;
        let taps: Vec<usize> = key_u64_array(&map, &k("target_layers"))?
            .into_iter()
            .map(|v| v as usize)
            .collect();
        let mask_token = key_u64(&map, "tokenizer.ggml.mask_token_id")? as u32;

        // DFlash2 keeps arch "dflash" and v1's whole tensor vocabulary, adding
        // a grouped dynamic conv per sublayer and a candidate selector. The
        // version is therefore not in the arch string - detect it from the
        // conv geometry key, and let the tensor audit below prove the file
        // really is what the key claims.
        let conv_cfg = match map.gguf().metadata.get(&k("conv_kernel_size")) {
            None => None,
            Some(_) => {
                let taps = key_u64(&map, &k("conv_kernel_size"))? as usize;
                let gsz = key_u64(&map, &k("conv_group_size"))? as usize;
                if !(1..=8).contains(&taps) {
                    return Err(LoadError::BadKey(format!(
                        "dflash conv_kernel_size {taps} outside 1..=8"
                    )));
                }
                if gsz == 0 || !embd.is_multiple_of(gsz) {
                    return Err(LoadError::BadKey(format!(
                        "dflash conv_group_size {gsz} does not divide embd {embd}"
                    )));
                }
                // REFUSE rather than fall back to the v1 forward. v2's weights
                // are trained expecting the conv, so running them without it
                // drafts worse than v1 while looking perfectly healthy - the
                // damage would show up only as quiet acceptance loss.
                if !self.exec.has_dflash_conv() {
                    return Err(LoadError::BadKey(
                        "dflash2 checkpoint needs the grouped-conv kernel (pack slot 459); \
                         this pack predates it - rebuild packs/cuda"
                            .into(),
                    ));
                }
                if !embd.is_multiple_of(4) || !gsz.is_multiple_of(4) {
                    return Err(LoadError::BadKey(format!(
                        "dflash2 conv geometry embd {embd} / group {gsz} must both be \
                         multiples of 4 for the packed channel walk"
                    )));
                }
                Some((taps, gsz, embd / gsz))
            }
        };
        let sel_cfg = match conv_cfg {
            None => None,
            Some(_) => Some((
                key_u64(&map, &k("selector_rank"))? as usize,
                key_u64(&map, &k("selector_top_k"))? as usize,
            )),
        };
        // v1 declares neither key and wants no epilogue; v2 ships both. They do
        // not commute, so this is carried as the pair and applied scale-then-cap.
        let logit_scale = key_f32(&map, &k("logit_scale"), Some(1.0))?;
        let softcap = key_f32(&map, &k("final_logit_softcapping"), Some(0.0))?;

        // Geometry has to agree with the target on exactly the things the two
        // models SHARE: the residual width the taps carry, and the embedding
        // + head planes the draft round borrows.
        if embd != self.hp.n_embd {
            return Err(LoadError::BadKey(format!(
                "dflash embedding_length {embd} != target n_embd {}",
                self.hp.n_embd
            )));
        }
        if hd != hd_v {
            return Err(LoadError::BadKey(format!(
                "dflash key_length {hd} != value_length {hd_v} - the ring is one dim"
            )));
        }
        if taps.is_empty() || taps.iter().any(|&t| t >= self.hp.n_layer) || !taps.is_sorted() {
            return Err(LoadError::BadKey(format!(
                "dflash target_layers {taps:?} don't index the {}-layer target",
                self.hp.n_layer
            )));
        }
        if !(2..=64).contains(&block) {
            return Err(LoadError::BadKey(format!(
                "dflash block_size {block} outside the 2..=64 a draft round sizes"
            )));
        }
        if window == 0 {
            return Err(LoadError::BadKey(
                "dflash sliding_window 0 - the ring needs a bound".into(),
            ));
        }
        // A layer that is not sliding-window would need a second, unbounded
        // ring; this checkpoint marks all five SWA and llama.cpp's loader
        // asserts the same. Refuse rather than quietly window a global layer.
        if let Ok(pat) = super::load::key_bool_array(&map, &k("attention.sliding_window_pattern"))
            && (pat.len() != n_layer || pat.iter().any(|&b| !b))
        {
            return Err(LoadError::BadKey(format!(
                "dflash sliding_window_pattern {pat:?}: this lane implements the all-SWA drafter"
            )));
        }

        // The selector's codebooks are indexed by TARGET token ids, so their
        // vocabulary has to be the target's, not merely some 202048.
        let vocab = self.hp.n_vocab;

        self.exec
            .vram_load_gate(file.metadata().map(|m| m.len()).unwrap_or(0), "muse-dflash")
            .map_err(|e| LoadError::WontFit(e.to_string()))?;

        let exec = self.exec.clone();
        let mut bytes = 0u64;
        let mut plane = |name: &str, want: [usize; 2]| -> Result<QuantW, LoadError> {
            let w = exec.load_quantw(&map, name)?;
            let dims = match &w {
                QuantW::Q8(q) => q.dims.clone(),
                QuantW::Kq(q) => q.dims.clone(),
            };
            // Q8_0 repack pads the in dim up to a tile multiple, so compare
            // the out dim exactly and the in dim as ">= what the file said".
            if dims.len() != 2 || dims[1] != want[1] || dims[0] < want[0] {
                return Err(LoadError::Tensor(
                    name.to_owned(),
                    format!("dims {dims:?}, want {want:?}"),
                ));
            }
            bytes += w.bytes();
            Ok(w)
        };
        // `[embd, taps, 2]` F32: one base coefficient per (side, tap, channel).
        // Uploaded flat; the kernel indexes c + embd*(t + taps*s).
        // Its own counter: `plane` already holds the only mutable borrow of
        // `bytes`, and these fold in after the layer loop.
        let mut conv_bytes = 0u64;
        let mut conv_base = |name: &str, taps: usize| -> Result<CudaSlice<f32>, LoadError> {
            let tt = exec.upload(&map, name)?;
            let want = vec![embd, taps, 2];
            if tt.dims != want {
                return Err(LoadError::Tensor(
                    name.to_owned(),
                    format!("{:?} != {want:?}", tt.dims),
                ));
            }
            conv_bytes += 4 * (embd * taps * 2) as u64;
            if !conv_tside() {
                return Ok(tt.buf);
            }
            // Re-read as [embd, side, tap]: swap the two trailing axes so the
            // kernel's (t + taps*s) walk lands on the transposed source.
            let host = exec.stream.clone_dtoh(&tt.buf).map_err(|e| {
                LoadError::Tensor(name.to_owned(), format!("conv base readback: {e}"))
            })?;
            let mut sw = vec![0.0f32; host.len()];
            for sd in 0..2 {
                for t in 0..taps {
                    let src = embd * (sd + 2 * t);
                    let dst = embd * (t + taps * sd);
                    sw[dst..dst + embd].copy_from_slice(&host[src..src + embd]);
                }
            }
            exec.to_device(&sw)
                .map_err(|e| LoadError::Tensor(name.to_owned(), format!("conv base swap: {e}")))
        };
        let norm = |name: &str, n: usize| -> Result<CudaSlice<f32>, LoadError> {
            let t = exec.upload(&map, name)?;
            if t.dims != vec![n] {
                return Err(LoadError::Tensor(
                    name.to_owned(),
                    format!("{:?} != [{n}]", t.dims),
                ));
            }
            Ok(t.buf)
        };

        let (q_dim, kv_dim) = (n_heads * hd, n_kv * hd);
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            let p = |s: &str| format!("blk.{i}.{s}");
            layers.push(DflashLayer {
                attn_norm: norm(&p("attn_norm.weight"), embd)?,
                wq: plane(&p("attn_q.weight"), [embd, q_dim])?,
                wk: plane(&p("attn_k.weight"), [embd, kv_dim])?,
                wv: plane(&p("attn_v.weight"), [embd, kv_dim])?,
                wo: plane(&p("attn_output.weight"), [q_dim, embd])?,
                q_norm: norm(&p("attn_q_norm.weight"), hd)?,
                k_norm: norm(&p("attn_k_norm.weight"), hd)?,
                ffn_norm: norm(&p("ffn_norm.weight"), embd)?,
                w_gate: plane(&p("ffn_gate.weight"), [embd, ff])?,
                w_up: plane(&p("ffn_up.weight"), [embd, ff])?,
                w_down: plane(&p("ffn_down.weight"), [ff, embd])?,
                attn_conv: match conv_cfg {
                    None => None,
                    Some((taps, _, ng)) => Some(DflashConv {
                        base: conv_base(&p("attn_conv_base"), taps)?,
                        proj: plane(&p("attn_conv_proj.weight"), [embd, 2 * taps * ng])?,
                    }),
                },
                ffn_conv: match conv_cfg {
                    None => None,
                    Some((taps, _, ng)) => Some(DflashConv {
                        base: conv_base(&p("ffn_conv_base"), taps)?,
                        proj: plane(&p("ffn_conv_proj.weight"), [embd, 2 * taps * ng])?,
                    }),
                },
            });
        }
        let selector = match sel_cfg {
            None => None,
            Some((rank, top_k)) => {
                if top_k == 0 || top_k > block {
                    return Err(LoadError::BadKey(format!(
                        "dflash selector_top_k {top_k} outside 1..=block({block})"
                    )));
                }
                // Same refusal shape as the conv: a v2 checkpoint whose
                // selector cannot run drafts by per-row argmax, which loads
                // clean and simply drafts worse.
                if !self.exec.has_dflash_select() || !self.exec.has_topk_rows() {
                    return Err(LoadError::BadKey(
                        "dflash2 selector needs pack slots 460/461 + topk_rows; \
                         this pack predates them - rebuild packs/cuda"
                            .into(),
                    ));
                }
                if rank % 256 != 0 {
                    return Err(LoadError::BadKey(format!(
                        "dflash2 selector_rank {rank} must be a multiple of 256 - the \
                         codebook row-gather dequants whole k-quant super-blocks"
                    )));
                }
                let (hidden, pred, succ) = (
                    plane("selector_hidden.weight", [embd, rank])?,
                    plane("selector_predecessor.weight", [rank, vocab])?,
                    plane("selector_successor.weight", [rank, vocab])?,
                );
                if !matches!((&pred, &succ), (QuantW::Kq(_), QuantW::Kq(_))) {
                    return Err(LoadError::Tensor(
                        "selector_predecessor/successor".into(),
                        "codebook row-gather implements the k-quant classes only".into(),
                    ));
                }
                Some((DflashSelector { hidden, pred, succ }, rank, top_k))
            }
        };
        bytes += conv_bytes; // after `plane`'s last use
        // fc [n_taps*embd, embd] -> n_taps [embd, embd] bands, concat order
        // = target_layers order (llama.cpp's features_buf writes tap k at
        // k*n_embd of each row, so band k is tap k)
        let fc_bands = self.fc_bands(&map, taps.len(), embd, &mut bytes)?;
        let enc_norm = norm("enc.output_norm.weight", embd)?;
        let final_norm = norm("output_norm.weight", embd)?;

        // Audit the header: a tensor we silently ignore is modeling drift we
        // didn't notice. v1 is 11 per layer + fc + the two norms; v2 adds two
        // conv modules per layer (base + projection each) and the selector's
        // three planes. This is what refused DFlash2 before it was supported,
        // and it must keep refusing anything that is neither shape exactly:
        // v2's arch string is still "dflash", so the count is the only proof.
        let expected = if conv_cfg.is_some() {
            n_layer * 15 + 6
        } else {
            n_layer * 11 + 3
        };
        let seen = map.tensor_infos().count();
        if seen != expected {
            return Err(LoadError::BadKey(format!(
                "dflash: {seen} tensors in file, {expected} consumed"
            )));
        }

        // Serving spec knobs - one DFlash forward drafts the whole block, so
        // draft DEPTH is free, but verify rows are not: each costs a target
        // row. block-1 = 15 drafts is what the reference allows; k is capped
        // at the family's SPEC_K1_MAX so the verify shapes stay in the band
        // the target's kernels are tuned for. Explicit env always wins.
        // SAFETY: model load runs before the serving threads spawn.
        let k_default = (block - 1).min(super::spec::SPEC_K1_MAX);
        for (key, val) in [
            ("PADDOCK_SPEC_MAX_K", k_default.to_string()),
            ("PADDOCK_SPEC_K_MISS_FLOOR", k_default.to_string()),
            // Spec ENGAGEMENT cap, DFlash-only. gemma4's uncapped default is
            // sized for its MTP drafter, which commits ~2.78 rows per slot
            // per round; DFlash commits ~1.2 at c8 and pays a per-tick fusion
            // the MTP head does not, so the same default is wrong here.
            // Capping engagement at 2 live slots (plus the width-gated
            // fusion) is what keeps DFlash ahead under concurrency. c1 is
            // untouched: one live slot is already inside the cap, so
            // single-stream keeps the full speculation win.
            ("PADDOCK_G4_SPEC_LIVE_MAX", "2".to_string()),
        ] {
            if std::env::var_os(key).is_none() {
                crate::envset::set_env(key, &val);
            }
        }

        tracing::info!(
            "muse dflash{} drafter attached: {n_layer} layers, block {block}, mask {mask_token}, \
             taps {taps:?} (= residual leaving {:?}), swa {window}, {:.2} GB{}",
            if conv_cfg.is_some() { "2" } else { "" },
            taps.iter().map(|t| t.saturating_sub(1)).collect::<Vec<_>>(),
            bytes as f64 / 1e9,
            match (conv_cfg, sel_cfg) {
                (Some((tp, gs, ng)), Some((rank, tk))) => format!(
                    " | conv {tp}-tap x {ng} groups of {gs}, selector rank {rank} top-{tk}, epilogue scale {logit_scale} cap {softcap}"
                ),
                _ => String::new(),
            },
        );
        self.weights_bytes = Some(self.weights_bytes.unwrap_or(0) + bytes);
        self.dflash = Some(DflashDrafter {
            layers,
            fc_bands,
            enc_norm,
            final_norm,
            target_layers: taps,
            block,
            mask_token,
            n_heads,
            n_kv,
            hd,
            window,
            eps,
            rope: plain_rope(theta, hd),
            conv: conv_cfg,
            selector,
            logit_scale,
            softcap,
            state: None,
        });
        Ok(())
    }

    /// Split the fusion `fc` into one weight per tap. k-quant takes the
    /// superblock-strip path; a Q8_0 `fc` (no shipped drafter uses one yet,
    /// but the class is legal in a GGUF) would need its own 32-block gather,
    /// so refuse loudly rather than mis-slice it.
    fn fc_bands(
        &self,
        map: &MappedGguf,
        n_taps: usize,
        embd: usize,
        bytes: &mut u64,
    ) -> Result<Vec<QuantW>, LoadError> {
        let (info, _) = map.tensor_bytes("fc.weight").map_err(GpuError::from)?;
        let (fin, fout) = (info.dims[0] as usize, info.dims[1] as usize);
        if fin != n_taps * embd || fout != embd {
            return Err(LoadError::Tensor(
                "fc.weight".into(),
                format!("dims [{fin}, {fout}], want [{}, {embd}]", n_taps * embd),
            ));
        }
        if crate::gpu::kq_params(info.ggml_type).is_none() {
            return Err(LoadError::Tensor(
                "fc.weight".into(),
                format!(
                    "{:?}: the band split implements the k-quant classes",
                    info.ggml_type
                ),
            ));
        }
        let bands = self.exec.repack_kquant_bands(map, "fc.weight", n_taps)?;
        Ok(bands
            .into_iter()
            .map(|b| {
                let w = QuantW::Kq(b);
                *bytes += w.bytes();
                w
            })
            .collect())
    }

    /// Is a DFlash drafter attached AND servable by this pack? Attach-time
    /// fact only (no serving state needed), which is what `spec_live_cap`
    /// wants: it is read before `enable_batch` builds the rings, and a pack
    /// without `argmax_rows` must not have the scheduler warming slots for
    /// rounds `dflash_draft_batch` will decline anyway.
    pub(crate) fn dflash_attached(&self) -> bool {
        self.dflash.is_some() && self.exec.has_argmax_rows()
    }

    /// Is the serving state live? Every tap/append/draft site keys on this.
    pub(crate) fn dflash_armed(&self) -> bool {
        self.dflash.as_ref().is_some_and(|d| d.state.is_some())
    }

    /// Build the rings + row bands once the batch lane knows its slot count.
    pub(crate) fn dflash_ensure_state(&mut self) -> Result<(), GpuError> {
        if self.dflash.as_ref().is_none_or(|d| d.state.is_some()) {
            return Ok(());
        }
        let slots = self.n_slots.max(1);
        let embd = self.hp.n_embd;
        let vocab = self.hp.n_vocab;
        let bps = self.max_ctx.div_ceil(16);
        let e = self.exec.clone();
        let df = self.dflash.as_mut().expect("attached");
        let (kv_dim, q_dim) = (df.n_kv * df.hd, df.n_heads * df.hd);
        let ff = match &df.layers[0].w_gate {
            QuantW::Q8(w) => w.dims[1],
            QuantW::Kq(w) => w.dims[1],
        };
        // block rows + window + a block of slack, same formula as the
        // target's own SWA ring
        let any_q8 = df.layers.iter().any(|l| {
            [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down]
                .iter()
                .any(|w| matches!(w, QuantW::Q8(_)))
        });
        let f8t_head = self.head_f8t.is_some() || self.head_f8row.is_some();
        let ring = ((df.block + df.window).div_ceil(16) + 1).min(bps);
        let mut host = vec![0u32; slots * bps];
        for s in 0..slots {
            for j in 0..bps {
                host[s * bps + j] = (s * ring + (j % ring)) as u32;
            }
        }
        let d_bt = e.to_device_u32(&host)?;
        let mut kv = Vec::with_capacity(df.layers.len());
        for _ in 0..df.layers.len() {
            let b = slots * ring * 16 * kv_dim * 2; // f16
            kv.push((e.alloc_u8(b)?, e.alloc_u8(b)?));
        }
        // Fusion width: one serving tick's rows. The window+block floor is
        // what makes a truncating tap harmless - anything older than that
        // falls outside every future query's window anyway.
        let cap = super::batch::tick_row_cap(slots).max(df.window + df.block + 64);
        let rows = max_blocks().min(slots) * df.block;
        let wide = embd.max(ff).max(q_dim);
        // 0 on a v1 checkpoint, which is what stubs both conv planes.
        let conv_dim = df.conv.map_or(0, |(taps, _, ng)| 2 * taps * ng);
        let n_blk = max_blocks().min(slots);
        let (sel_r, sel_k) = df.selector.as_ref().map_or((0, 0), |(_, r, k)| (*r, *k));
        tracing::info!(
            "muse dflash state: {slots} slots x {ring} ring blocks x {} layers \
             ({:.1} MB/slot), fusion cap {cap} rows, {} draft rows/round",
            df.layers.len(),
            (df.layers.len() * ring * 16 * kv_dim * 2 * 2) as f64 / 1e6,
            rows,
        );
        df.state = Some(DflashState {
            kv,
            d_bt,
            bps,
            cap,
            zacc: e.alloc(cap * embd)?,
            ztmp: e.alloc(cap * embd)?,
            tq: e.alloc_i8(cap * embd)?,
            ts: e.alloc(cap * embd / 32)?,
            tyq: e.alloc_u8(embd.div_ceil(128) * cap.next_multiple_of(128) * 144)?,
            txs: e.alloc(embd.div_ceil(128) * cap.next_multiple_of(128) * 4)?,
            tss: e.alloc(cap * embd / 16)?,
            fk: e.alloc(cap * kv_dim)?,
            fkn: e.alloc(cap * kv_dim)?,
            fv: e.alloc(cap * kv_dim)?,
            x: e.alloc(rows * embd)?,
            xn: e.alloc(rows * embd)?,
            q: e.alloc(rows * q_dim)?,
            qn: e.alloc(rows * q_dim)?,
            k: e.alloc(rows * kv_dim)?,
            kn: e.alloc(rows * kv_dim)?,
            v: e.alloc(rows * kv_dim)?,
            attn: e.alloc(rows * q_dim)?,
            proj: e.alloc(rows * embd)?,
            cvx: e.alloc(if conv_dim > 0 { rows * embd } else { 1 })?,
            cvc: e.alloc(if conv_dim > 0 { rows * conv_dim } else { 1 })?,
            sel_params: e.alloc_u32(if sel_k > 0 { rows * 4 } else { 1 })?,
            sel_topk: e.alloc_u32(if sel_k > 0 { rows * sel_k * 2 } else { 1 })?,
            sel_ids: e.alloc_u32(if sel_k > 0 { rows * sel_k + n_blk } else { 1 })?,
            sel_pred: e.alloc(if sel_k > 0 {
                (rows * sel_k + n_blk) * sel_r
            } else {
                1
            })?,
            sel_succ: e.alloc(if sel_k > 0 { rows * sel_k * sel_r } else { 1 })?,
            sel_hs: e.alloc(if sel_k > 0 { rows * sel_r } else { 1 })?,
            ffn_gate: e.alloc(rows * ff)?,
            ffn_up: e.alloc(rows * ff)?,
            logits: e.alloc(rows * vocab)?,
            xq: e.alloc_i8(rows * wide)?,
            xs: e.alloc(rows * wide / 32)?,
            ssums: e.alloc(rows * wide / 16)?,
            part: e.alloc(8 * 64 * wide)?,
            yq: e.alloc_u8(wide.div_ceil(128) * rows.next_multiple_of(128) * 144)?,
            xsums: e.alloc(wide.div_ceil(128) * rows.next_multiple_of(128) * 4)?,
            // the f8t head wants a real skfix too - its K-split election can
            // land nz>1 at row counts the target's own head never sees
            skfix: e.alloc(if any_q8 || f8t_head {
                256 * 128 * 128 + 256
            } else {
                1
            })?,
            e4q: e.alloc_i8(if f8t_head { rows * embd } else { 1 })?,
            e4rs: e.alloc(if f8t_head { rows } else { 1 })?,
            sinks: e.alloc_no_sinks(df.n_heads)?,
            d_toks: e.alloc_u32(rows)?,
            d_pos: e.alloc_u32(rows)?,
            d_apos: e.alloc_u32(rows)?,
            d_slots: e.alloc_u32(rows)?,
            d_out: e.alloc_u32(rows)?,
            feat: vec![(0, 0); slots],
            cov: vec![Vec::new(); slots],
            stale: false,
            graphs: HashMap::new(),
        });
        // `topk_rows` gates per row on the host sampler's mode byte, so the
        // drafter publishes a table that says "mode 4, every row" ONCE. It is
        // constant for the life of the state, which is also what keeps it
        // inside the captured draft graph.
        if sel_k > 0 {
            let st = df.state.as_mut().expect("just built");
            // PdSampleRow = { f32 inv_t, f32 u, u32 mode, u32 pad }
            let host: Vec<u32> = (0..rows).flat_map(|_| [0u32, 0, 4, 0]).collect();
            let mut v = st
                .sel_params
                .try_slice_mut(0..host.len())
                .ok_or_else(|| GpuError::Driver("dflash selector params slice".into()))?;
            e.stream.memcpy_htod(&host, &mut v).map_err(drv)?;
        }
        Ok(())
    }

    /// Can the drafter legally draft for `slot` at position `p`? Features
    /// contiguous up to exactly p, covering the whole window.
    pub(crate) fn dflash_warm(&self, slot: usize, p: usize) -> bool {
        let Some(df) = self.dflash.as_ref() else {
            return false;
        };
        let Some(st) = df.state.as_ref() else {
            return false;
        };
        let Some(&(s, e)) = st.feat.get(slot) else {
            return false;
        };
        e as usize == p && (s as usize) <= p.saturating_sub(df.window)
    }

    /// Drop a slot's coverage (fresh sequence / release: the ring no longer
    /// describes what will serve).
    pub(crate) fn dflash_clear_slot(&mut self, slot: usize) {
        if let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut()) {
            if let Some(f) = st.feat.get_mut(slot) {
                *f = (0, 0);
            }
            if let Some(c) = st.cov.get_mut(slot) {
                c.clear();
            }
        }
    }

    /// Cut a slot's coverage back to the longest prefix of `tokens` the ring
    /// provably describes - what a prefix restore leaves valid.
    ///
    /// A feature at position `i` is a function of `tokens[..=i]`, so if this
    /// slot already walked a sequence that agrees with the incoming one over
    /// `[s, m)`, the ring rows for `[s, m)` are exactly right and the tail
    /// re-prefill extends them. Anything at or past the first disagreement
    /// describes a different sequence and must go. Coverage that no longer
    /// starts at `s` (a hole would open) collapses to cold.
    pub(crate) fn dflash_trim_slot(&mut self, slot: usize, tokens: &[u32]) {
        let Some(st) = self.dflash.as_mut().and_then(|d| d.state.as_mut()) else {
            return;
        };
        let (Some(&(s, e)), Some(cov)) = (st.feat.get(slot), st.cov.get(slot)) else {
            return;
        };
        // the record is [s, e); compare it against the same absolute span of
        // the incoming tokens and stop at the first difference
        let span = (e - s) as usize;
        let mut agree = 0usize;
        while agree < span
            && agree < cov.len()
            && (s as usize) + agree < tokens.len()
            && cov[agree] == tokens[(s as usize) + agree]
        {
            agree += 1;
        }
        if agree == 0 {
            self.dflash_clear_slot(slot);
            return;
        }
        st.feat[slot] = (s, s + agree as u32);
        st.cov[slot].truncate(agree);
    }

    /// The selftest's tap driver - serving taps go through the free
    /// `tap_band` below, because the layer walk is already holding `self`
    /// apart (scratch mutably, layers immutably) and cannot call a
    /// `&mut self` method.
    pub(crate) fn dflash_tap(&mut self, band: usize, r: usize) -> Result<(), GpuError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let sc = &self.scratch;
        let df = self.dflash.as_mut().expect("armed");
        tap_band(&exec, df, &sc.pf_x, band, embd, r)
    }

    /// Fuse + ring-append feature K/V for the rows the walk just tapped.
    ///
    /// Preconditions (the call site's contract): `dflash_tap` ran for every
    /// band over exactly these `r` rows, and `positions`/`slots` are their
    /// host mirrors in row order. `spans = None` appends every row grouped
    /// into same-slot runs (prefill chunks, decode ticks); `Some` appends
    /// only those row ranges (the verify commit - rejected rows' features
    /// are computed but never committed, since their input tokens are not
    /// what the sequence kept).
    pub(crate) fn dflash_append_features(
        &mut self,
        tokens: &[u32],
        positions: &[u32],
        slots: &[u32],
        spans: Option<&[(usize, usize)]>,
    ) -> Result<(), GpuError> {
        let r = positions.len();
        assert_eq!(r, slots.len());
        assert_eq!(r, tokens.len(), "coverage mirror needs one token per row");
        if r == 0 || !self.dflash_armed() {
            return Ok(());
        }
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let df = self.dflash.as_mut().expect("armed");
        let (n_kv, hd, eps, rope, window, block) =
            (df.n_kv, df.hd, df.eps, df.rope, df.window, df.block);
        let kv_dim = n_kv * hd;
        let DflashDrafter {
            layers,
            enc_norm,
            state,
            ..
        } = df;
        let st = state.as_mut().expect("armed");
        if st.stale || r > st.cap {
            // the tap could not record this tick - every slot it touched
            // loses coverage rather than keeping a hole
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[dflash-append] WIPE r={r} stale={} cap={} slots={:?}",
                    st.stale,
                    st.cap,
                    slots
                );
            }
            st.stale = false;
            for &s in slots {
                if let Some(f) = st.feat.get_mut(s as usize) {
                    *f = (0, 0);
                }
                if let Some(c) = st.cov.get_mut(s as usize) {
                    c.clear();
                }
            }
            return Ok(());
        }

        // z = enc_norm(fc(concat taps)); the fc sum already landed in zacc
        // during the walk, so the encoder is one norm away.
        exec.rmsnorm_batch(&st.zacc, enc_norm, &mut st.ztmp, embd, eps, r)?;
        exec.quantize_q8(&st.ztmp, &mut st.tq, &mut st.ts, r * embd)?;
        if r > 64 {
            exec.quantize_q8_mmq(&st.ztmp, &mut st.tyq, embd, r)?;
        }

        // Append units: explicit spans, or same-slot runs off the row
        // stream. Keep only each unit's trailing window+block positions
        // (older rows sit outside every future query's window and would
        // alias their own ring mates), then cut at the ring block size so
        // one launch never writes a physical slot twice.
        let keep = window + block;
        let runs: Vec<(usize, usize)> = match spans {
            Some(s) => s.to_vec(),
            None => {
                let mut runs = Vec::new();
                let mut i = 0;
                while i < r {
                    let mut j = i + 1;
                    while j < r && slots[j] == slots[i] && positions[j] == positions[j - 1] + 1 {
                        j += 1;
                    }
                    runs.push((i, j - i));
                    i = j;
                }
                runs
            }
        };
        let mut cuts: Vec<(usize, usize)> = Vec::new();
        for &(row, n) in &runs {
            let (off, len) = if n > keep {
                (row + n - keep, keep)
            } else {
                (row, n)
            };
            let mut o = off;
            while o < off + len {
                let l = (off + len - o).min(keep);
                cuts.push((o, l));
                o += l;
            }
        }

        let (d_pos, d_slots) = (&self.scratch.pf_pos, &self.scratch.pf_slots);
        for (li, layer) in layers.iter().enumerate() {
            prefill_mm_pre_any(
                &exec,
                &layer.wk,
                &st.tq,
                &st.ts,
                &st.tyq,
                &mut st.txs,
                &mut st.tss,
                &mut st.skfix,
                &mut st.fk,
                r,
            )
            .map_err(|e| GpuError::Driver(e.to_string()))?;
            exec.rmsnorm_batch(&st.fk, &layer.k_norm, &mut st.fkn, hd, eps, r * n_kv)?;
            exec.rope_yarn_batch(&mut st.fkn, d_pos, n_kv, hd, rope, r)?;
            prefill_mm_pre_any(
                &exec,
                &layer.wv,
                &st.tq,
                &st.ts,
                &st.tyq,
                &mut st.txs,
                &mut st.tss,
                &mut st.skfix,
                &mut st.fv,
                r,
            )
            .map_err(|e| GpuError::Driver(e.to_string()))?;
            let (pool_k, pool_v) = &mut st.kv[li];
            for &(off, len) in &cuts {
                exec.kv_append_batch_paged_rows(
                    &st.fkn,
                    pool_k,
                    d_pos,
                    Some(d_slots),
                    &st.d_bt,
                    st.bps,
                    kv_dim,
                    off,
                    len,
                    KvDtype::Fp16,
                )?;
                exec.kv_append_batch_paged_rows(
                    &st.fv,
                    pool_v,
                    d_pos,
                    Some(d_slots),
                    &st.d_bt,
                    st.bps,
                    kv_dim,
                    off,
                    len,
                    KvDtype::Fp16,
                )?;
            }
        }

        // coverage bookkeeping off the (possibly truncated) runs
        for &(row, len) in &runs {
            let slot = slots[row] as usize;
            let (p0, p1) = (positions[row], positions[row] + len as u32);
            let new_start = if len > keep { p1 - keep as u32 } else { p0 };
            let Some(&(s, e)) = st.feat.get(slot) else {
                continue;
            };
            // extendable only when this run joins the recorded span without
            // a hole AND doesn't reach back before its start (positions only
            // move forward in serving; the guard is belt-and-braces)
            let extend = p0 <= e && new_start <= e && new_start >= s;
            let cov = &mut st.cov[slot];
            let rows = &tokens[row + (new_start - p0) as usize..row + len];
            if extend {
                st.feat[slot] = (s, p1.max(e));
                // a re-drafted span can commit different tokens at positions
                // the record already holds, so overwrite rather than append
                let base = (new_start - s) as usize;
                cov.resize(base.max(cov.len()), 0);
                for (i, &t) in rows.iter().enumerate() {
                    match base + i < cov.len() {
                        true => cov[base + i] = t,
                        false => cov.push(t),
                    }
                }
                cov.truncate((p1.max(e) - s) as usize);
            } else {
                st.feat[slot] = (new_start, p1);
                cov.clear();
                cov.extend_from_slice(rows);
            }
        }
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[dflash-append] r={r} spans={} runs={} pos0={} posN={} feat={:?}",
                spans.is_some(),
                runs.len(),
                positions[0],
                positions[r - 1],
                st.feat
            );
        }
        Ok(())
    }

    /// Post-verify commit: replay the SERVICE's accept rule on this round's
    /// picks and ring-append only the accepted rows' features. The rule must
    /// stay bit-identical to service.rs's walk - the watermark tracks each
    /// slot's true committed point and warmth wants `end == pos` exactly.
    pub(crate) fn dflash_spec_commit(
        &mut self,
        reqs: &[(usize, usize, Vec<u32>)],
        picks: &[u32],
    ) -> Result<(), GpuError> {
        if !self.dflash_armed() {
            return Ok(());
        }
        let total: usize = reqs.iter().map(|q| q.2.len()).sum();
        debug_assert_eq!(total, picks.len());
        let mut positions = Vec::with_capacity(total);
        let mut slots = Vec::with_capacity(total);
        let mut spans = Vec::with_capacity(reqs.len());
        let mut base = 0usize;
        let mut toks = Vec::with_capacity(total);
        for (slot, start, chunk) in reqs {
            let mut a = 0usize;
            while a + 1 < chunk.len() && chunk[a + 1] == picks[base + a] {
                a += 1;
            }
            spans.push((base, a + 1));
            for j in 0..chunk.len() {
                positions.push((*start + j) as u32);
                slots.push(*slot as u32);
                // the row's INPUT token is what its feature encodes - for a
                // verify chunk that is the chunk itself (pending + drafts),
                // and only the accepted prefix of it is committed by `spans`
                toks.push(chunk[j]);
            }
            base += chunk.len();
        }
        self.dflash_append_features(&toks, &positions, &slots, Some(&spans))
    }

    /// Warmth for the service's spec gate: can the drafter draft for `slot`
    /// at the next round's start? There is deliberately no re-warm path -
    /// features flow from every batched forward while armed, so a cold slot
    /// means a genuinely un-walked span (a warm-resume tail shorter than the
    /// window). Verify rounds still run then; decode appends refill it.
    pub(crate) fn dflash_ensure_warm(&self, slot: usize, want_pos: u32) -> bool {
        let ok = self.dflash_armed() && self.dflash_warm(slot, want_pos as usize + 1);
        if !ok && paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            let feat = self
                .dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .and_then(|st| st.feat.get(slot).copied());
            tracing::info!(
                "[dflash-warm] slot={slot} want={want_pos} armed={} feat={feat:?}",
                self.dflash_armed()
            );
        }
        ok
    }

    /// Model-side drafts for one serving spec round: one DFlash forward
    /// covers every warm slot with block-1 drafts each. Slots that don't fit
    /// (cold, ctx-full, or past the per-round block cap) get empty lists and
    /// the service verifies them pending-only. `None` = nothing draftable,
    /// and the service falls back to its n-gram drafter.
    pub(crate) fn dflash_draft_batch(
        &mut self,
        pendings: &[(usize, u32)],
        k: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuError> {
        if !self.dflash_armed() || k == 0 {
            return Ok(None);
        }
        let (block, feat) = {
            let df = self.dflash.as_ref().expect("armed");
            (df.block, df.state.as_ref().expect("armed").feat.clone())
        };
        let cap = max_blocks().min(self.n_slots.max(1));
        // RUNTIME block = k drafts + the committed anchor, not the trained
        // block_size. llama.cpp does exactly this (common/speculative.cpp:
        // n_block_tokens = n_draft + 1, positions n..n+n_draft) and its own
        // default n_draft is 3 against this checkpoint's block_size of 16.
        // Building all 16 rows to use k+1 of them pays 16 rows of attention,
        // FFN and TARGET HEAD per round for nothing.
        let rows = k.min(block - 1) + 1;
        let mut reqs: Vec<(usize, usize, u32)> = Vec::new();
        let mut which: Vec<usize> = Vec::new();
        for (i, &(slot, tok)) in pendings.iter().enumerate() {
            let Some(&(_, e)) = feat.get(slot) else {
                continue;
            };
            let p = e as usize;
            if reqs.len() < cap && p + rows <= self.max_ctx && self.dflash_warm(slot, p) {
                reqs.push((slot, p, tok));
                which.push(i);
            }
        }
        if reqs.is_empty() {
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!(
                    "[dflash-draft] pendings={} eligible=0 k={k}",
                    pendings.len()
                );
            }
            return Ok(None);
        }
        let Some(blocks) = self.dflash_draft_blocks(&reqs, rows)? else {
            if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
                tracing::info!("[dflash-draft] round DECLINED n={} rows={rows}", reqs.len());
            }
            return Ok(None);
        };
        if paddock_models::dev_var_os!("PADDOCK_SPEC_DEBUG").is_some() {
            tracing::info!(
                "[dflash-draft] pendings={} eligible={} k={k} rows={rows} cap={cap} first={:?}",
                pendings.len(),
                reqs.len(),
                blocks.first().map(|b: &Vec<u32>| b.as_slice())
            );
        }
        let mut out = vec![Vec::new(); pendings.len()];
        for (w, b) in which.into_iter().zip(blocks) {
            debug_assert_eq!(b.len(), rows - 1, "one draft per noise row");
            out[w] = b;
        }
        Ok(Some(out))
    }

    /// One draft round: for each (slot, p, committed) request, one forward
    /// over [committed, (block-1) x mask] rows at positions p..p+block
    /// drafts the next block-1 tokens greedily.
    pub(crate) fn dflash_draft_blocks(
        &mut self,
        reqs: &[(usize, usize, u32)],
        rows: usize,
    ) -> Result<Option<Vec<Vec<u32>>>, GpuError> {
        if !self.dflash_armed() || !self.exec.has_argmax_rows() {
            return Ok(None);
        }
        let (block, mask) = {
            let df = self.dflash.as_ref().expect("armed");
            (df.block, df.mask_token)
        };
        let n = reqs.len();
        if n == 0 || rows < 2 || rows > block || n > max_blocks().min(self.n_slots.max(1)) {
            return Ok(None);
        }
        for &(slot, p, _) in reqs {
            if p + rows > self.max_ctx || !self.dflash_warm(slot, p) {
                return Ok(None);
            }
        }

        let r = n * rows;
        let mut toks = Vec::with_capacity(r);
        let mut pos = Vec::with_capacity(r);
        let mut apos = Vec::with_capacity(r);
        let mut slots_v = Vec::with_capacity(r);
        for &(slot, p, committed) in reqs {
            toks.push(committed);
            toks.extend(std::iter::repeat_n(mask, rows - 1));
            pos.extend(p as u32..(p + rows) as u32);
            // non-causal block: every row's attention bound is the block end
            apos.extend(std::iter::repeat_n((p + rows - 1) as u32, rows));
            slots_v.extend(std::iter::repeat_n(slot as u32, rows));
        }
        {
            let exec = self.exec.clone();
            let st = self
                .dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("armed");
            let up = |host: &[u32], dst: &mut CudaSlice<u32>| -> Result<(), GpuError> {
                let mut v = dst
                    .try_slice_mut(0..host.len())
                    .ok_or_else(|| GpuError::Driver("dflash row stage".into()))?;
                exec.stream.memcpy_htod(host, &mut v).map_err(drv)
            };
            up(&toks, &mut st.d_toks)?;
            up(&pos, &mut st.d_pos)?;
            up(&apos, &mut st.d_apos)?;
            up(&slots_v, &mut st.d_slots)?;
        }

        // One captured replay per block count: every per-row input is staged
        // device data and all shapes key on n alone. First sight of an n runs
        // eagerly (serving the round AND warming any lazy workspace before
        // recording), then records the identical launch stream.
        let have = self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .is_some_and(|st| st.graphs.contains_key(&(n, rows)));
        if paddock_models::dev_var_os!("PADDOCK_SPEC_NOGRAPH").is_some() {
            self.dflash_draft_body(n, rows)?;
        } else if !have {
            let exec = self.exec.clone();
            self.dflash_draft_body(n, rows)?;
            exec.stream
                .synchronize()
                .map_err(|e| GpuError::Driver(format!("dflash pre-capture sync: {e}")))?;
            exec.stream
                .begin_capture(
                    cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                )
                .map_err(|e| GpuError::Driver(format!("dflash begin_capture: {e}")))?;
            let rec = self.dflash_draft_body(n, rows);
            let graph = crate::gpu::end_capture_no_flags(&exec.stream)
                .map_err(|e| GpuError::Driver(format!("dflash end_capture: {e}")));
            rec?;
            let g = super::SendGraph(
                graph?
                    .ok_or_else(|| GpuError::Driver("dflash capture produced no graph".into()))?,
            );
            g.0.launch()
                .map_err(|e| GpuError::Driver(format!("dflash first launch: {e}")))?;
            self.dflash
                .as_mut()
                .and_then(|d| d.state.as_mut())
                .expect("armed")
                .graphs
                .insert((n, rows), g);
        } else {
            self.dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("armed")
                .graphs[&(n, rows)]
                .0
                .launch()
                .map_err(|e| GpuError::Driver(format!("dflash draft graph launch: {e}")))?;
        }

        let exec = self.exec.clone();
        let st = self
            .dflash
            .as_ref()
            .and_then(|d| d.state.as_ref())
            .expect("armed");
        let v = st
            .d_out
            .try_slice(0..r)
            .ok_or_else(|| GpuError::Driver("dflash d_out slice".into()))?;
        let picks = exec.stream.clone_dtoh(&v).map_err(drv)?;
        Ok(Some(
            (0..n)
                .map(|b| picks[b * rows + 1..(b + 1) * rows].to_vec())
                .collect(),
        ))
    }

    /// The draft round's device body (capture-safe): embed the staged rows,
    /// five drafter layers with transient block appends + ring attention,
    /// drafter final norm, TARGET head, device argmax.
    fn dflash_draft_body(&mut self, n: usize, rows: usize) -> Result<(), GpuError> {
        let exec = self.exec.clone();
        let embd = self.hp.n_embd;
        let vocab = self.hp.n_vocab;
        let r = n * rows;
        let head = &self.head;
        let head_f8t = self.head_f8t.as_ref();
        let head_f8row = self.head_f8row.as_ref();
        let df = self.dflash.as_mut().expect("armed");
        let (n_heads, n_kv, hd, eps, rope, window) =
            (df.n_heads, df.n_kv, df.hd, df.eps, df.rope, df.window);
        let (q_dim, kv_dim) = (n_heads * hd, n_kv * hd);
        let scale = 1.0 / (hd as f32).sqrt();
        let conv = if conv_off() { None } else { df.conv };
        let (logit_scale, softcap) = (df.logit_scale, df.softcap);
        let use_sel = !sel_off();
        let DflashDrafter {
            layers,
            final_norm,
            state,
            selector,
            ..
        } = df;
        let sel = use_sel
            .then_some(())
            .and(selector.as_ref().map(|(s, r, k)| (s, *r, *k)));
        let st = state.as_mut().expect("armed");
        let ff = match &layers[0].w_gate {
            QuantW::Q8(w) => w.dims[1],
            QuantW::Kq(w) => w.dims[1],
        };

        // RAW embedding rows from the target's table: llama.cpp's dflash
        // decoder is a plain `ggml_get_rows(tok_embd, tokens)` with no scale
        // and no embedding norm, whatever the target's own graph does with
        // its embeddings. The drafter was trained against that input.
        exec.embed_gather_plane(&self.token_embd, &st.d_toks, &mut st.x, embd, r, 1.0)?;

        for (li, layer) in layers.iter().enumerate() {
            exec.rmsnorm_batch(&st.x, &layer.attn_norm, &mut st.xn, embd, eps, r)?;
            exec.quantize_q8(&st.xn, &mut st.xq, &mut st.xs, r * embd)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.xn, &mut st.yq, embd, r)?;
            }
            // DFlash2 `prepare`: the projection reads the sublayer INPUT (this
            // norm's output, hence the quantize above serves it), then side 0
            // convolves that same input into what attention actually sees.
            // Side 1 stays in `cvc` for `finish` - it is derived from the
            // input too, not from the sublayer's output.
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.attn_conv) {
                draft_mm(
                    &exec,
                    &cv.proj,
                    &st.xq,
                    &st.xs,
                    &st.yq,
                    &mut st.xsums,
                    &mut st.ssums,
                    &mut st.part,
                    &mut st.skfix,
                    &mut st.cvc,
                    r,
                )?;
                exec.dflash_conv(
                    &st.xn,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    0,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.quantize_q8(&st.cvx, &mut st.xq, &mut st.xs, r * embd)?;
                if r > 64 {
                    exec.quantize_q8_mmq(&st.cvx, &mut st.yq, embd, r)?;
                }
            }
            for (w, dst) in [
                (&layer.wq, &mut st.q as *mut CudaSlice<f32>),
                (&layer.wk, &mut st.k as *mut CudaSlice<f32>),
                (&layer.wv, &mut st.v as *mut CudaSlice<f32>),
            ] {
                // SAFETY: the three destinations are distinct fields; the
                // raw pointers only exist to keep one loop over three planes
                // while `st` stays borrowed for the shared staging.
                let dst = unsafe { &mut *dst };
                draft_mm(
                    &exec,
                    w,
                    &st.xq,
                    &st.xs,
                    &st.yq,
                    &mut st.xsums,
                    &mut st.ssums,
                    &mut st.part,
                    &mut st.skfix,
                    dst,
                    r,
                )?;
            }
            exec.rmsnorm_batch(&st.q, &layer.q_norm, &mut st.qn, hd, eps, r * n_heads)?;
            exec.rmsnorm_batch(&st.k, &layer.k_norm, &mut st.kn, hd, eps, r * n_kv)?;
            exec.rope_yarn_batch(&mut st.qn, &st.d_pos, n_heads, hd, rope, r)?;
            exec.rope_yarn_batch(&mut st.kn, &st.d_pos, n_kv, hd, rope, r)?;
            // The block's own rows append transiently at p..p+block. Their
            // ring slots sit `ring` blocks away from every windowed read, so
            // append-before-attend is safe by construction and the next
            // round overwrites them.
            let (pool_k, pool_v) = &mut st.kv[li];
            exec.kv_append_batch_paged_rows(
                &st.kn,
                pool_k,
                &st.d_pos,
                Some(&st.d_slots),
                &st.d_bt,
                st.bps,
                kv_dim,
                0,
                r,
                KvDtype::Fp16,
            )?;
            exec.kv_append_batch_paged_rows(
                &st.v,
                pool_v,
                &st.d_pos,
                Some(&st.d_slots),
                &st.d_bt,
                st.bps,
                kv_dim,
                0,
                r,
                KvDtype::Fp16,
            )?;
            // Prefill (WMMA) class, not the decode kernel: the block is 16
            // rows over a 2048-key window, which is the tile walk's shape,
            // and the decode grid would re-read the window per row.
            // `d_apos` (block end) is what makes it non-causal.
            for b in 0..n {
                exec.attn_prefill_f16_rows_paged(
                    &st.qn,
                    pool_k,
                    pool_v,
                    &st.sinks,
                    &mut st.attn,
                    &st.d_apos,
                    &st.d_slots,
                    &st.d_bt,
                    st.bps,
                    n_heads,
                    n_kv,
                    hd,
                    kv_dim,
                    window,
                    b * rows,
                    rows,
                    scale,
                    KvDtype::Fp16,
                )?;
            }
            exec.quantize_q8(&st.attn, &mut st.xq, &mut st.xs, r * q_dim)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.attn, &mut st.yq, q_dim, r)?;
            }
            draft_mm(
                &exec,
                &layer.wo,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.proj,
                r,
            )?;
            // DFlash2 `finish`: side 1 of the same projection row, over the
            // sublayer output, before it joins the residual.
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.attn_conv) {
                exec.dflash_conv(
                    &st.proj,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    1,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.add(&mut st.x, &st.cvx, r * embd)?;
            } else {
                exec.add(&mut st.x, &st.proj, r * embd)?;
            }

            exec.rmsnorm_batch(&st.x, &layer.ffn_norm, &mut st.xn, embd, eps, r)?;
            exec.quantize_q8(&st.xn, &mut st.xq, &mut st.xs, r * embd)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.xn, &mut st.yq, embd, r)?;
            }
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.ffn_conv) {
                draft_mm(
                    &exec,
                    &cv.proj,
                    &st.xq,
                    &st.xs,
                    &st.yq,
                    &mut st.xsums,
                    &mut st.ssums,
                    &mut st.part,
                    &mut st.skfix,
                    &mut st.cvc,
                    r,
                )?;
                exec.dflash_conv(
                    &st.xn,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    0,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.quantize_q8(&st.cvx, &mut st.xq, &mut st.xs, r * embd)?;
                if r > 64 {
                    exec.quantize_q8_mmq(&st.cvx, &mut st.yq, embd, r)?;
                }
            }
            draft_mm(
                &exec,
                &layer.w_gate,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.ffn_gate,
                r,
            )?;
            draft_mm(
                &exec,
                &layer.w_up,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.ffn_up,
                r,
            )?;
            exec.swiglu(&mut st.ffn_gate, &st.ffn_up, r * ff)?;
            exec.quantize_q8(&st.ffn_gate, &mut st.xq, &mut st.xs, r * ff)?;
            if r > 64 {
                exec.quantize_q8_mmq(&st.ffn_gate, &mut st.yq, ff, r)?;
            }
            draft_mm(
                &exec,
                &layer.w_down,
                &st.xq,
                &st.xs,
                &st.yq,
                &mut st.xsums,
                &mut st.ssums,
                &mut st.part,
                &mut st.skfix,
                &mut st.proj,
                r,
            )?;
            if let (Some((taps, gsz, ng)), Some(cv)) = (conv, &layer.ffn_conv) {
                exec.dflash_conv(
                    &st.proj,
                    &mut st.cvx,
                    &cv.base,
                    &st.cvc,
                    1,
                    embd,
                    taps,
                    ng,
                    gsz,
                    rows,
                    r,
                )?;
                exec.add(&mut st.x, &st.cvx, r * embd)?;
            } else {
                exec.add(&mut st.x, &st.proj, r * embd)?;
            }
        }

        // drafter final norm -> the TARGET's head -> device argmax. The
        // target's logit_scale and final softcap are deliberately skipped:
        // both are monotone in the logit, so the argmax is identical and the
        // drafts are greedy picks, never sampled. DFlash2 ships its own pair
        // (`output_multiplier` / `final_logit_softcapping`, held on the
        // drafter as `logit_scale` / `softcap`) and the same argument retires
        // them here - but only while drafting stays greedy-per-row. The
        // candidate selector adds logits across a path, and addition does not
        // commute with a softcap, so stage C must apply the epilogue before
        // it scores anything.
        //
        // The head is the whole round: it reads the target's full lm_head
        // regardless of row count - roughly 70% of a round's time - which is
        // also why the round is flat across rows 2..16.
        // Ride the f8t tile plane exactly as every other head call site does
        // - half the bytes and the tuned kernel instead of the bf16 fallback.
        exec.rmsnorm_batch(&st.x, final_norm, &mut st.xn, embd, eps, r)?;
        match (head_f8t, head_f8row) {
            (Some(ht), _) => {
                exec.quantize_e4m3_row(&st.xn, &mut st.e4q, &mut st.e4rs, embd, r)?;
                exec.f8t_gemm(
                    ht,
                    &st.e4q,
                    &st.e4rs,
                    &mut st.skfix,
                    &mut st.logits,
                    embd,
                    vocab,
                    r,
                )?;
            }
            (None, Some(hr)) => {
                exec.quantize_e4m3_row(&st.xn, &mut st.e4q, &mut st.e4rs, embd, r)?;
                exec.f8row_gemm(hr, &st.e4q, &st.e4rs, &mut st.logits, embd, vocab, r)?;
            }
            (None, None) => head.gemm(&exec, &st.xn, &mut st.logits, r)?,
        }
        let Some((sc, rank, top_k)) = sel else {
            return exec.argmax_rows(&st.logits, &mut st.d_out, r, vocab);
        };
        // DFlash2's candidate selector. v1 (and v2 without this) takes a
        // per-row argmax, so the rows of a block are chosen independently and
        // can be individually plausible yet jointly incoherent - which is
        // precisely what the verifier rejects. Here the block is chosen as a
        // PATH: top-K candidates per row, a bilinear edge score from the
        // candidate actually taken one row back, greedy forward.
        let n = r / rows;
        exec.topk_rows(
            &st.logits,
            &st.sel_params,
            &mut st.sel_topk,
            r,
            vocab,
            top_k,
        )?;
        exec.dflash_cand_ids(
            &st.sel_topk,
            &st.d_toks,
            &mut st.sel_ids,
            top_k,
            rows,
            r,
            vocab,
        )?;
        let (QuantW::Kq(pw), QuantW::Kq(sw)) = (&sc.pred, &sc.succ) else {
            unreachable!("selector codebooks are k-quant - attach refuses anything else")
        };
        // One id plane feeds both gathers; pred also reads the anchor tail.
        exec.kquant_gather(pw, &st.sel_ids, &mut st.sel_pred, rank, r * top_k + n)?;
        exec.kquant_gather(sw, &st.sel_ids, &mut st.sel_succ, rank, r * top_k)?;
        // The selector's hidden projection reads the same final-normed hidden
        // the head just consumed, so `xn` is still the right plane.
        exec.quantize_q8(&st.xn, &mut st.xq, &mut st.xs, r * embd)?;
        if r > 64 {
            exec.quantize_q8_mmq(&st.xn, &mut st.yq, embd, r)?;
        }
        draft_mm(
            &exec,
            &sc.hidden,
            &st.xq,
            &st.xs,
            &st.yq,
            &mut st.xsums,
            &mut st.ssums,
            &mut st.part,
            &mut st.skfix,
            &mut st.sel_hs,
            r,
        )?;
        exec.dflash_select(
            &st.sel_topk,
            &st.sel_pred,
            &st.sel_succ,
            &st.sel_hs,
            &mut st.d_out,
            logit_scale,
            softcap,
            rank,
            top_k,
            rows,
            r,
        )
    }

    /// Synthetic end-to-end smoke (the bring-up gate, driven by
    /// `examples/muse_dflash.rs`): seed slot 0's ring with deterministic
    /// pseudo-features past the wrap point, draft twice (equality catches
    /// append races), and time the eager round. Real-feature acceptance is
    /// the serving harness's job.
    pub fn dflash_selftest(&mut self) -> Result<DflashSelftest, GpuError> {
        if self.dflash.is_none() {
            return Err(GpuError::Driver("dflash: not attached".into()));
        }
        self.dflash_ensure_state()?;
        let embd = self.hp.n_embd;
        let (window, block) = {
            let d = self.dflash.as_ref().expect("attached");
            (d.window, d.block)
        };
        // past the ring wrap, and past the window so the coverage gate opens
        let n = (window + block + 64).min(
            self.dflash
                .as_ref()
                .and_then(|d| d.state.as_ref())
                .expect("armed")
                .cap,
        );

        // deterministic xorshift features, zero-mean unit-ish scale
        let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 40) as f32 / (1u64 << 23) as f32 - 0.5
        };
        let host: Vec<f32> = (0..n * embd).map(|_| next()).collect();
        let positions: Vec<u32> = (0..n as u32).collect();
        let slots = vec![0u32; n];
        {
            let exec = self.exec.clone();
            let sc = &mut self.scratch;
            let mut dst = sc
                .pf_x
                .try_slice_mut(0..n * embd)
                .ok_or_else(|| GpuError::Driver("selftest pf_x slice".into()))?;
            exec.stream.memcpy_htod(&host, &mut dst).map_err(drv)?;
            let mut p = sc
                .pf_pos
                .try_slice_mut(0..n)
                .ok_or_else(|| GpuError::Driver("selftest pf_pos slice".into()))?;
            exec.stream.memcpy_htod(&positions, &mut p).map_err(drv)?;
            let mut sl = sc
                .pf_slots
                .try_slice_mut(0..n)
                .ok_or_else(|| GpuError::Driver("selftest pf_slots slice".into()))?;
            exec.stream.memcpy_htod(&slots, &mut sl).map_err(drv)?;
        }
        let n_taps = self.dflash.as_ref().expect("attached").fc_bands.len();
        for band in 0..n_taps {
            self.dflash_tap(band, n)?;
        }
        // synthetic driver: the pseudo-features aren't token-derived, so the
        // coverage record just gets a distinct id per row
        let stoks: Vec<u32> = (0..n as u32).collect();
        self.dflash_append_features(&stoks, &positions, &slots, None)?;

        let reqs = [(0usize, n, 1u32)];
        let rows = std::env::var("DFLASH_ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|r: &usize| (2..=block).contains(r))
            .unwrap_or(block);
        let d1 = self
            .dflash_draft_blocks(&reqs, rows)?
            .ok_or_else(|| GpuError::Driver("dflash selftest: draft declined".into()))?;
        let d2 = self
            .dflash_draft_blocks(&reqs, rows)?
            .ok_or_else(|| GpuError::Driver("dflash selftest: repeat declined".into()))?;
        let vocab = self.hp.n_vocab as u32;
        if d1[0].iter().any(|&t| t >= vocab) {
            return Err(GpuError::Driver(format!(
                "dflash selftest: draft out of vocab: {:?}",
                d1[0]
            )));
        }
        self.exec.synchronize()?;
        let t0 = std::time::Instant::now();
        let rounds = 50;
        for _ in 0..rounds {
            let _ = self.dflash_draft_blocks(&reqs, rows)?;
        }
        self.exec.synchronize()?;
        Ok(DflashSelftest {
            repeat_identical: d1 == d2,
            drafts: d1.into_iter().next().expect("one req"),
            ms_per_round: t0.elapsed().as_secs_f64() * 1e3 / rounds as f64,
        })
    }
}

/// Fold one tap band into the drafter's fusion accumulator, straight off the
/// residual rows the layer walk is holding. Called inside the captured step
/// body, so everything it touches is device-resident and every shape keys on
/// `r` alone.
///
/// One drafter GEMM, over whichever quant ladder actually covers this round's
/// width. Neither shared helper spans the drafter's range on its own:
///
///   * `mmq_pre_any` - the tick/decode ladder. Its k-quant arm has the
///     `kquant_gemm_mma_ks` K-split rung for `r <= 64` and then falls to
///     plain `dp4a` above it.
///   * `prefill_mm_pre_any` - the prefill ladder. Its k-quant arm reaches the
///     `w4a8_pipe2` double-buffered rung for `r > 64` and falls to
///     the same `dp4a` below it.
///
/// A draft round is `slots * (k + 1)` rows, which straddles 64 at around four
/// concurrent streams - so a single ladder leaves either the narrow or the
/// wide end on `dp4a`. Wiring only the tick ladder means every c8-and-wider
/// round runs the synchronous kernel while the round dutifully quantizes an
/// MMQ-layout activation plane that nothing reads.
///
/// INTERIM, and the gap is structural rather than in this function: the two
/// ladders should be one ladder that dispatches on width by itself, at which
/// point this collapses back to a single call. Splitting on `r` here is safe
/// for the drafter specifically - drafts are proposals the target's verify
/// re-derives, so a width-dependent rung moves acceptance, never output. The
/// same split would not be safe on a verify plane (see `mmq_pre`'s own note
/// on spec gates demanding one numeric class across the whole `r` range).
#[allow(clippy::too_many_arguments)]
fn draft_mm(
    exec: &crate::gpu::GpuExecutor,
    w: &QuantW,
    xq: &CudaSlice<i8>,
    xs: &CudaSlice<f32>,
    yq: &CudaSlice<u8>,
    xsums: &mut CudaSlice<f32>,
    ssums: &mut CudaSlice<f32>,
    part: &mut CudaSlice<f32>,
    skfix: &mut CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    r: usize,
) -> Result<(), GpuError> {
    // 64 is both ladders' own boundary, so this picks the better-covered side
    // of each rather than introducing a threshold of its own. `yq` is only
    // populated above it (the caller guards the same way), which is exactly
    // the range the prefill ladder consumes it in.
    if r > 64 {
        prefill_mm_pre_any(exec, w, xq, xs, yq, xsums, ssums, skfix, y, r)
    } else {
        mmq_pre_any(exec, w, xq, xs, ssums, part, y, r)
    }
    .map_err(|e| GpuError::Driver(e.to_string()))
}

/// `r > cap` cannot record the tick, so it marks the state stale and the
/// append then clears the slots it touched - a hole in the ring would let a
/// later round draft off features that never covered the window.
pub(crate) fn tap_band(
    exec: &crate::gpu::GpuExecutor,
    df: &mut DflashDrafter,
    pf_x: &CudaSlice<f32>,
    band: usize,
    embd: usize,
    r: usize,
) -> Result<(), GpuError> {
    let DflashDrafter {
        fc_bands, state, ..
    } = df;
    let st = state.as_mut().expect("armed");
    if r > st.cap {
        st.stale = true;
        return Ok(());
    }
    // Each band reads a different tap, so each quantizes its own rows; the
    // staging is shared only within one band's two forms.
    exec.quantize_q8(pf_x, &mut st.tq, &mut st.ts, r * embd)?;
    if r > 64 {
        exec.quantize_q8_mmq(pf_x, &mut st.tyq, embd, r)?;
    }
    let QuantW::Kq(kw) = &fc_bands[band] else {
        unreachable!("fc bands are k-quant - attach refuses anything else")
    };
    // band 0 writes the accumulator, the rest land in the twin and add
    let dst = if band == 0 {
        &mut st.zacc
    } else {
        &mut st.ztmp
    };
    kq_mm_pre(
        exec,
        kw,
        &st.tq,
        &st.ts,
        &st.tyq,
        &mut st.txs,
        &mut st.tss,
        dst,
        r,
    )
    .map_err(|e| GpuError::Driver(e.to_string()))?;
    if band > 0 {
        let (zacc, ztmp) = (&mut st.zacc, &st.ztmp);
        exec.add(zacc, ztmp, r * embd)?;
    }
    Ok(())
}
