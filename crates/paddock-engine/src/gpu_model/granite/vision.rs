//! Granite-Vision 4.1 vision stack (`clip` GGUF, projector `granite4_vision`).
//!
//! Two stages, and the second one is this model's whole contribution:
//!
//!   Stage A - a plain SigLIP tower. 384px / patch 16 -> 24×24 = 576 tokens,
//!   27 PRE-norm blocks, LayerNorm (weight AND bias, unlike our RMSNorm
//!   families), GELU FFN, learned absolute position table, no CLS token.
//!   Every block output is kept: the projectors tap four different depths.
//!
//!   Stage B - EIGHT independent windowed BLIP-2 Q-Formers, each with its own
//!   full weight set, run over the tower's outputs and emitting one stream
//!   apiece. Per block, for a 24×24 grid:
//!
//! ```text
//!     LN(1e-6) -> 3×3 windows of 8×8 (64 enc tokens each)
//!     downsample 24->12  ── taps 0..3: area interpolate (= 2×2 average pool)
//!                       └─ taps 4..7: pick one cell per 2×2 block, at
//!                          offsets TL/TR/BL/BR
//!     -> 3×3 windows of 4×4 (16 query tokens)
//!     query   = learned query[16,1152] + the downsampled window   <- CONDITIONED
//!     encoder = windowed feats + learned img_pos[64,1152]
//!     -> 1-layer Q-Former, POST-norm throughout (add then norm):
//!         post_norm(queries)                       <- input LN, despite the name
//!         self-attn(q,q,q)  + residual -> self_attn_norm
//!         cross-attn(q, enc, enc) + residual -> cross_attn_norm
//!         FFN up->gelu_erf->down + residual -> ffn_norm
//!     -> unwindow to 12×12 raster = 144 tokens
//!     -> linear 1152->2560
//! ```
//!
//! Details that are silent when wrong, all cross-checked against both the
//! upstream `downsampling.py`/`modeling.py` and llama.cpp's
//! `tools/mtmd/models/granite4-vision.cpp` (b10262, study reference only):
//!
//! - **Two different epsilons.** The block's top-level `norm` uses the vision
//!   eps (1e-6, from `clip.vision.attention.layer_norm_epsilon`); all four
//!   Q-Former norms use **1e-12**, the BLIP-2 config default. Upstream sets
//!   `nn.LayerNorm(..., eps=1e-6)` explicitly for the former and inherits
//!   1e-12 for the latter, and llama.cpp hard-codes `qformer_eps = 1e-12f`.
//! - **Two different head geometries.** The tower runs 16 heads × 72 dims
//!   (1152/16). The Q-Former hard-codes d_head = 64, giving **18** heads
//!   (`num_attention_heads = vision_hidden_size // 64` upstream). Reusing the
//!   tower's head count in the Q-Former is a plausible-looking mistake that
//!   changes every number without erroring.
//! - **`post_norm` is an INPUT norm.** Despite the name it is applied to the
//!   query embeddings before self-attention, not at the end of the block.
//! - **Windowing is a row gather, not a transpose.** llama.cpp precomputes
//!   index vectors on the host and uses `get_rows`; we do the same, which
//!   keeps the whole window/unwindow dance out of the kernels.
//! - **`v.post_ln` appears unused by this graph** - the projectors read raw
//!   per-layer outputs (`layer_outs[vlayer]`), and llama.cpp never applies the
//!   tower's final post-LN in the granite4_vision path. Loaded but not wired;
//!   revisit if parity is off by a whole-tensor normalization.
//!
//! Tap -> LLM-layer mapping lives in `deepstack.rs`; this module only produces
//! the eight streams.
//!
//! ## Precision
//!
//! Weight matrices live on the device at **F16 - the file's own storage class**
//! - and every GEMM runs f16 × f16 with an **f32 accumulate** (cuBLAS
//!   `CUBLAS_COMPUTE_32F`, i.e. tensor cores). Norms, biases and the learned
//!   tables stay f32 because the elementwise kernels are f32; only the GEMM
//!   operands are staged down, via one shared scratch per call.
//!
//! This replaced a load-time widen to f32 that cost twice the resident bytes
//! (the 1.16 GB mmproj occupied ~2.3 GB) AND ran every GEMM as cuBLAS SGEMM -
//! the A6000 does f32 at 38.7 TFLOPS and f16-with-f32-accumulate at 155.
//!
//! NUMERIC CLASS: this MOVES US TOWARD llama.cpp, which runs the same mmproj in
//! f16 - 22 of the 32 values its MTMD_DEBUG_EMBEDDINGS dump prints land
//! exactly on the f16 grid, and 0 of the old f32 ones did. We keep the more accurate accumulate class (f32) with
//! their storage class, so an exact bit-match is still not the bar - do not
//! read image-conditioned greedy divergence from llama.cpp as a bug here.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use paddock_models::gguf::Value;
use paddock_models::mapped::MappedGguf;

use super::preprocess::{AnyResPlan, PackRow, TileGeom};
use crate::gpu::{DeviceTensor, GpuError, GpuExecutor, HalfTensor};

/// One SigLIP tower block. LayerNorm carries a bias, and every projection has
/// one - this family is Linear-with-bias throughout, unlike our RMSNorm/no-bias
/// text stacks.
///
/// Weight matrices are [`HalfTensor`] (the file's own F16, un-widened); norms
/// and biases stay f32 because their consumers are the f32 elementwise ops.
pub struct TowerBlock {
    pub ln1_w: DeviceTensor,
    pub ln1_b: DeviceTensor,
    pub q_w: HalfTensor,
    pub q_b: DeviceTensor,
    pub k_w: HalfTensor,
    pub k_b: DeviceTensor,
    pub v_w: HalfTensor,
    pub v_b: DeviceTensor,
    pub o_w: HalfTensor,
    pub o_b: DeviceTensor,
    pub ln2_w: DeviceTensor,
    pub ln2_b: DeviceTensor,
    pub ff_up_w: HalfTensor,
    pub ff_up_b: DeviceTensor,
    pub ff_down_w: HalfTensor,
    pub ff_down_b: DeviceTensor,
}

/// One windowed Q-Former projector - eight of these, each fully independent.
pub struct ProjBlock {
    /// Top-level LN over the tower features, at the VISION eps (1e-6).
    pub norm_w: DeviceTensor,
    pub norm_b: DeviceTensor,
    /// Learned queries [16, 1152] - added to the downsampled window, not used
    /// as free queries.
    pub query: DeviceTensor,
    /// Learned per-window encoder positions [64, 1152].
    pub img_pos: DeviceTensor,
    /// Input LN on the query embeddings (name notwithstanding), eps 1e-12.
    pub post_norm_w: DeviceTensor,
    pub post_norm_b: DeviceTensor,
    pub sa_q_w: HalfTensor,
    pub sa_q_b: DeviceTensor,
    pub sa_k_w: HalfTensor,
    pub sa_k_b: DeviceTensor,
    pub sa_v_w: HalfTensor,
    pub sa_v_b: DeviceTensor,
    pub sa_o_w: HalfTensor,
    pub sa_o_b: DeviceTensor,
    pub sa_norm_w: DeviceTensor,
    pub sa_norm_b: DeviceTensor,
    pub ca_q_w: HalfTensor,
    pub ca_q_b: DeviceTensor,
    pub ca_k_w: HalfTensor,
    pub ca_k_b: DeviceTensor,
    pub ca_v_w: HalfTensor,
    pub ca_v_b: DeviceTensor,
    pub ca_o_w: HalfTensor,
    pub ca_o_b: DeviceTensor,
    pub ca_norm_w: DeviceTensor,
    pub ca_norm_b: DeviceTensor,
    pub ff_up_w: HalfTensor,
    pub ff_up_b: DeviceTensor,
    pub ff_down_w: HalfTensor,
    pub ff_down_b: DeviceTensor,
    pub ff_norm_w: DeviceTensor,
    pub ff_norm_b: DeviceTensor,
    /// 1152 -> 2560, into the LLM residual width.
    pub linear_w: HalfTensor,
    pub linear_b: DeviceTensor,
    /// Which tower block output this projector reads (already resolved to a
    /// 0-based block index - see `VisionHparams::feature_layers`).
    pub feature_layer: usize,
    /// -1 = area-interpolate downsampler; 0..3 = pick TL/TR/BL/BR from each
    /// 2×2 block.
    pub spatial_offset: i32,
}

/// Geometry, read from the file - never hard-coded, same rule as the text side.
pub struct VisionHparams {
    pub n_layers: usize,
    pub embd: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub image_size: usize,
    pub patch: usize,
    /// Patches per side (image_size / patch) - 24 here.
    pub grid: usize,
    /// Output width into the LLM (2560).
    pub proj_dim: usize,
    /// 8 - the Q-Former window side.
    pub window_side: usize,
    /// 4 - the query side per window.
    pub query_side: usize,
    /// Vision LayerNorm eps (1e-6). The Q-Former's 1e-12 is a constant, not a
    /// file field - see QFORMER_EPS.
    pub eps: f32,
    /// Per-projector tower tap, 0-based block index.
    pub feature_layers: Vec<usize>,
    /// Per-projector downsampler selector, -1 or 0..3.
    pub spatial_offsets: Vec<i32>,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// AnyRes grid choices as geometric (width, height) pairs, in FILE ORDER -
    /// the order breaks ties in `select_best_resolution`, so never sort it.
    /// The file stores HF's (height, width); see `preprocess`'s module note on
    /// why llama.cpp's opposite reading happens not to matter for this list.
    pub grid_pinpoints: Vec<(usize, usize)>,
}

/// The BLIP-2 Q-Former's LayerNorm eps. Not the vision eps and not in the file
/// - it is the `Blip2QFormerConfig` default that upstream inherits by
///   constructing the config without overriding it, and llama.cpp hard-codes the
///   same value.
pub const QFORMER_EPS: f32 = 1e-12;

/// The Q-Former's head dimension is fixed at 64 regardless of the tower's,
/// giving 1152/64 = 18 heads.
pub const QFORMER_HEAD_DIM: usize = 64;

pub struct VisionModel {
    exec: Arc<GpuExecutor>,
    pub hp: VisionHparams,
    /// Patch conv as GEMM: [3*patch*patch, embd] plus bias.
    pub patch_w: HalfTensor,
    pub patch_b: DeviceTensor,
    /// Learned absolute positions [576, embd].
    pub pos: DeviceTensor,
    pub blocks: Vec<TowerBlock>,
    pub projs: Vec<ProjBlock>,
    /// Loaded for completeness; see the module note - this graph does not use
    /// it, because the projectors read pre-post-LN block outputs.
    pub post_ln_w: DeviceTensor,
    pub post_ln_b: DeviceTensor,
    /// The learned row appended per tile row, at LLM width (2560).
    pub image_newline: DeviceTensor,
}

fn key_u64(map: &MappedGguf, key: &str) -> Result<usize, GpuError> {
    map.gguf()
        .metadata
        .get(key)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
        .ok_or_else(|| GpuError::Driver(format!("granite-vision mmproj missing {key}")))
}

fn key_arr_i64(map: &MappedGguf, key: &str) -> Result<Vec<i64>, GpuError> {
    match map.gguf().metadata.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_i64().ok_or_else(|| {
                    GpuError::Driver(format!("granite-vision mmproj {key}: non-integer entry"))
                })
            })
            .collect(),
        _ => Err(GpuError::Driver(format!(
            "granite-vision mmproj missing array {key}"
        ))),
    }
}

fn key_arr_f32(map: &MappedGguf, key: &str) -> Result<Vec<f32>, GpuError> {
    match map.gguf().metadata.get(key) {
        Some(Value::Array(items)) => items
            .iter()
            .map(|v| {
                v.as_f32().ok_or_else(|| {
                    GpuError::Driver(format!("granite-vision mmproj {key}: non-float entry"))
                })
            })
            .collect(),
        _ => Err(GpuError::Driver(format!(
            "granite-vision mmproj missing array {key}"
        ))),
    }
}

impl VisionModel {
    pub fn load(exec: Arc<GpuExecutor>, map: &MappedGguf) -> Result<Self, GpuError> {
        let proj_ty = map
            .gguf()
            .metadata
            .get("clip.projector_type")
            .and_then(Value::as_str)
            .unwrap_or("");
        if proj_ty != "granite4_vision" {
            return Err(GpuError::Driver(format!(
                "expected a granite4_vision mmproj, got projector_type {proj_ty:?} - \
                 granite-vision needs its own tower, not gemma4's or qwen35's"
            )));
        }

        let n_layers = key_u64(map, "clip.vision.block_count")?;
        let embd = key_u64(map, "clip.vision.embedding_length")?;
        let n_heads = key_u64(map, "clip.vision.attention.head_count")?;
        let ff = key_u64(map, "clip.vision.feed_forward_length")?;
        let image_size = key_u64(map, "clip.vision.image_size")?;
        let patch = key_u64(map, "clip.vision.patch_size")?;
        let proj_dim = key_u64(map, "clip.vision.projection_dim")?;
        let window_side = key_u64(map, "clip.vision.projector.window_side")?;
        let query_side = key_u64(map, "clip.vision.projector.query_side")?;
        let eps = map
            .gguf()
            .metadata
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(Value::as_f32)
            .unwrap_or(1e-6);

        let grid = image_size / patch;
        if grid * patch != image_size {
            return Err(GpuError::Driver(format!(
                "image_size {image_size} is not a whole number of {patch}px patches"
            )));
        }
        if grid % window_side != 0 {
            return Err(GpuError::Driver(format!(
                "grid {grid} is not divisible by window_side {window_side}"
            )));
        }

        // Tap resolution, and the trap it carries. config.json's
        // deepstack_layer_map indexes hidden_states, whose entry 0 is the
        // EMBEDDING output - so its -1 means block 26 here, and the GGUF's
        // feature_layer array is already shifted into 0-based block indices.
        // We consume the GGUF form and range-check it; a value that lands
        // outside the stack means the converter changed convention, which must
        // be loud rather than silently reading the wrong depth. (Same shape as
        // llama.cpp's DFlash target_layers off-by-one.)
        let feature_layers: Vec<usize> =
            key_arr_i64(map, "clip.vision.feature_layer")?
                .into_iter()
                .map(|v| {
                    usize::try_from(v).ok().filter(|&i| i < n_layers).ok_or_else(|| {
                    GpuError::Driver(format!(
                        "granite-vision feature_layer {v} outside the {n_layers}-block tower - \
                         the GGUF's indices are 0-based block outputs, config.json's are \
                         negative indices into hidden_states (entry 0 = embeddings); a shift \
                         between them reads the wrong depth silently"
                    ))
                })
                })
                .collect::<Result<_, _>>()?;
        let spatial_offsets: Vec<i32> = key_arr_i64(map, "clip.vision.projector.spatial_offsets")?
            .into_iter()
            .map(|v| v as i32)
            .collect();
        if feature_layers.len() != spatial_offsets.len() {
            return Err(GpuError::Driver(format!(
                "granite-vision projector arrays disagree: {} feature layers vs {} spatial \
                 offsets - they are parallel, one entry per Q-Former",
                feature_layers.len(),
                spatial_offsets.len()
            )));
        }
        let n_proj = feature_layers.len();

        let mean = key_arr_f32(map, "clip.vision.image_mean")?;
        let std = key_arr_f32(map, "clip.vision.image_std")?;
        if mean.len() != 3 || std.len() != 3 {
            return Err(GpuError::Driver("image_mean/std must be 3 channels".into()));
        }
        let pins = key_arr_i64(map, "clip.vision.image_grid_pinpoints")?;
        if pins.len() % 2 != 0 {
            return Err(GpuError::Driver(
                "image_grid_pinpoints must be (w,h) pairs".into(),
            ));
        }
        // stored (height, width) per HF; we work in geometric (width, height)
        let grid_pinpoints: Vec<(usize, usize)> = pins
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| (c[1] as usize, c[0] as usize))
            .collect();

        // Every weight matrix in this mmproj is F16 and every norm/bias F32.
        // `up` = f32 (the elementwise ops' operand class); `uph` = f16, kept in
        // the file's own storage class for the tensor-core GEMMs. Halving the
        // resident plane is the point, but the bigger effect is that the GEMMs
        // stop running as cuBLAS SGEMM.
        let up = |name: &str| -> Result<DeviceTensor, GpuError> { exec.upload(map, name) };
        let uph = |name: &str| -> Result<HalfTensor, GpuError> { exec.upload_f16(map, name) };

        let mut blocks = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let t = |s: &str| format!("v.blk.{i}.{s}");
            blocks.push(TowerBlock {
                ln1_w: up(&t("ln1.weight"))?,
                ln1_b: up(&t("ln1.bias"))?,
                q_w: uph(&t("attn_q.weight"))?,
                q_b: up(&t("attn_q.bias"))?,
                k_w: uph(&t("attn_k.weight"))?,
                k_b: up(&t("attn_k.bias"))?,
                v_w: uph(&t("attn_v.weight"))?,
                v_b: up(&t("attn_v.bias"))?,
                o_w: uph(&t("attn_out.weight"))?,
                o_b: up(&t("attn_out.bias"))?,
                ln2_w: up(&t("ln2.weight"))?,
                ln2_b: up(&t("ln2.bias"))?,
                ff_up_w: uph(&t("ffn_up.weight"))?,
                ff_up_b: up(&t("ffn_up.bias"))?,
                ff_down_w: uph(&t("ffn_down.weight"))?,
                ff_down_b: up(&t("ffn_down.bias"))?,
            });
        }

        let mut projs = Vec::with_capacity(n_proj);
        for i in 0..n_proj {
            let t = |s: &str| format!("v.proj_blk.{i}.{s}");
            projs.push(ProjBlock {
                norm_w: up(&t("norm.weight"))?,
                norm_b: up(&t("norm.bias"))?,
                query: up(&t("query"))?,
                img_pos: up(&t("img_pos"))?,
                post_norm_w: up(&t("post_norm.weight"))?,
                post_norm_b: up(&t("post_norm.bias"))?,
                sa_q_w: uph(&t("self_attn_q.weight"))?,
                sa_q_b: up(&t("self_attn_q.bias"))?,
                sa_k_w: uph(&t("self_attn_k.weight"))?,
                sa_k_b: up(&t("self_attn_k.bias"))?,
                sa_v_w: uph(&t("self_attn_v.weight"))?,
                sa_v_b: up(&t("self_attn_v.bias"))?,
                sa_o_w: uph(&t("self_attn_out.weight"))?,
                sa_o_b: up(&t("self_attn_out.bias"))?,
                sa_norm_w: up(&t("self_attn_norm.weight"))?,
                sa_norm_b: up(&t("self_attn_norm.bias"))?,
                ca_q_w: uph(&t("cross_attn_q.weight"))?,
                ca_q_b: up(&t("cross_attn_q.bias"))?,
                ca_k_w: uph(&t("cross_attn_k.weight"))?,
                ca_k_b: up(&t("cross_attn_k.bias"))?,
                ca_v_w: uph(&t("cross_attn_v.weight"))?,
                ca_v_b: up(&t("cross_attn_v.bias"))?,
                ca_o_w: uph(&t("cross_attn_out.weight"))?,
                ca_o_b: up(&t("cross_attn_out.bias"))?,
                ca_norm_w: up(&t("cross_attn_norm.weight"))?,
                ca_norm_b: up(&t("cross_attn_norm.bias"))?,
                ff_up_w: uph(&t("ffn_up.weight"))?,
                ff_up_b: up(&t("ffn_up.bias"))?,
                ff_down_w: uph(&t("ffn_down.weight"))?,
                ff_down_b: up(&t("ffn_down.bias"))?,
                ff_norm_w: up(&t("ffn_norm.weight"))?,
                ff_norm_b: up(&t("ffn_norm.bias"))?,
                linear_w: uph(&t("linear.weight"))?,
                linear_b: up(&t("linear.bias"))?,
                feature_layer: feature_layers[i],
                spatial_offset: spatial_offsets[i],
            });
        }

        let hp = VisionHparams {
            n_layers,
            embd,
            n_heads,
            head_dim: embd / n_heads,
            ff,
            image_size,
            patch,
            grid,
            proj_dim,
            window_side,
            query_side,
            eps,
            feature_layers,
            spatial_offsets,
            image_mean: [mean[0], mean[1], mean[2]],
            image_std: [std[0], std[1], std[2]],
            grid_pinpoints,
        };

        tracing::info!(
            tower_blocks = hp.n_layers,
            embd = hp.embd,
            heads = hp.n_heads,
            head_dim = hp.head_dim,
            qformer_heads = hp.embd / QFORMER_HEAD_DIM,
            grid = format!("{}x{}", hp.grid, hp.grid),
            projectors = n_proj,
            taps = ?hp.feature_layers,
            offsets = ?hp.spatial_offsets,
            eps = hp.eps,
            qformer_eps = QFORMER_EPS,
            "granite-vision tower loaded"
        );

        // The patch conv ships 4-D `[patch, patch, 3, embd]`; as a GEMM it is
        // `[3*patch*patch, embd]`. Reshape here, because `matvec_batch` reads
        // dims[0]/dims[1] as (in, out) and would otherwise silently take
        // (16, 16). The flattened input index is `x + patch*y + patch²*c`, so
        // the host patch buffer must be channel-major then row then column.
        let mut patch_w = uph("v.patch_embd.weight")?;
        let patch_in = 3 * patch * patch;
        if patch_w.element_count() != patch_in * embd {
            return Err(GpuError::Driver(format!(
                "v.patch_embd.weight has {} elements, expected 3*{patch}*{patch}*{embd} = {}",
                patch_w.element_count(),
                patch_in * embd
            )));
        }
        patch_w.dims = vec![patch_in, embd];

        let me = Self {
            patch_w,
            patch_b: up("v.patch_embd.bias")?,
            pos: up("v.position_embd.weight")?,
            post_ln_w: up("v.post_ln.weight")?,
            post_ln_b: up("v.post_ln.bias")?,
            image_newline: up("v.image_newline")?,
            blocks,
            projs,
            hp,
            exec: Arc::clone(&exec),
        };
        tracing::info!(
            weight_mib = me.weight_bytes() / (1 << 20),
            "granite-vision mmproj resident at f16 (f32 accumulate)"
        );
        Ok(me)
    }

    /// Device bytes the f16 weight planes hold - everything the GEMMs read.
    /// Equals the mmproj file's own weight bytes, which is the point: before
    ///  this was twice that. Norms/biases/tables are excluded (a few MB
    /// of f32, and they are not what the estimator gets wrong).
    pub fn weight_bytes(&self) -> usize {
        let blk: usize = self
            .blocks
            .iter()
            .map(|b| {
                b.q_w.bytes()
                    + b.k_w.bytes()
                    + b.v_w.bytes()
                    + b.o_w.bytes()
                    + b.ff_up_w.bytes()
                    + b.ff_down_w.bytes()
            })
            .sum();
        let prj: usize = self
            .projs
            .iter()
            .map(|p| {
                p.sa_q_w.bytes()
                    + p.sa_k_w.bytes()
                    + p.sa_v_w.bytes()
                    + p.sa_o_w.bytes()
                    + p.ca_q_w.bytes()
                    + p.ca_k_w.bytes()
                    + p.ca_v_w.bytes()
                    + p.ca_o_w.bytes()
                    + p.ff_up_w.bytes()
                    + p.ff_down_w.bytes()
                    + p.linear_w.bytes()
            })
            .sum();
        self.patch_w.bytes() + blk + prj
    }

    /// Projected tokens each Q-Former emits per 384px tile: the grid reduced by
    /// window_side/query_side on each axis (24 -> 12 here, so 144).
    pub fn tokens_per_tile(&self) -> usize {
        let side = self.hp.grid / self.hp.window_side * self.hp.query_side;
        side * side
    }

    /// Width of a projected row - the LLM's embedding width, which is what the
    /// streams are packed at.
    pub(crate) fn proj_width(&self) -> usize {
        self.hp.proj_dim
    }

    /// The largest image this tower can use, straight out of the AnyRes grid
    /// pinpoints the mmproj carries.
    ///
    /// Granite is the easy case and the only one of our three families that is
    /// fully self-describing here: `clip.vision.image_grid_pinpoints` is the
    /// list of resolutions the processor will select from, so the ceiling is
    /// the largest of them by area and the longest edge is the largest single
    /// side. Verified against IBM's own `preprocessor_config.json` (27 entries,
    /// 384x384 up to 3840x384 and its transpose).
    ///
    /// The floor is one base tile: below `image_size²` the plan still picks the
    /// smallest pinpoint and upsamples, so sending less buys nothing.
    pub fn budget(&self) -> crate::generator::VisionBudget {
        let (max_px, max_edge) = self
            .hp
            .grid_pinpoints
            .iter()
            .fold((0u64, 0u32), |(a, e), &(w, h)| {
                (a.max(w as u64 * h as u64), e.max(w.max(h) as u32))
            });
        let side = self.hp.image_size as u64;
        // Cost every pinpoint exactly rather than estimating: an image at a
        // pinpoint's size plans to that pinpoint, and `n_tokens` is the real
        // row count including the base tile and the spliced newlines. Pure
        // arithmetic, 27 iterations, done once at load.
        let (mut lo, mut hi) = (u32::MAX, 0u32);
        for &(w, h) in &self.hp.grid_pinpoints {
            if let Some(p) = AnyResPlan::new(w, h, self.tile_geom(), &self.hp.grid_pinpoints) {
                let n = p.n_tokens() as u32;
                lo = lo.min(n);
                hi = hi.max(n);
            }
        }
        crate::generator::VisionBudget {
            max_pixels: max_px,
            min_pixels: side * side,
            max_edge: Some(max_edge),
            // one 384² tile's worth of source per `tokens_per_tile` rows - the
            // ratio a caller prices an arbitrary image with, bounded by the
            // exact min/max above
            pixels_per_token: (side * side).div_ceil(self.tokens_per_tile() as u64),
            max_tokens: hi,
            min_tokens: if lo == u32::MAX { 0 } else { lo },
        }
    }

    /// DeepStack streams this mmproj emits, one per Q-Former (8 here). The
    /// budgeted lane sizes its accumulation from this rather than assuming.
    pub(crate) fn n_streams(&self) -> usize {
        self.projs.len()
    }

    /// Side of the downsampled grid (12 for a 24×24 tile).
    fn new_side(&self) -> usize {
        self.hp.grid / self.hp.window_side * self.hp.query_side
    }

    /// The AnyRes geometry this mmproj implies.
    pub fn tile_geom(&self) -> TileGeom {
        TileGeom {
            image_size: self.hp.image_size,
            tokens_side: self.new_side(),
        }
    }

    /// Plan one image's tiling and row layout. `None` only for a degenerate
    /// (zero-sided) image or a file with no pinpoints.
    pub fn plan(&self, w: usize, h: usize) -> Option<AnyResPlan> {
        AnyResPlan::new(w, h, self.tile_geom(), &self.hp.grid_pinpoints)
    }

    /// Rows this image will occupy in the prompt - what the caller must reserve
    /// as `<image>` placeholders before anything is encoded. Equals HF's
    /// `_get_number_of_features`.
    pub fn image_tokens(&self, w: usize, h: usize) -> Result<usize, GpuError> {
        self.plan(w, h)
            .map(|p| p.n_tokens())
            .ok_or_else(|| GpuError::Driver(format!("granite-vision: cannot plan a {w}x{h} image")))
    }
}

/// The four gather tables the Q-Former runs on, built once per tile count and
/// shared by all eight projectors (they depend only on geometry, not weights).
///
/// Every table is replicated per tile with its source AND destination offsets
/// shifted, so one launch covers a whole AnyRes tile set - a tile is an
/// independent image as far as windowing is concerned.
///
/// Index values are non-negative and go to a kernel typed `int32_t`; u32 and
/// i32 share bit patterns over this range, and `to_device_u32` is the existing
/// upload path.
pub struct WindowIndices {
    /// raster -> window-major over the full grid (24×24 in 3×3 windows of 8×8).
    win: CudaSlice<u32>,
    /// raster -> window-major over the downsampled grid (12×12 in 3×3 of 4×4).
    qwin: CudaSlice<u32>,
    /// the inverse of `qwin` - window-major back to 12×12 raster.
    unwin: CudaSlice<u32>,
    /// 4-way fan-in for the area-interpolate downsampler (2×2 average pool).
    pool: CudaSlice<u32>,
    /// one table per 2×2 offset TL/TR/BL/BR, indexed by `spatial_offset`.
    spatial: Vec<CudaSlice<u32>>,
    n_tiles: usize,
}

/// `dst[w*win² + p] = y*side + x` - the raster->window-major permutation, for
/// one tile. Mirrors llama.cpp's `make_win_idx` exactly (clip.cpp).
fn win_idx(side: usize, win: usize) -> Vec<u32> {
    let nn = side / win;
    let mut idx = vec![0u32; side * side];
    for wy in 0..nn {
        for wx in 0..nn {
            for iy in 0..win {
                for ix in 0..win {
                    let w = wy * nn + wx;
                    let p = iy * win + ix;
                    let (y, x) = (wy * win + iy, wx * win + ix);
                    idx[w * (win * win) + p] = (y * side + x) as u32;
                }
            }
        }
    }
    idx
}

impl WindowIndices {
    /// Build every table for `n_tiles` tiles of the model's geometry.
    pub fn build(v: &VisionModel, n_tiles: usize) -> Result<Self, GpuError> {
        let exec = &v.exec;
        let (grid, wside, qside) = (v.hp.grid, v.hp.window_side, v.hp.query_side);
        let new_side = v.new_side();
        let (src_rows, dst_rows) = (grid * grid, new_side * new_side);

        // Replicate a one-tile table across tiles. Only the SOURCE index needs
        // shifting - destination rows are already laid out consecutively per
        // tile, so tile t's outputs land where the caller expects them.
        let tile = |base: &[u32], src_stride: usize| {
            let mut out = Vec::with_capacity(base.len() * n_tiles);
            for t in 0..n_tiles {
                out.extend(base.iter().map(|&i| i + (t * src_stride) as u32));
            }
            out
        };

        let win_base = win_idx(grid, wside);
        let qwin_base = win_idx(new_side, qside);
        let mut unwin_base = vec![0u32; qwin_base.len()];
        for (i, &f) in qwin_base.iter().enumerate() {
            unwin_base[f as usize] = i as u32;
        }

        // 2×2 average pool: TL, TR, BL, BR per output cell - the same order
        // ggml's pool_2d accumulates in (kernel row outer, column inner).
        let mut pool_base = Vec::with_capacity(dst_rows * 4);
        for y in 0..new_side {
            for x in 0..new_side {
                for dy in 0..2 {
                    for dx in 0..2 {
                        pool_base.push(((y * 2 + dy) * grid + (x * 2 + dx)) as u32);
                    }
                }
            }
        }

        // offset o selects cell (o>>1, o&1) of each 2×2 block - llama.cpp's
        // make_spatial_idx. The order matters: 2 is BL, not TR.
        let mut spatial = Vec::with_capacity(4);
        for o in 0..4usize {
            let (off_y, off_x) = ((o >> 1) & 1, o & 1);
            let mut base = Vec::with_capacity(dst_rows);
            for y in 0..new_side {
                for x in 0..new_side {
                    base.push(((y * 2 + off_y) * grid + (x * 2 + off_x)) as u32);
                }
            }
            spatial.push(exec.to_device_u32(&tile(&base, src_rows))?);
        }

        Ok(Self {
            win: exec.to_device_u32(&tile(&win_base, src_rows))?,
            qwin: exec.to_device_u32(&tile(&qwin_base, dst_rows))?,
            unwin: exec.to_device_u32(&tile(&unwin_base, dst_rows))?,
            pool: exec.to_device_u32(&tile(&pool_base, src_rows))?,
            spatial,
            n_tiles,
        })
    }
}

/// One encoded media item's projected streams, each `[tokens, width]`.
///
/// Vision fills one stream per Q-Former (`[n_tiles * tokens_per_tile, 2560]`)
/// and the DeepStack injection consumes them positionally: stream `i` is
/// added at the LLM layer that projector `i` targets. Granite-speech fills
/// exactly one - its projector output replaces the `<|audio|>` rows and there
/// is no mid-stack injection at all (that checkpoint's `deepstack_mapping`
/// is all -1), so `apply_embed` is the whole story there.
pub struct MediaFeatures {
    pub streams: Vec<CudaSlice<f32>>,
    /// Rows per stream - `n_tiles * tokens_per_tile`, before any newline rows.
    pub tokens: usize,
    pub width: usize,
}

impl VisionModel {
    /// Normalize interleaved RGB u8 into the planar f32 a tile wants:
    /// `out[c][y][x] = (px/255 - mean[c]) / std[c]`.
    pub fn normalize_rgb(&self, rgb: &[u8], w: usize, h: usize) -> Vec<f32> {
        assert_eq!(rgb.len(), 3 * w * h, "expected tightly-packed RGB");
        let mut out = vec![0f32; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    let v = rgb[(y * w + x) * 3 + c] as f32 / 255.0;
                    out[c * w * h + y * w + x] = (v - self.hp.image_mean[c]) / self.hp.image_std[c];
                }
            }
        }
        out
    }

    /// Cut a planar tile into patch rows for the conv-as-GEMM. Row order is
    /// raster over patches; within a row the 768 values run channel-major, then
    /// patch row, then column - the flattening `patch_w`'s reshape assumes.
    fn patch_rows(&self, tiles: &[Vec<f32>]) -> Vec<f32> {
        let (grid, patch, side) = (self.hp.grid, self.hp.patch, self.hp.image_size);
        let (k2, plane) = (patch * patch, side * side);
        let mut out = vec![0f32; tiles.len() * grid * grid * 3 * k2];
        let mut row = 0usize;
        for img in tiles {
            for py in 0..grid {
                for px in 0..grid {
                    let dst = &mut out[row * 3 * k2..(row + 1) * 3 * k2];
                    for c in 0..3 {
                        for ky in 0..patch {
                            let src = c * plane + (py * patch + ky) * side + px * patch;
                            dst[c * k2 + ky * patch..c * k2 + ky * patch + patch]
                                .copy_from_slice(&img[src..src + patch]);
                        }
                    }
                    row += 1;
                }
            }
        }
        out
    }

    /// Run the SigLIP tower and return the outputs of the DISTINCT tapped
    /// blocks, keyed by block index. Only 4 of 27 outputs are ever read (the 8
    /// projectors share block 26 four ways), so keeping the whole stack resident
    /// would be ~7× the memory for nothing.
    fn tower(&self, tiles: &[Vec<f32>]) -> Result<HashMap<usize, CudaSlice<f32>>, GpuError> {
        let exec = &self.exec;
        let hp = &self.hp;
        let (e, n) = (hp.embd, hp.grid * hp.grid);
        let rows = tiles.len() * n;

        let mut taps: Vec<usize> = hp.feature_layers.clone();
        taps.sort_unstable();
        taps.dedup();

        let d_patches = exec.to_device(&self.patch_rows(tiles))?;
        let mut d_x = exec.alloc(rows * e)?;
        // One f16 staging buffer for every GEMM's activations, sized by the
        // widest thing a block feeds a GEMM (the FFN's 4304) and rewritten in
        // sequence. Conversions are the price of an f32 elementwise chain
        // driving f16 tensor-core GEMMs; each is a single streaming pass over
        // rows that the GEMM is about to read many times over.
        let stage = rows * hp.ff.max(e).max(self.patch_w.dims[0]);
        let mut s16 = exec.alloc_f16(stage)?;
        exec.convert_f32_f16(&d_patches, &mut s16, rows * self.patch_w.dims[0])?;
        exec.matvec_batch_f16(&self.patch_w, &s16, &mut d_x, rows)?;
        exec.bias_add(&mut d_x, &self.patch_b.buf, rows, e)?;
        // Learned absolute positions, the same 576 rows for every tile.
        exec.add_rows_bcast(&mut d_x, &self.pos.buf, rows, n, e)?;

        let mut d_n = exec.alloc(rows * e)?;
        let mut d_q = exec.alloc(rows * e)?;
        let mut d_k = exec.alloc(rows * e)?;
        let mut d_v = exec.alloc(rows * e)?;
        let mut d_a = exec.alloc(rows * e)?;
        let mut d_up = exec.alloc(rows * hp.ff)?;
        let scale = 1.0 / (hp.head_dim as f32).sqrt();

        let mut out = HashMap::new();
        for (li, blk) in self.blocks.iter().enumerate() {
            exec.layernorm(
                &d_x,
                &blk.ln1_w.buf,
                &blk.ln1_b.buf,
                &mut d_n,
                rows,
                e,
                hp.eps,
            )?;
            // q, k and v all read the same normed rows - stage them once
            exec.convert_f32_f16(&d_n, &mut s16, rows * e)?;
            exec.matvec_batch_f16(&blk.q_w, &s16, &mut d_q, rows)?;
            exec.matvec_batch_f16(&blk.k_w, &s16, &mut d_k, rows)?;
            exec.matvec_batch_f16(&blk.v_w, &s16, &mut d_v, rows)?;
            exec.bias_add(&mut d_q, &blk.q_b.buf, rows, e)?;
            exec.bias_add(&mut d_k, &blk.k_b.buf, rows, e)?;
            exec.bias_add(&mut d_v, &blk.v_b.buf, rows, e)?;
            // patches attend only within their own tile
            for t in 0..tiles.len() {
                exec.vision_attn_at(
                    &d_q,
                    &d_k,
                    &d_v,
                    &mut d_a,
                    t * n,
                    n,
                    hp.n_heads,
                    hp.head_dim,
                    scale,
                )?;
            }
            exec.convert_f32_f16(&d_a, &mut s16, rows * e)?;
            exec.matvec_batch_f16(&blk.o_w, &s16, &mut d_n, rows)?;
            exec.bias_add(&mut d_n, &blk.o_b.buf, rows, e)?;
            exec.add(&mut d_x, &d_n, rows * e)?;

            exec.layernorm(
                &d_x,
                &blk.ln2_w.buf,
                &blk.ln2_b.buf,
                &mut d_n,
                rows,
                e,
                hp.eps,
            )?;
            exec.convert_f32_f16(&d_n, &mut s16, rows * e)?;
            exec.matvec_batch_f16(&blk.ff_up_w, &s16, &mut d_up, rows)?;
            exec.bias_add(&mut d_up, &blk.ff_up_b.buf, rows, hp.ff)?;
            // tanh GELU here (clip.use_gelu), erf GELU in the Q-Former - see
            // the module note; they are not interchangeable.
            exec.gelu(&mut d_up, rows * hp.ff)?;
            exec.convert_f32_f16(&d_up, &mut s16, rows * hp.ff)?;
            exec.matvec_batch_f16(&blk.ff_down_w, &s16, &mut d_n, rows)?;
            exec.bias_add(&mut d_n, &blk.ff_down_b.buf, rows, e)?;
            exec.add(&mut d_x, &d_n, rows * e)?;

            if taps.binary_search(&li).is_ok() {
                let mut keep = exec.alloc(rows * e)?;
                exec.copy_region(&d_x, 0, &mut keep, 0, rows * e)?;
                out.insert(li, keep);
            }
        }
        Ok(out)
    }

    /// One windowed Q-Former: `h` is a tapped tower output `[n_tiles*576, C]`,
    /// the result is `[n_tiles*144, 2560]`.
    fn project(
        &self,
        p: &ProjBlock,
        h: &CudaSlice<f32>,
        idx: &WindowIndices,
    ) -> Result<CudaSlice<f32>, GpuError> {
        let exec = &self.exec;
        let hp = &self.hp;
        let (c, n_tiles) = (hp.embd, idx.n_tiles);
        let new_side = self.new_side();
        let n_win = (hp.grid / hp.window_side) * (hp.grid / hp.window_side);
        let (enc_len, query_len) = (
            hp.window_side * hp.window_side,
            hp.query_side * hp.query_side,
        );
        let e_rows = n_tiles * n_win * enc_len; // == n_tiles * 576
        let q_rows = n_tiles * n_win * query_len; // == n_tiles * 144
        let n_batch = n_tiles * n_win;
        let heads = c / QFORMER_HEAD_DIM;
        let scale = 1.0 / (QFORMER_HEAD_DIM as f32).sqrt();
        let src_rows = n_tiles * hp.grid * hp.grid;

        // 1. top-level LN, at the VISION eps (not the Q-Former's)
        let mut x = exec.alloc(src_rows * c)?;
        exec.layernorm(h, &p.norm_w.buf, &p.norm_b.buf, &mut x, src_rows, c, hp.eps)?;

        // 2. encoder tokens: window-major gather, then + learned positions
        let mut enc = exec.alloc(e_rows * c)?;
        exec.gather_rows_avg(&x, &idx.win, &mut enc, e_rows, 1, c)?;
        exec.add_rows_bcast(&mut enc, &p.img_pos.buf, e_rows, enc_len, c)?;

        // 3. downsample 24->12: average pool, or pick one cell per 2×2 block.
        //    Same op, different index table - that is the whole difference
        //    between the two projector families.
        let mut down = exec.alloc(n_tiles * new_side * new_side * c)?;
        if p.spatial_offset < 0 {
            exec.gather_rows_avg(
                &x,
                &idx.pool,
                &mut down,
                n_tiles * new_side * new_side,
                4,
                c,
            )?;
        } else {
            let t = idx.spatial.get(p.spatial_offset as usize).ok_or_else(|| {
                GpuError::Driver(format!(
                    "granite-vision spatial_offset {} is not one of -1 (area interpolate) or \
                     0..3 (2×2 cell pick)",
                    p.spatial_offset
                ))
            })?;
            exec.gather_rows_avg(&x, t, &mut down, n_tiles * new_side * new_side, 1, c)?;
        }

        // 4. queries are CONDITIONED: learned table + the downsampled window
        let mut q_in = exec.alloc(q_rows * c)?;
        exec.gather_rows_avg(&down, &idx.qwin, &mut q_in, q_rows, 1, c)?;
        exec.add_rows_bcast(&mut q_in, &p.query.buf, q_rows, query_len, c)?;

        // 6. the single Q-Former layer, POST-norm (add then norm) throughout.
        //    `post_norm` is the INPUT norm despite its name.
        let mut qn = exec.alloc(q_rows * c)?;
        exec.layernorm(
            &q_in,
            &p.post_norm_w.buf,
            &p.post_norm_b.buf,
            &mut qn,
            q_rows,
            c,
            QFORMER_EPS,
        )?;

        let mut qq = exec.alloc(q_rows * c)?;
        let mut qk = exec.alloc(q_rows * c)?;
        let mut qv = exec.alloc(q_rows * c)?;
        let mut attn = exec.alloc(q_rows * c)?;
        let mut proj = exec.alloc(q_rows * c)?;
        // Shared f16 GEMM staging, as in `tower`. The widest operand here is
        // whichever is larger: the encoder rows the cross-attention K/V read
        // (4× the query rows) or the FFN's intermediate.
        let mut s16 = exec.alloc_f16((e_rows * c).max(q_rows * p.ff_up_w.dims[1]))?;

        // 6a. self-attention among the 16 queries of each window
        exec.convert_f32_f16(&qn, &mut s16, q_rows * c)?;
        exec.matvec_batch_f16(&p.sa_q_w, &s16, &mut qq, q_rows)?;
        exec.matvec_batch_f16(&p.sa_k_w, &s16, &mut qk, q_rows)?;
        exec.matvec_batch_f16(&p.sa_v_w, &s16, &mut qv, q_rows)?;
        exec.bias_add(&mut qq, &p.sa_q_b.buf, q_rows, c)?;
        exec.bias_add(&mut qk, &p.sa_k_b.buf, q_rows, c)?;
        exec.bias_add(&mut qv, &p.sa_v_b.buf, q_rows, c)?;
        exec.vision_attn_x(
            &qq,
            &qk,
            &qv,
            &mut attn,
            query_len,
            query_len,
            heads,
            QFORMER_HEAD_DIM,
            n_batch,
            scale,
        )?;
        exec.convert_f32_f16(&attn, &mut s16, q_rows * c)?;
        exec.matvec_batch_f16(&p.sa_o_w, &s16, &mut proj, q_rows)?;
        exec.bias_add(&mut proj, &p.sa_o_b.buf, q_rows, c)?;
        exec.add(&mut proj, &qn, q_rows * c)?;
        let mut sa = exec.alloc(q_rows * c)?;
        exec.layernorm(
            &proj,
            &p.sa_norm_w.buf,
            &p.sa_norm_b.buf,
            &mut sa,
            q_rows,
            c,
            QFORMER_EPS,
        )?;

        // 6b. cross-attention: 16 queries read the window's 64 encoder tokens
        let mut ek = exec.alloc(e_rows * c)?;
        let mut ev = exec.alloc(e_rows * c)?;
        exec.convert_f32_f16(&sa, &mut s16, q_rows * c)?;
        exec.matvec_batch_f16(&p.ca_q_w, &s16, &mut qq, q_rows)?;
        // K and V read the ENCODER rows, not the queries - a second staging
        exec.convert_f32_f16(&enc, &mut s16, e_rows * c)?;
        exec.matvec_batch_f16(&p.ca_k_w, &s16, &mut ek, e_rows)?;
        exec.matvec_batch_f16(&p.ca_v_w, &s16, &mut ev, e_rows)?;
        exec.bias_add(&mut qq, &p.ca_q_b.buf, q_rows, c)?;
        exec.bias_add(&mut ek, &p.ca_k_b.buf, e_rows, c)?;
        exec.bias_add(&mut ev, &p.ca_v_b.buf, e_rows, c)?;
        exec.vision_attn_x(
            &qq,
            &ek,
            &ev,
            &mut attn,
            query_len,
            enc_len,
            heads,
            QFORMER_HEAD_DIM,
            n_batch,
            scale,
        )?;
        exec.convert_f32_f16(&attn, &mut s16, q_rows * c)?;
        exec.matvec_batch_f16(&p.ca_o_w, &s16, &mut proj, q_rows)?;
        exec.bias_add(&mut proj, &p.ca_o_b.buf, q_rows, c)?;
        exec.add(&mut proj, &sa, q_rows * c)?;
        let mut ca = exec.alloc(q_rows * c)?;
        exec.layernorm(
            &proj,
            &p.ca_norm_w.buf,
            &p.ca_norm_b.buf,
            &mut ca,
            q_rows,
            c,
            QFORMER_EPS,
        )?;

        // 6c. FFN - exact-erf GELU, unlike the tower
        let ff = p.ff_up_w.dims[1];
        let mut up = exec.alloc(q_rows * ff)?;
        exec.convert_f32_f16(&ca, &mut s16, q_rows * c)?;
        exec.matvec_batch_f16(&p.ff_up_w, &s16, &mut up, q_rows)?;
        exec.bias_add(&mut up, &p.ff_up_b.buf, q_rows, ff)?;
        exec.gelu_erf(&mut up, q_rows * ff)?;
        exec.convert_f32_f16(&up, &mut s16, q_rows * ff)?;
        exec.matvec_batch_f16(&p.ff_down_w, &s16, &mut proj, q_rows)?;
        exec.bias_add(&mut proj, &p.ff_down_b.buf, q_rows, c)?;
        exec.add(&mut proj, &ca, q_rows * c)?;
        let mut ffo = exec.alloc(q_rows * c)?;
        exec.layernorm(
            &proj,
            &p.ff_norm_w.buf,
            &p.ff_norm_b.buf,
            &mut ffo,
            q_rows,
            c,
            QFORMER_EPS,
        )?;

        // 7-8. back to raster order, then out to the LLM width
        let mut raster = exec.alloc(q_rows * c)?;
        exec.gather_rows_avg(&ffo, &idx.unwin, &mut raster, q_rows, 1, c)?;
        let mut out = exec.alloc(q_rows * hp.proj_dim)?;
        exec.convert_f32_f16(&raster, &mut s16, q_rows * c)?;
        exec.matvec_batch_f16(&p.linear_w, &s16, &mut out, q_rows)?;
        exec.bias_add(&mut out, &p.linear_b.buf, q_rows, hp.proj_dim)?;
        Ok(out)
    }

    /// Encode AnyRes tiles into the 8 DeepStack streams. Each tile is planar
    /// normalized f32 `[3, image_size, image_size]` (see `normalize_rgb`).
    ///
    /// TILE INDEPENDENCE - the property two features ride on. A tile never sees
    /// another tile: `vision_attn_at` runs one attention per tile, and
    /// `WindowIndices::build` replicates the window tables per tile with shifted
    /// offsets. So encoding a SLICE of a stack produces exactly the rows that
    /// slice would have contributed to the whole, bit for bit. That is what
    /// makes N pictures one pass and what makes an encoder budget
    /// possible at all  - a budgeted group is just a slice.
    pub fn encode(&self, tiles: &[Vec<f32>]) -> Result<MediaFeatures, GpuError> {
        if tiles.is_empty() {
            return Err(GpuError::Driver("granite-vision encode: no tiles".into()));
        }
        let want = 3 * self.hp.image_size * self.hp.image_size;
        for (i, t) in tiles.iter().enumerate() {
            if t.len() != want {
                return Err(GpuError::Driver(format!(
                    "granite-vision tile {i} has {} floats, expected 3*{}² = {want}",
                    t.len(),
                    self.hp.image_size
                )));
            }
        }
        let idx = WindowIndices::build(self, tiles.len())?;
        let taps = self.tower(tiles)?;
        let mut streams = Vec::with_capacity(self.projs.len());
        for p in &self.projs {
            let h = taps.get(&p.feature_layer).ok_or_else(|| {
                GpuError::Driver(format!(
                    "tower output for block {} was not kept",
                    p.feature_layer
                ))
            })?;
            streams.push(self.project(p, h, &idx)?);
        }
        Ok(MediaFeatures {
            streams,
            tokens: tiles.len() * self.tokens_per_tile(),
            width: self.hp.proj_dim,
        })
    }

    /// The whole path for one picture: plan -> tile -> normalize -> encode ->
    /// pack. The result's `tokens` is the number of `<image>` placeholder rows
    /// the prompt must have reserved (see [`Self::image_tokens`], which answers
    /// the same question without touching the GPU).
    pub fn encode_image(&self, rgb: &[u8], w: usize, h: usize) -> Result<MediaFeatures, GpuError> {
        let mut out = self.encode_images(std::slice::from_ref(&(rgb, w, h)))?;
        Ok(out.pop().expect("one in, one out"))
    }

    /// Encode SEVERAL pictures in one tower + Q-Former pass.
    ///
    /// The tower already batches tiles - one AnyRes picture is 1..11 of them -
    /// and a tile is an independent image as far as the windowing tables are
    /// concerned. So N pictures are just more tiles: `WindowIndices` replicates
    /// per tile with shifted offsets either way, and the projector output stays
    /// tile-major, which makes each picture's slice of it a contiguous row
    /// range. That is the whole trick, and it is why this is a concatenation
    /// rather than a second code path through the model.
    ///
    /// Why it matters: encoding a 640x440 chart costs ~206 ms, so four
    /// different pictures arriving together cost ~0.8 s of strictly serial work
    /// before the first token of any of them (the same TTFT staircase already
    /// fixed for qwen35). Batched, they share one pass over the 27-block
    /// tower and all 8 Q-Formers.
    pub fn encode_images(
        &self,
        images: &[(&[u8], usize, usize)],
    ) -> Result<Vec<MediaFeatures>, GpuError> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let (plans, tiles, bases) = self.tile_stack(images)?;
        let streams = self.encode(&tiles)?.streams;
        self.pack_wave(
            &streams,
            tiles.len() * self.tokens_per_tile(),
            &plans,
            &bases,
        )
    }

    /// Host-side preparation for a batched encode: each image's AnyRes plan, the
    /// concatenated normalized tile stack, and each image's first TILE in it.
    ///
    /// Split out from [`Self::encode_images`] because it touches no device at
    /// all - which is what lets the encoder-budget lane decide how
    /// much of the stack to run in a given tick without redoing this work.
    pub(crate) fn tile_stack(
        &self,
        images: &[(&[u8], usize, usize)],
    ) -> Result<(Vec<AnyResPlan>, Vec<Vec<f32>>, Vec<usize>), GpuError> {
        let side = self.hp.image_size;
        let mut plans = Vec::with_capacity(images.len());
        let mut tiles: Vec<Vec<f32>> = Vec::new();
        // tile_base per image - its first tile in the batched, tile-major output
        let mut bases = Vec::with_capacity(images.len());
        for &(rgb, w, h) in images {
            let plan = self.plan(w, h).ok_or_else(|| {
                GpuError::Driver(format!("granite-vision: cannot plan a {w}x{h} image"))
            })?;
            bases.push(tiles.len());
            tiles.extend(
                plan.tiles(rgb, w, h)
                    .iter()
                    .map(|t| self.normalize_rgb(t, side, side)),
            );
            plans.push(plan);
        }
        Ok((plans, tiles, bases))
    }

    /// Turn a FINISHED tile stack's 8 streams back into per-image features.
    ///
    /// `rows` is the stack's total projected rows (`n_tiles * tokens_per_tile`).
    /// The newline row is appended once for the whole stack rather than per
    /// picture - every image's pack gather reads the same source buffer.
    pub(crate) fn pack_wave(
        &self,
        streams: &[CudaSlice<f32>],
        rows: usize,
        plans: &[AnyResPlan],
        bases: &[usize],
    ) -> Result<Vec<MediaFeatures>, GpuError> {
        let src = self.with_newline(streams, rows)?;
        let nl_row = rows as u32;
        plans
            .iter()
            .zip(bases)
            .map(|(plan, &base)| self.pack_from(&src, nl_row, base, plan))
            .collect()
    }

    /// Copy each stream into a buffer with the learned `image_newline` appended
    /// as one extra row - the source layout [`Self::pack_from`]'s gather reads.
    fn with_newline(
        &self,
        streams: &[CudaSlice<f32>],
        rows: usize,
    ) -> Result<Vec<CudaSlice<f32>>, GpuError> {
        let (exec, c) = (&self.exec, self.hp.proj_dim);
        streams
            .iter()
            .map(|s| {
                let mut buf = exec.alloc((rows + 1) * c)?;
                exec.copy_region(s, 0, &mut buf, 0, rows * c)?;
                exec.copy_region(&self.image_newline.buf, 0, &mut buf, rows * c, c)?;
                Ok(buf)
            })
            .collect()
    }

    /// Reorder one image's tile-major rows into the LLM's row layout, splicing
    /// in `image_newline` rows - `pack_and_unpad_image_features`.
    ///
    /// Done as a row gather, the same trick the windowing uses: the newline is
    /// one extra source row, so a single index vector expresses both "feature
    /// row" and "newline" with no branch and no second kernel.
    ///
    /// `tile_base` is the image's first TILE in `src`, which is 0 for a
    /// single-image encode and the running offset when a wave was batched.
    fn pack_from(
        &self,
        src: &[CudaSlice<f32>],
        nl_row: u32,
        tile_base: usize,
        plan: &AnyResPlan,
    ) -> Result<MediaFeatures, GpuError> {
        let exec = &self.exec;
        let (c, per_tile) = (self.hp.proj_dim, self.tokens_per_tile());
        let idx: Vec<u32> = plan
            .rows()
            .into_iter()
            .map(|r| match r {
                PackRow::Feature { tile, idx } => ((tile_base + tile) * per_tile + idx) as u32,
                PackRow::Newline => nl_row,
            })
            .collect();
        // Every gathered row must be inside the batched buffer, newline row
        // included. Out of range would silently read a neighbouring picture's
        // features - the one failure this whole slicing scheme can produce.
        let src_rows = nl_row as usize + 1;
        if let Some(&bad) = idx.iter().find(|&&i| i as usize >= src_rows) {
            return Err(GpuError::Driver(format!(
                "granite-vision pack: row {bad} is outside the {src_rows}-row encode (tile_base \
                 {tile_base}) - the batched slice and the plan disagree"
            )));
        }
        let n = idx.len();
        let d_idx = exec.to_device_u32(&idx)?;
        let mut streams = Vec::with_capacity(src.len());
        for s in src {
            let mut out = exec.alloc(n * c)?;
            exec.gather_rows_avg(s, &d_idx, &mut out, n, 1, c)?;
            streams.push(out);
        }
        Ok(MediaFeatures {
            streams,
            tokens: n,
            width: c,
        })
    }
}
