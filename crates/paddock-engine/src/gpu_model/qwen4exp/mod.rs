//! Qwen3.8-Flash-Next (`qwen4_exp`) - safetensors-primary NVFP4 family.
//!
//! Recon + rung log. 48 layers = 36 GDN +
//! 12 gated-attention-with-QSA (every 4th), 512 routed NVFP4 experts top-10 +
//! one bf16 shared expert per layer, 4-stream gated hyper-connections instead
//! of a residual, one PLE n-gram layer (decoder index 1), untied bf16 lm_head,
//! no final norm (the hyper-connection mixer feeds lm_head directly).
//!
//! CHARTER: bf16 byte-exact parity first - every residency
//! in this stage is bit-identical to the checkpoint (bf16 planes device-
//! resident as shipped, small vectors widened f32 exactly, expert nibbles
//! packed unchanged). The 8-bit dense-plane lane comes after parity, behind a
//! flag, quality-gated.
//!
//! Stage 2 = loader + GPU-side oracle only; the forward graph is the next
//! stage. MTP and vision planes are not loaded yet (spec/vision rungs).

mod forward;
mod load;

pub use forward::Qwen4ExpGpu;
pub use load::{load_layer, load_ple_projections, load_ple_table};

use cudarc::driver::CudaSlice;

use crate::gpu::{DeviceTensor, F8RowPlane, GpuError, GpuExecutor, Nvf4MoePlane, QuantTensor};

/// Which class the dense projections load in. bf16 is the parity class and
/// the default; anything else is the chartered throughput lane and is LOSSY,
/// so it is opt-in and every benchmark must stamp what it ran.
///
/// Chosen once at load from `PADDOCK_Q38FN_DENSE`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DenseClass {
    /// checkpoint bf16 bytes, device-resident unchanged
    #[default]
    Bf16,
    /// per-ROW e4m3: one f32 power-of-2 scale per output row, applied in the
    /// epilogue. Halves the plane and, at batch 1, `f8r_gemv` eats f32
    /// activations directly - so it adds no staging launch to a tick the
    /// profile showed is launch-sensitive. Lossy: coarser than the per-32
    /// block scales a Q8_0 lane would carry.
    F8Row,
}

impl DenseClass {
    pub fn label(self) -> &'static str {
        match self {
            DenseClass::Bf16 => "bf16",
            DenseClass::F8Row => "f8row",
        }
    }
    /// Bytes per weight element, for the residency ledger.
    pub fn elem_bytes(self) -> f32 {
        match self {
            DenseClass::Bf16 => 2.0,
            DenseClass::F8Row => 1.0,
        }
    }
}

/// Read the elected dense class from the environment. Unknown values are a
/// loud panic rather than a silent fallback - a benchmark that believes it
/// ran an 8-bit class while running bf16 is worse than a crash.
pub fn dense_class_from_env() -> DenseClass {
    static ELECTED: std::sync::OnceLock<DenseClass> = std::sync::OnceLock::new();
    *ELECTED.get_or_init(parse_dense_class)
}

fn parse_dense_class() -> DenseClass {
    match std::env::var("PADDOCK_Q38FN_DENSE").as_deref() {
        Err(_) | Ok("") | Ok("bf16") => DenseClass::Bf16,
        Ok("f8row") | Ok("f8r") => DenseClass::F8Row,
        Ok(other) => panic!("PADDOCK_Q38FN_DENSE={other:?} is not a dense class this build knows"),
    }
}

/// A dense weight plane in whichever class this lane elected for it.
///
/// `Bf16` is the PARITY class - the checkpoint's own bytes, device-resident
/// unchanged, and the class every gate in `tests/gpu_qwen4exp_*.rs` was
/// stamped against. The 8-bit variants are the chartered throughput lane
/// (dense planes may move to our 8-bit class after bf16
/// parity, behind a flag, quality-gated, with every benchmark stamping the class
/// honestly) - they are lossy, and nothing elects them by default.
///
/// Routing every dense projection through one type is what keeps the class a
/// LOAD-TIME decision instead of 20 call sites that can disagree.
pub enum DensePlane {
    /// checkpoint bf16, as shipped
    Bf16(QuantTensor),
    /// per-row e4m3 + f32 row scales. Carries its own dims: `F8RowPlane` has
    /// none, and a transposed GEMM off one is silent.
    F8Row {
        plane: F8RowPlane,
        in_dim: usize,
        out_dim: usize,
    },
}

/// Staging the 8-bit classes need when the activation must be quantized too
/// (the batch > 1 arm). Owned by the model, not the scratch, so a call site
/// can borrow an activation and the staging at the same time.
pub struct DenseStage {
    pub q: cudarc::driver::CudaSlice<i8>,
    pub rs: CudaSlice<f32>,
}

impl DensePlane {
    /// `y = W x` over `batch` rows. Same operand convention as the bf16 lane:
    /// `x` is `[batch, in_dim]`, `y` is `[batch, out_dim]`.
    pub fn matmul(
        &self,
        e: &GpuExecutor,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
        stage: &mut DenseStage,
    ) -> Result<(), GpuError> {
        match self {
            DensePlane::Bf16(w) => e.bf16_gemm(w, None, x, y, batch),
            DensePlane::F8Row {
                plane,
                in_dim,
                out_dim,
            } => {
                if batch == 1 {
                    // f32 activations straight in - no quantize launch, which
                    // is why this class was elected first on a tick whose
                    // profile is launch-sensitive.
                    e.f8r_gemv(plane, x, y, *in_dim, *out_dim)
                } else {
                    // W8A8: the row-scaled activation pair the batched GEMM
                    // wants. One staging launch per call here; the batch arm
                    // is not what this lane's benchmark runs.
                    e.quantize_e4m3_row(x, &mut stage.q, &mut stage.rs, *in_dim, batch)?;
                    e.f8row_gemm(plane, &stage.q, &stage.rs, y, *in_dim, *out_dim, batch)
                }
            }
        }
    }

    /// `y = W[first_row .. first_row+out_dim] x` - the batch > 1 arm of a
    /// launch-folded plane, which holds two projections in one residency and
    /// reads them as row segments when the fused single call cannot serve
    /// (a fused output is only contiguous per projection at batch 1).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_rows(
        &self,
        e: &GpuExecutor,
        first_row: usize,
        out_dim: usize,
        x: &CudaSlice<f32>,
        y: &mut CudaSlice<f32>,
        batch: usize,
    ) -> Result<(), GpuError> {
        match self {
            DensePlane::Bf16(w) => e.bf16_gemm_rows(w, first_row, out_dim, x, y, batch),
            DensePlane::F8Row { .. } => Err(GpuError::Unsupported(
                "matmul_rows: the 8-bit dense class holds no folded planes".into(),
            )),
        }
    }

    /// The plane's raw device bytes, when it is resident in the checkpoint's
    /// own class. `None` for the lossy classes, which hold a re-encoding -
    /// so a byte-identity oracle cannot silently pass on one.
    pub fn raw_bf16(&self) -> Option<&CudaSlice<u8>> {
        match self {
            DensePlane::Bf16(w) => Some(&w.bytes),
            DensePlane::F8Row { .. } => None,
        }
    }

    /// Device bytes this plane occupies - for the load-time residency ledger.
    pub fn bytes(&self) -> usize {
        match self {
            DensePlane::Bf16(w) => w.dims.iter().product::<usize>() * 2,
            DensePlane::F8Row {
                in_dim, out_dim, ..
            } => in_dim * out_dim + out_dim * 4,
        }
    }

    /// What to print alongside a result so the class is never implicit.
    pub fn class(&self) -> &'static str {
        match self {
            DensePlane::Bf16(_) => "bf16",
            DensePlane::F8Row { .. } => "f8row",
        }
    }
}

/// One hyper-connection sub-block (attn_ and mlp_ each have one; the final
/// mixer is the same minus `inject`).
pub struct HcW {
    /// group RMSNorm weight, (1+w) form, [4*hidden] f32
    pub norm: DeviceTensor,
    /// input_mix_weight_down [lowrank, 4*hidden] - with the `block_inject`
    /// rows APPENDED when `inject_rows > 0`. Both projections read the same
    /// normalized state, so one plane serves both and one launch covers them
    /// at batch 1. The inject was a [4, 10240] f32 matvec: 164 KB that cost
    /// 20.3 us because a 4-row grid is 4 blocks on a 148-SM card.
    pub down: DensePlane,
    /// rows of `down` that are the low-rank projection
    pub lowrank: usize,
    /// inject rows appended to `down`, or 0 when they were not folded (the
    /// 8-bit dense class, whose plane would QUANTIZE them, and the final
    /// mixer, which injects nothing)
    pub inject_rows: usize,
    /// input_mix_weight_up [4*hidden, lowrank]
    pub up: DensePlane,
    /// block_inject_weight [4, 4*hidden] f32, when it was not folded in
    pub inject: Option<DeviceTensor>,
}

/// GDN (gated DeltaNet) mixer - split planes, sigmoid output gate.
pub struct GdnW {
    /// in_proj_qkv [10240, hidden] (rows: q 2048 | k 2048 | v 6144)
    pub qkv: DensePlane,
    /// in_proj_z [6144, hidden] (output gate; bypasses the conv)
    pub z: DensePlane,
    /// in_proj_a and in_proj_b CONCATENATED: [2*v_heads, hidden] f32, rows
    /// [0, h) = a and [h, 2h) = b - which is exactly `delta_gate_ab`'s fused
    /// activation layout, so one matvec and one gate kernel replace two of
    /// each, at identical per-element math (deltanet/core.cuh:751).
    pub ab: DeviceTensor,
    /// conv1d [10240, 4] f32 (checkpoint [10240,1,4] squeezed); silu act
    pub conv: DeviceTensor,
    /// RAW A_log [48] f32 - the g = -exp(A_log)*softplus(a+dt_bias) fold is
    /// the graph's call, kept un-baked so the kernel lane stays flexible
    pub a_log: DeviceTensor,
    /// `-exp(A_log)`, the form `pd_delta_gate` consumes (it computes
    /// `g = ssm_a * softplus(a + dt_bias)`). Derived at load beside the raw
    /// plane above; 48 floats.
    pub ssm_a: DeviceTensor,
    pub dt_bias: DeviceTensor,
    /// gated-norm weight [128] f32 (plain w, sigmoid gate - Not qwen35 silu)
    pub norm: DeviceTensor,
    /// out_proj [hidden, 6144]
    pub out: DensePlane,
}

/// Gated attention + QSA indexer mixer.
pub struct AttnW {
    /// q_proj [12288, hidden] bf16 - PER-HEAD interleave: head h owns rows
    /// [h*512, (h+1)*512), first 256 = q, last 256 = the sigmoid output gate
    /// (vLLM qwen3_next._project_qkv_gate: view(heads, 512).chunk(2))
    pub q: DensePlane,
    /// k_proj / v_proj [512, hidden]
    pub k: DensePlane,
    pub v: DensePlane,
    /// o_proj [hidden, 6144]
    pub o: DensePlane,
    /// per-head q/k RMSNorm (1+w) [256] f32
    pub q_norm: DeviceTensor,
    pub k_norm: DeviceTensor,
    /// indexer.index_qk_proj [640, hidden] bf16 (rows: q 4x128 | k 1x128)
    pub idx_qk: QuantTensor,
    /// indexer per-head RMSNorms [128] f32
    pub idx_q_norm: DeviceTensor,
    pub idx_k_norm: DeviceTensor,
}

pub enum MixerW {
    Gdn(GdnW),
    Attn(AttnW),
}

/// MoE block: NVFP4 routed experts (bytes as shipped) + bf16 shared expert.
pub struct MoeW {
    /// router gate with the shared expert's scalar gate APPENDED as row
    /// `n_expert`: [n_expert+1, hidden] f32. Both read the same block input;
    /// the shared gate alone was a ONE-BLOCK launch costing 6.7 us for 10 KB.
    pub router: DeviceTensor,
    /// routed expert planes, 512 experts each, checkpoint nibbles unchanged
    pub gate: Nvf4MoePlane,
    pub up: Nvf4MoePlane,
    pub down: Nvf4MoePlane,
    /// shared expert [640, hidden] x2 + [hidden, 640]
    pub sh_gate: DensePlane,
    pub sh_up: DensePlane,
    pub sh_down: DensePlane,
}

/// PLE n-gram layer (decoder index 1): projections + the 51 GB fp8 table.
pub struct PleW {
    /// key_proj [4*hidden, hidden], value_proj [hidden, hidden]
    pub key: DensePlane,
    pub value: DensePlane,
    /// conv1d [4*hidden, 4] f32 (k=4, dilation 3 -> 9-token ring); silu
    pub conv: DeviceTensor,
    /// group norms (1+w) [4*hidden] f32
    pub norm_key: DeviceTensor,
    pub norm_query: DeviceTensor,
    pub norm_conv: DeviceTensor,
    /// the n-gram table: 128 shards concatenated, e4m3 bytes as shipped,
    /// [total_rows, 160]; None until `load_ple_table` runs (51 GB upload,
    /// kept separate so partial loads and the oracle stay cheap)
    pub table: Option<CudaSlice<u8>>,
    pub table_rows: usize,
    pub table_scale: f32,
    /// I64 hash constants, host-side - n-gram ids are computed on CPU before
    /// the forward starts (addresses depend only on token ids)
    pub multipliers: Vec<i64>,
    pub head_vocab: Vec<i64>,
    pub head_offset: Vec<i64>,
}

pub struct Qwen4ExpLayer {
    pub attn_hc: HcW,
    pub mlp_hc: HcW,
    pub mixer: MixerW,
    pub moe: MoeW,
    /// present only on PLE layers (decoder index 1); loaded separately from
    /// the 51 GB table via `load.rs` so partial loads stay cheap
    pub ple: Option<PleW>,
}
