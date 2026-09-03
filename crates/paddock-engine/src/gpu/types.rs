//! Device tensor / quant-plane types.

use cudarc::driver::CudaSlice;
use paddock_models::ggml_type::GgmlType;

/// An f32 tensor resident on the device. dims follow GGUF order (dims[0] =
/// row length / input dim).
pub struct DeviceTensor {
    pub buf: CudaSlice<f32>,
    pub dims: Vec<usize>,
}

impl DeviceTensor {
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// An f16 tensor resident on the device, same dims convention as
/// [`DeviceTensor`] (dims[0] = row length / input dim).
///
/// For weight planes that ship F16 on disk and are only ever consumed by a
/// tensor-core GEMM, this is the storage class the file already has: no widen
/// at load, half the resident bytes, and the GEMM accumulates in f32 anyway.
/// The vision towers are the users - they are dense-f16 mmproj files with no
/// quantization to speak of, so the k-quant machinery does not apply and a
/// plain half-precision plane is the whole story.
pub struct HalfTensor {
    pub buf: CudaSlice<half::f16>,
    pub dims: Vec<usize>,
}

impl HalfTensor {
    pub fn element_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// Resident bytes - what the f32 widen used to cost twice.
    pub fn bytes(&self) -> usize {
        self.element_count() * 2
    }
}

/// Narrow host f32 values to f16, refusing to produce an infinity a finite
/// input did not have.
///
/// Every mmproj we ship is F16 or BF16 on disk, and the two narrow very
/// differently. F16 -> f16 is the identity. **BF16 -> f16 is exact for every
/// value whose exponent fits** (bf16 keeps 7 mantissa bits, f16 keeps 10), but
/// bf16's exponent range is f32's, so a weight above 65504 comes back as `inf`
/// - and an `inf` weight does not produce a slightly-worse picture, it produces
///   NaN rows and an answer made of garbage, with nothing in the log saying why.
///   Trained tower weights live nowhere near that, which is exactly why this must
///   be checked rather than assumed: if it ever fires, the assumption was wrong
///   and the caller needs to hear it at load, not at the first image.
///
/// Underflow is deliberately not an error. Values under 2^-14 go subnormal and
/// under 2^-24 flush to zero; those are weights contributing less than the f32
/// accumulator's own rounding, and refusing them would reject every real file.
pub(super) fn narrow_to_f16(host: &[f32], what: &str) -> Result<Vec<half::f16>, super::GpuError> {
    let mut over = 0usize;
    let out: Vec<half::f16> = host
        .iter()
        .map(|&x| {
            let h = half::f16::from_f32(x);
            if x.is_finite() && !h.is_finite() {
                over += 1;
            }
            h
        })
        .collect();
    if over > 0 {
        return Err(super::GpuError::Driver(format!(
            "{what}: {over} of {} weights overflow f16 (|w| > 65504) - this plane cannot be \
             narrowed, it needs an f32 or bf16 lane",
            host.len()
        )));
    }
    Ok(out)
}

/// A tensor kept quantized on the device; slices dequant into scratch on use.
pub struct QuantTensor {
    pub bytes: CudaSlice<u8>,
    pub ty: GgmlType,
    pub dims: Vec<usize>,
}

impl QuantTensor {
    /// Bytes one `n`-element row of this tensor occupies on device - the
    /// stride a per-row `dequant_slice` offset is measured in. Q8_0 packs
    /// 32 elements into a 34-byte block; a bf16 plane is just 2 bytes an
    /// element. Call sites used to hardcode the Q8_0 form, which silently
    /// gathered the wrong row once mixed UD files brought bf16 embeddings in.
    pub fn row_bytes(&self, n: usize) -> usize {
        // block_layout answers for every type including the dense ones
        // (1 element per "block"), so ask it rather than assume Q8_0's 34
        match self.ty.block_layout() {
            Some((be, bb)) => (n / be) * bb,
            None => n * 4,
        }
    }
}

/// An MXFP4 expert weight repacked for the sorted MoE GEMM: the 16 data bytes of
/// each block laid out 16-aligned and contiguous (`data`), with the e8m0 scale
/// bytes split into a separate contiguous stream (`scale`). The on-disk 17-byte
/// stride misaligns every load (11.5/32 sectors); this makes it coalesced. Costs
/// ~16/17 of the original size again - kept only for the tensors the sorted path
/// reads (gate/up/down experts).
pub struct RepackedMxfp4 {
    pub data: CudaSlice<u8>,
    pub scale: CudaSlice<u8>,
}

impl RepackedMxfp4 {
    /// True when `data` holds tile-linear boxes (gemm/f8_lin.cuh) instead of
    /// row-major bytes: lin planes carry a 4-byte marker `scale` (the real
    /// scales live inside the boxes). Exec wrappers dispatch on this so call
    /// sites stay layout-blind; call-site arms that can't take the lin route
    /// (f32-activation gemvs) branch on it explicitly.
    pub fn is_lin(&self) -> bool {
        // 4 = strip lin (per-32 scales inside the boxes), 12 = rowwise lin
        // (data-only boxes + per-row exponent tail on `data`). Both are the
        // tile-linear box layout; exec wrappers dispatch the arm.
        self.scale.len() == 4 || self.scale.len() == 12
    }
}

/// A modelopt NVFP4 checkpoint plane, uploaded byte-for-byte from the
/// shipped triple: `data` [out, in/2] adjacent-packed e2m1, `scale`
/// [out, in/16] e4m3 block scales, `scale2` the per-tensor f32 global
/// scale. Unlike [`RepackedMxfp4`] it carries its dims and global scale -
/// checkpoint planes have per-TENSOR scale2 values that must ride with the
/// buffers (fold-into-e4m3 would be lossy; consumers apply scale2 once in
/// the epilogue, which is exact).
/// Device residency layout of an [`Nvf4Plane`]'s packed bytes (the scale
/// records are tile-major `[row][8B]` per block under both non-row layouts).
/// Set only by the upload functions; the exec wrappers dispatch the matching
/// kernel-twin ABI slots (bit-exact per class), and consumers that only
/// understand row-major refuse the others loudly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nvf4Layout {
    /// The shipped checkpoint order: `data` `[out, in/2]`, `scale`
    /// `[out, in/16]`, byte-for-byte.
    Row,
    /// TILE-MAJOR (the lm_head repack layout):
    /// `[row_tile 128][k_stage 128][row]` blocks, out padded to 128 rows and
    /// zero-filled, `in_dim % 128 == 0`. `nvf4_upload_tiled`.
    Tiled,
    /// FRAGMENT order: the tile-major blocks
    /// additionally permuted to `[w][k16][g][u32 of a0..a3 mma-fragment
    /// bytes per lane]`. `nvf4_upload_frag`.
    Frag,
}

pub struct Nvf4Plane {
    pub data: CudaSlice<u8>,
    pub scale: CudaSlice<u8>,
    pub scale2: f32,
    pub out_dim: usize,
    pub in_dim: usize,
    pub layout: Nvf4Layout,
    /// A merged gate|up plane whose rows are INTERLEAVED (row 2j = gate_j,
    /// row 2j+1 = up_j) at upload, so the prefill GEMM's swiglu + nvf4-quant
    /// epilogue (slot 533) sees each (gate, up) pair in adjacent rows. Every
    /// consumer of the plane's [rows, 2ff] output must read pairs (the `_il`
    /// twins, slots 534-536). Set only by the granite HF loader.
    pub gu_pairs: bool,
}

/// MoE expert-plane byte order - the [`Nvf4Layout`] analog for
/// [`Nvf4MoePlane`]. Set only by the upload functions; the exec wrappers
/// refuse a plane whose layout their kernel class cannot read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nvf4MoeLayout {
    /// The house MoE order: row of expert `e` at `e * ff + r`, data
    /// `[.., in/2]`, scale `[.., in/16]`, byte-for-byte from the checkpoint.
    Row,
    /// Piece-major 64x64 tiles (the skinny-tile pair):
    /// data `[e][rt 64-row][ks 64-el][piece 2][row 64][16 B]` (2048 B
    /// blocks), scale `[e][rt][ks][row 64][4 B]` (256 B blocks). Requires
    /// `ff % 64 == 0 && in_dim % 64 == 0` - both nemotron planes tile
    /// exactly, so byte count is identical to `Row`. `nvf4_moe_upload_tiled`;
    /// consumers are the `_st`/`_stw`/`_mtt` kernel family (slots 472-477).
    Tiled64,
}

/// A per-layer NVFP4 MoE expert residency: all experts of one
/// role concatenated into one plane (row of expert `e` at `e * ff + r` - the
/// house MoE layout), with the per-EXPERT scale2 factors as a device f32
/// array the kernels index in the epilogue (modelopt quantizes each expert
/// separately; folding scale2 into e4m3 would be lossy). The shared expert
/// rides the same struct at `n_expert == 1`.
pub struct Nvf4MoePlane {
    pub data: CudaSlice<u8>,
    pub scale: CudaSlice<u8>,
    pub scale2: CudaSlice<f32>,
    pub n_expert: usize,
    /// rows per expert (`ff` for up planes, `n_embd` for down planes)
    pub ff: usize,
    pub in_dim: usize,
    pub layout: Nvf4MoeLayout,
}

/// Per-ROW-scaled e4m3 weight plane (the sm_100 prefill class): e4m3 bytes +
/// one f32 power-of-2 scale per output row, applied in the GEMM epilogue -
/// no per-32 scale stream, no inner-loop fold.
pub struct F8RowPlane {
    pub data: CudaSlice<u8>,
    pub scale: CudaSlice<f32>,
}

/// An NVFP4 checkpoint plane made consumable by the CUTLASS sm100
/// block-scaled GEMM (`pd_nv4cut_gemm`, slots 462-465).
///
/// `data` is the checkpoint's own e2m1 nibbles, UNCHANGED - `[out_dim]
/// [in_dim/2]` bytes with element 2j in the low nibble, which is already
/// CUTLASS's k-major B operand. Only the scale vector is repacked: `sf`
/// holds the per-16 e4m3 scales scattered into CUTLASS's blocked SF layout
/// (`pd_nv4cut_sf_repack`, once at load). `alpha` is the plane's per-tensor
/// `scale2` (= 1/weight_global_scale), folded in the GEMM epilogue, so D
/// comes out dequantized.
pub struct Nvf4CutPlane {
    pub data: CudaSlice<u8>,
    pub sf: CudaSlice<u8>,
    pub alpha: f32,
    pub in_dim: usize,
    pub out_dim: usize,
}

/// The v4 decode plane: the same rowwise e4m3 payload as F8RowPlane
/// but with the SW128 smem image pre-baked into contiguous 16 KB tiles laid
/// (row_tile, k_slab)-major, so the decode GEMM streams the whole K walk as
/// one linear 1D-bulk sequence (TMA-2D's strided rows halve HBM-cold rate).
pub struct F8TilePlane {
    pub tiles: CudaSlice<u8>,
    pub scale: CudaSlice<f32>,
    /// flat k-major e4m3 twin for the vendored cutlass route
    /// (built at load when PADDOCK_F8CUT elects; None everywhere else)
    pub flat: Option<CudaSlice<u8>>,
    /// Per-plane batch floor for the cutlass intercept (0 = all widths).
    /// The qkv plane rides chunks-only: at decode widths the 64-row tile
    /// pads m=32 2x and the launch breaks the tc5 PDL chain.
    pub flat_minb: usize,
    /// P62 gluq: `flat` holds the gate/up-INTERLEAVED layout (row 2f = gate
    /// f, 2f+1 = up f). Every plain-cutlass consumer of `flat` must skip an
    /// interleaved plane - only the gluq export understands it; the classic
    /// tile-image routes are unaffected (they never read `flat`).
    pub flat_gui: bool,
    /// P62 gluq: interleaved twin of `scale` (matches the `flat_gui` row
    /// order). The tile image keeps the original `scale`.
    pub scale_il: Option<CudaSlice<f32>>,
}

/// A Q8_0 weight repacked for the vectorized decode GEMV: each block's 32 int8
/// values laid out contiguous + 16-aligned (`data`), scales split into a separate
/// f16 stream (`scale`). Same total bytes as the source (a reorganization) - lets
/// the GEMV load 16 weights per int4 transaction instead of byte-wise.
/// A GGUF k-quant weight (Q4_K/Q5_K/Q6_K) repacked at load into the pack's
/// aligned data stream + 24 B/super-block scale records (see 18_kquant.cuh).
/// Stays 4/5/6-bit RESIDENT - the stage-1 W4A8 route's VRAM win; the fused
/// GEMV reads it exactly (f32 products in-kernel, same values as reference
/// dequant up to reduction order).
pub struct RepackedKQ {
    pub data: CudaSlice<u8>,
    pub scales: CudaSlice<u8>,
    /// GGUF order: dims[0] = in_dim, dims[1] = out_dim.
    pub dims: Vec<usize>,
    pub ty: GgmlType,
}

/// (GGUF raw id, source block bytes, repacked data bytes) per 256-weight
/// super-block for the k-quant family; scale records are 24 B for all four.
pub(crate) fn kq_params(ty: GgmlType) -> Option<(u32, usize, usize)> {
    match ty {
        // Q4_0 rides as a degenerate super-block: 8 x 18-byte blocks = 144 B
        // raw per 256 weights (the same 144 as Q4_K), repacked into the Q4_K
        // data convention with its own {f16 dsub[8]} scale record. Native
        // format of the QAT lineage (Google's Gemma QAT trains at Q4_0), so
        // it is served first-class rather than as legacy PTQ.
        // Gate on GpuExecutor::has_kquant_q40 before routing here: the dtype
        // rides existing kernels, so only the capability slot can say.
        GgmlType::Q4_0 => Some((2, 144, 128)),
        GgmlType::Q4K => Some((12, 144, 128)),
        GgmlType::Q5K => Some((13, 176, 160)),
        GgmlType::Q6K => Some((14, 210, 192)),
        // IQ4_XS rides the same kernel family (nonlinear codebook, 4.25 bpw) -
        // UD-Q4_K_XL files mix it in for select ffn tensors.
        GgmlType::Iq4Xs => Some((23, 136, 128)),
        _ => None,
    }
}

/// A weight kept quantized-resident, dispatched per-TENSOR (UD/XL GGUF files
/// mix Q8_0 / Q4_K / Q5_K / Q6_K / IQ4_XS within one model - the loader seam
/// is per-tensor quant dispatch, per the quantization strategy).
pub enum QuantW {
    Q8(RepackedQ8),
    Kq(RepackedKQ),
}

impl QuantW {
    /// Exact resident bytes (data + scale streams). Use this for per-component
    /// VRAM accounting instead of differencing `cudaMemGetInfo` around a load:
    /// allocations come from a stream-ordered pool, so free-VRAM deltas track
    /// pool GROWTH, not individual tensors, and attribute whole pool blocks to
    /// whichever tensor happened to trigger them.
    pub fn bytes(&self) -> u64 {
        match self {
            QuantW::Q8(w) => (w.data.len() + w.scale.len()) as u64,
            QuantW::Kq(w) => (w.data.len() + w.scales.len()) as u64,
        }
    }

    /// GGUF order: dims[0] = in_dim, dims[1] = out_dim.
    pub fn dims(&self) -> &[usize] {
        match self {
            QuantW::Q8(w) => &w.dims,
            QuantW::Kq(w) => &w.dims,
        }
    }

    /// The Q8_0 repack, for call sites that only have a Q8_0 kernel class
    /// (batched serving, spec, MoE - the stage-2 W4A8 targets). Service
    /// routing keeps k-quant models on the serial path, so reaching one of
    /// those sites with a k-quant weight is a routing bug, not a user error.
    pub fn q8(&self) -> &RepackedQ8 {
        match self {
            QuantW::Q8(w) => w,
            QuantW::Kq(w) => panic!(
                "k-quant weight ({:?}) reached a Q8_0-only path - stage-1 routing \
                 must keep k-quant models on the serial spine",
                w.ty
            ),
        }
    }

    /// Some for the k-quant arm (the serial-path dispatch match).
    pub fn kq(&self) -> Option<&RepackedKQ> {
        match self {
            QuantW::Q8(_) => None,
            QuantW::Kq(w) => Some(w),
        }
    }
}

pub struct RepackedQ8 {
    pub data: CudaSlice<u8>,
    pub scale: CudaSlice<u8>,
    /// GGUF order: dims[0] = in_dim, dims[1] = out_dim.
    pub dims: Vec<usize>,
}

/// KV cache element type. `Fp16` is the greedy-exact default (2 bytes/elem);
/// `Fp8E4m3` (1 byte) is an opt-in throughput/memory mode - lossy (3 mantissa
/// bits, flips some tokens), validated by perplexity not exact match, and most
/// valuable on fp8-hardware arches (sm_89+). The discriminant matches the pack's
/// `PD_KV_*` enum and is passed to the batched KV kernels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvDtype {
    Fp16 = 0,
    Fp8E4m3 = 1,
}

impl KvDtype {
    /// Bytes per KV element - for cache allocation sizing and byte-offset math.
    pub fn bytes(self) -> usize {
        match self {
            KvDtype::Fp16 => 2,
            KvDtype::Fp8E4m3 => 1,
        }
    }
}

/// Which nonlinearity a gated FFN folds into its gate half: gemma4 is GeGLU
/// (the tanh-approximated GELU ggml uses), muse-glimmer is SwiGLU. This is a
/// model constant read from the file - never a knob and never a fallback. The
/// pack ships both instantiations of every carrier kernel (one `pd_glu_act`
/// template, two ABI slots), so picking one costs nothing at runtime and no
/// arch is ever served by the wrong activation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GluAct {
    Gelu,
    Silu,
}
