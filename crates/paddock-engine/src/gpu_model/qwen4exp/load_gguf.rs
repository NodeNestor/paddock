//! Qwen3.8-Flash-Next off a llama.cpp `qwen4exp` GGUF - the Unsloth UD
//! exports (UD-IQ1_S, UD-Q2_K_XL, UD-Q4_K_XL, ...) and any requant of them.
//!
//! The safetensors loader (`load.rs`) is the parity lane and stays as it is;
//! this is the consumer-card lane: every dense projection lands as a
//! `DensePlane::Kq` on the qwen35 k-quant/Q8_0 dense streams, the routed
//! experts as k-quant / i-quant seats on the repacked MoE stream - in VRAM,
//! or host-mapped under `[moe_offload]` with the slot cache
//! (`enable_moe_cache`) - and the PLE n-gram table stays a host mmap of the
//! GGUF, gathered per token on the CPU (`forward.rs::gather_ple_rows`).
//!
//! What the converter did to the bytes, and what this loader does about it
//! (llama.cpp `conversion/qwen4exp.py` + the `_LinearAttentionVReorderBase`
//! it inherits):
//! - GDN value heads are in TILED order (qkv V rows, z, a/b, A, dt_bias,
//!   the conv's V channels, out_proj's columns). The pack's DeltaNet kernels
//!   read exactly that order (`hk = hv % n_k_heads`), so nothing is permuted
//!   here.
//! - `A_log` is stored as `-exp(A_log)` (`ssm_a`) - which is the form the
//!   gate kernel consumes; `a_log` is recovered as `ln(-ssm_a)`.
//! - Every `*norm.weight` except the GDN's own carries the Gemma `+1`. The
//!   pack applies `(1+w)` itself for the hyper-connection, PLE and indexer
//!   norms (`q4x_group_norm_1p`), so those are handed back as raw `w`; the
//!   attention q/k norms are consumed in `(1+w)` form and pass through.
//! - The indexer's fused `index_qk_proj` was split into `indexer.q_proj` and
//!   `indexer.k_proj`; they are joined back (rows q then k, bf16 as shipped).
//! - The shared-expert scalar gate (`ffn_gate_inp_shexp`) is appended to the
//!   router as row `n_expert`, as the safetensors loader does.
//!
//! Unknown shape or type is an error, never a default.

use std::sync::Arc;

use paddock_models::ggml_type::GgmlType;
use paddock_models::mapped::MappedGguf;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig, Qwen4ExpPleHash};

use crate::gpu::{DeviceTensor, GpuExecutor, QuantTensor, RepackedKQ};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::{
    AttnW, DensePlane, Embed, ExpertSeats, GdnW, HcW, KqSeat, MixerW, MoeW, PleW, Qwen4ExpLayer,
};

fn unsupported(m: String) -> GpuModelError {
    GpuModelError::Unsupported(m)
}

/// GGUF dims of a tensor, `[ne0, ne1, ...]` (ne0 = the contiguous dim).
fn dims_of(map: &MappedGguf, name: &str) -> Result<Vec<usize>, GpuModelError> {
    let (info, _) = map
        .tensor_bytes(name)
        .map_err(|e| unsupported(format!("{name}: {e}")))?;
    Ok(info.dims.iter().map(|&d| d as usize).collect())
}

fn want_dims(map: &MappedGguf, name: &str, want: &[usize]) -> Result<(), GpuModelError> {
    let d = dims_of(map, name)?;
    if d != want {
        return Err(unsupported(format!("{name}: dims {d:?}, want {want:?}")));
    }
    Ok(())
}

/// An F32 tensor's values, checked to hold exactly `want` elements.
fn f32_vec(map: &MappedGguf, name: &str, want: usize) -> Result<Vec<f32>, GpuModelError> {
    let (info, bytes) = map
        .tensor_bytes(name)
        .map_err(|e| unsupported(format!("{name}: {e}")))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(unsupported(format!(
            "{name}: type {:?}, want F32",
            info.ggml_type
        )));
    }
    if bytes.len() != want * 4 {
        return Err(unsupported(format!(
            "{name}: {} bytes for {want} f32",
            bytes.len()
        )));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

fn f32_dt(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    dims: Vec<usize>,
) -> Result<DeviceTensor, GpuModelError> {
    let n: usize = dims.iter().product();
    super::load::dt(exec, f32_vec(map, name, n)?, dims).map_err(GpuModelError::from)
}

/// A norm the converter stored as `(1+w)` for a kernel that applies the `+1`
/// itself: hand back `w`.
fn f32_dt_m1(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    dims: Vec<usize>,
) -> Result<DeviceTensor, GpuModelError> {
    let n: usize = dims.iter().product();
    let v: Vec<f32> = f32_vec(map, name, n)?.into_iter().map(|w| w - 1.0).collect();
    super::load::dt(exec, v, dims).map_err(GpuModelError::from)
}

/// A dense projection `[n (out), k (in)]` on the k-quant / Q8_0 dense
/// streams. GGUF dims are `[k, n]`.
fn dense(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    n: usize,
    k: usize,
) -> Result<DensePlane, GpuModelError> {
    want_dims(map, name, &[k, n])?;
    let w = exec
        .load_quantw(map, name)
        .map_err(|e| unsupported(format!("{name}: {e}")))?;
    Ok(DensePlane::Kq {
        w,
        in_dim: k,
        out_dim: n,
    })
}

/// Row-concatenated bf16 plane `[sum rows, k]` off BF16 tensors, dims
/// `[k, rows]` in the QuantTensor convention the bf16 lanes read.
fn bf16_concat(
    exec: &GpuExecutor,
    map: &MappedGguf,
    parts: &[(&str, usize)],
    k: usize,
) -> Result<QuantTensor, GpuModelError> {
    let rows: usize = parts.iter().map(|(_, r)| r).sum();
    let mut raw: Vec<u8> = Vec::with_capacity(rows * k * 2);
    for (name, r) in parts {
        let (info, bytes) = map
            .tensor_bytes(name)
            .map_err(|e| unsupported(format!("{name}: {e}")))?;
        if info.ggml_type != GgmlType::Bf16 {
            return Err(unsupported(format!(
                "{name}: type {:?}, want BF16",
                info.ggml_type
            )));
        }
        want_dims(map, name, &[k, *r])?;
        if bytes.len() != r * k * 2 {
            return Err(unsupported(format!(
                "{name}: {} bytes for [{r}, {k}] bf16",
                bytes.len()
            )));
        }
        raw.extend_from_slice(bytes);
    }
    Ok(QuantTensor {
        bytes: exec.to_device_u8(&raw).map_err(GpuModelError::from)?,
        ty: GgmlType::Bf16,
        dims: vec![k, rows],
    })
}

/// One routed-expert plane `[in, out, n_expert]` on the repacked k-quant
/// stream: host-mapped when `[moe_offload]` is on, else in VRAM. i-quant
/// types need the pack's `kquant_iq` marker.
fn kq_seat(
    exec: &GpuExecutor,
    map: &MappedGguf,
    name: &str,
    in_dim: usize,
    out_dim: usize,
    n_expert: usize,
) -> Result<KqSeat, GpuModelError> {
    want_dims(map, name, &[in_dim, out_dim, n_expert])?;
    let (info, _) = map
        .tensor_bytes(name)
        .map_err(|e| unsupported(format!("{name}: {e}")))?;
    if !exec.has_kquant_moe() {
        return Err(unsupported(
            "kernel pack has no k-quant MoE lanes - rebuild packs/cuda".into(),
        ));
    }
    if crate::gpu::kq_is_iq(info.ggml_type) && !exec.has_kquant_iq() {
        return Err(unsupported(format!(
            "{name} is {:?} but the kernel pack has no i-quant seats (slot 539) - rebuild packs/cuda",
            info.ggml_type
        )));
    }
    let seat = if crate::gpu::moe_offload().enabled {
        exec.try_repack_kquant_host_mapped(map, name)
            .map_err(|e| unsupported(format!("{name}: {e}")))?
            .map(KqSeat::Host)
    } else {
        exec.try_repack_kquant(map, name)
            .map_err(|e| unsupported(format!("{name}: {e}")))?
            .map(KqSeat::Dev)
    };
    seat.ok_or_else(|| {
        unsupported(format!(
            "{name}: type {:?} has no k-quant expert seat (Q8_0 experts are not served on this lane)",
            info.ggml_type
        ))
    })
}

/// One hyper-connection sub-block off `{pfx}_norm/_down/_up[/_inject]`
/// (`blk.N.hc_attn`, `blk.N.hc_ffn`, `output_hc`). The 8-bit `down` plane
/// cannot fold the f32 inject rows, so they stay a separate plane exactly as
/// the safetensors loader's 8-bit class does.
pub(super) fn hc_weights(
    exec: &GpuExecutor,
    map: &MappedGguf,
    c: &Qwen4ExpConfig,
    pfx: &str,
    inject: bool,
) -> Result<HcW, GpuModelError> {
    let hw = c.hc_width();
    let (lr, hc) = (c.hc_lowrank, c.hc_count);
    Ok(HcW {
        // stored (1+w); the group-norm kernel adds the 1
        norm: f32_dt_m1(exec, map, &format!("{pfx}_norm.weight"), vec![hw])?,
        down: dense(exec, map, &format!("{pfx}_down.weight"), lr, hw)?,
        lowrank: lr,
        inject_rows: 0,
        up: dense(exec, map, &format!("{pfx}_up.weight"), hw, lr)?,
        up_hcmix: None,
        down_p42: None,
        up_p42: None,
        inject: if inject {
            Some(f32_dt(exec, map, &format!("{pfx}_inject.weight"), vec![hc, hw])?)
        } else {
            None
        },
    })
}

pub(super) fn load_layer(
    exec: &Arc<GpuExecutor>,
    map: &MappedGguf,
    c: &Qwen4ExpConfig,
    li: usize,
) -> Result<Qwen4ExpLayer, GpuModelError> {
    let p = format!("blk.{li}");
    let h = c.hidden;
    let mixer = match c.blocks[li] {
        Qwen4ExpBlock::Gdn => {
            let hv = c.gdn_v_heads;
            // a || b as one [2*v_heads, hidden] plane (delta_gate_ab's layout)
            let mut ab_v = f32_vec(map, &format!("{p}.ssm_alpha.weight"), hv * h)?;
            ab_v.extend(f32_vec(map, &format!("{p}.ssm_beta.weight"), hv * h)?);
            // the converter stores -exp(A_log); the gate kernel wants that,
            // the raw plane is kept for the graph's own use
            let ssm_a = f32_vec(map, &format!("{p}.ssm_a"), hv)?;
            if ssm_a.iter().any(|&a| !(a < 0.0)) {
                return Err(unsupported(format!(
                    "{p}.ssm_a: expected -exp(A_log) (all negative), got {ssm_a:?}"
                )));
            }
            let a_log: Vec<f32> = ssm_a.iter().map(|&a| (-a).ln()).collect();
            MixerW::Gdn(GdnW {
                qkv: dense(exec, map, &format!("{p}.attn_qkv.weight"), c.gdn_qkv_rows(), h)?,
                z: dense(exec, map, &format!("{p}.attn_gate.weight"), c.gdn_z_rows(), h)?,
                zqkv: None,
                ab: super::load::dt(exec, ab_v, vec![2 * hv, h])?,
                ab16: None,
                conv: {
                    want_dims(map, &format!("{p}.ssm_conv1d.weight"), &[c.gdn_conv, c.gdn_qkv_rows()])?;
                    f32_dt(
                        exec,
                        map,
                        &format!("{p}.ssm_conv1d.weight"),
                        vec![c.gdn_qkv_rows(), c.gdn_conv],
                    )?
                },
                a_log: super::load::dt(exec, a_log, vec![hv])?,
                ssm_a: super::load::dt(exec, ssm_a, vec![hv])?,
                dt_bias: f32_dt(exec, map, &format!("{p}.ssm_dt.bias"), vec![hv])?,
                // the GDN norm is the one norm the converter leaves raw
                norm: f32_dt(exec, map, &format!("{p}.ssm_norm.weight"), vec![c.gdn_v_dim])?,
                out: dense(exec, map, &format!("{p}.ssm_out.weight"), h, c.gdn_z_rows())?,
                // the converter tiles the value heads (see the module doc)
                tiled_heads: true,
            })
        }
        Qwen4ExpBlock::Attention => {
            let kv = c.n_kv_heads * c.head_dim;
            MixerW::Attn(AttnW {
                q: dense(exec, map, &format!("{p}.attn_q.weight"), c.attn_q_rows(), h)?,
                k: dense(exec, map, &format!("{p}.attn_k.weight"), kv, h)?,
                v: dense(exec, map, &format!("{p}.attn_v.weight"), kv, h)?,
                qkv_f: None,
                o: dense(exec, map, &format!("{p}.attn_output.weight"), h, c.attn_o_in())?,
                // consumed in (1+w) form - the converter's +1 is the fold
                q_norm: f32_dt(exec, map, &format!("{p}.attn_q_norm.weight"), vec![c.head_dim])?,
                k_norm: f32_dt(exec, map, &format!("{p}.attn_k_norm.weight"), vec![c.head_dim])?,
                idx_qk: bf16_concat(
                    exec,
                    map,
                    &[
                        (&format!("{p}.indexer.q_proj.weight"), c.idx_heads * c.idx_head_dim),
                        (&format!("{p}.indexer.k_proj.weight"), c.idx_kv_heads * c.idx_head_dim),
                    ],
                    h,
                )?,
                // stored (1+w); the pack's indexer norm adds the 1
                idx_q_norm: f32_dt_m1(
                    exec,
                    map,
                    &format!("{p}.indexer.q_norm.weight"),
                    vec![c.idx_head_dim],
                )?,
                idx_k_norm: f32_dt_m1(
                    exec,
                    map,
                    &format!("{p}.indexer.k_norm.weight"),
                    vec![c.idx_head_dim],
                )?,
            })
        }
    };
    // router [n_expert, hidden] f32 + the shared expert's scalar gate as row
    // n_expert (see the safetensors loader)
    want_dims(map, &format!("{p}.ffn_gate_inp.weight"), &[h, c.n_expert])?;
    let mut router_v = f32_vec(map, &format!("{p}.ffn_gate_inp.weight"), c.n_expert * h)?;
    router_v.extend(f32_vec(map, &format!("{p}.ffn_gate_inp_shexp.weight"), h)?);
    let seats = ExpertSeats::Kq {
        gate: kq_seat(exec, map, &format!("{p}.ffn_gate_exps.weight"), h, c.moe_ff, c.n_expert)?,
        up: kq_seat(exec, map, &format!("{p}.ffn_up_exps.weight"), h, c.moe_ff, c.n_expert)?,
        down: kq_seat(exec, map, &format!("{p}.ffn_down_exps.weight"), c.moe_ff, h, c.n_expert)?,
        cache: None,
    };
    let moe = MoeW {
        router: super::load::dt(exec, router_v, vec![c.n_expert + 1, h])?,
        router16: None,
        seats,
        sh_gate: dense(exec, map, &format!("{p}.ffn_gate_shexp.weight"), c.shared_ff, h)?,
        sh_gu: None,
        sh_up: dense(exec, map, &format!("{p}.ffn_up_shexp.weight"), c.shared_ff, h)?,
        sh_down: dense(exec, map, &format!("{p}.ffn_down_shexp.weight"), h, c.shared_ff)?,
    };
    Ok(Qwen4ExpLayer {
        attn_hc: hc_weights(exec, map, c, &format!("{p}.hc_attn"), true)?,
        mlp_hc: hc_weights(exec, map, c, &format!("{p}.hc_ffn"), true)?,
        mixer,
        moe,
        ple: None,
    })
}

/// The n-gram table tensor of a GGUF export: `[width, rows]`, one of the
/// 32-wide block types `forward.rs::ple_row_dequant` decodes.
pub(super) const PLE_TABLE: &str = "per_layer_token_embd.weight";

/// The PLE projections + hash constants; the table itself stays in the
/// GGUF mmap (see [`PLE_TABLE`]), `table: None` selects the host gather.
pub(super) fn load_ple(
    exec: &Arc<GpuExecutor>,
    map: &MappedGguf,
    c: &Qwen4ExpConfig,
    li: usize,
    hash: &Qwen4ExpPleHash,
) -> Result<PleW, GpuModelError> {
    let p = format!("blk.{li}");
    let hw = c.hc_width();
    let width = c.ple_embed / c.ple_heads();
    let (info, _) = map
        .tensor_bytes(PLE_TABLE)
        .map_err(|e| unsupported(format!("{PLE_TABLE}: {e}")))?;
    let d: Vec<usize> = info.dims.iter().map(|&x| x as usize).collect();
    if d.len() != 2 || d[0] != width {
        return Err(unsupported(format!(
            "{PLE_TABLE}: dims {d:?}, want [{width}, rows]"
        )));
    }
    if super::forward::ple_row_bytes(info.ggml_type, width).is_none() {
        return Err(unsupported(format!(
            "{PLE_TABLE}: type {:?} has no host row decoder (want IQ4_NL / Q4_0 / Q8_0 / F16 / BF16 / F32)",
            info.ggml_type
        )));
    }
    // the exporter pads the table's row count (to a multiple of 512 in the
    // Unsloth files: 320,001,536 rows for a 320,001,446-row hash space);
    // every hashed row id must land inside the tensor, padding is never read
    let total: i64 = hash.head_vocab_sizes.iter().sum();
    if total > d[1] as i64 {
        return Err(unsupported(format!(
            "{PLE_TABLE}: {} rows but the head vocab sizes sum to {total}",
            d[1]
        )));
    }
    if hash.multipliers.len() != c.ngram_size
        || hash.head_offsets.len() != c.ple_heads()
        || hash.head_vocab_sizes.len() != c.ple_heads()
    {
        return Err(unsupported(format!(
            "ple hash constants: {} multipliers, {} offsets, {} vocab sizes for ngram {} x {} heads",
            hash.multipliers.len(),
            hash.head_offsets.len(),
            hash.head_vocab_sizes.len(),
            c.ngram_size,
            c.ple_heads()
        )));
    }
    Ok(PleW {
        key: dense(exec, map, &format!("{p}.ple_key.weight"), hw, c.hidden)?,
        value: dense(exec, map, &format!("{p}.ple_value.weight"), c.hidden, c.hidden)?,
        conv: {
            want_dims(map, &format!("{p}.ple_conv1d.weight"), &[c.ple_conv, hw])?;
            f32_dt(exec, map, &format!("{p}.ple_conv1d.weight"), vec![hw, c.ple_conv])?
        },
        // all three stored (1+w); the pack's PLE norms add the 1
        norm_key: f32_dt_m1(exec, map, &format!("{p}.ple_norm_key.weight"), vec![hw])?,
        norm_query: f32_dt_m1(exec, map, &format!("{p}.ple_norm_query.weight"), vec![hw])?,
        norm_conv: f32_dt_m1(exec, map, &format!("{p}.ple_norm_conv.weight"), vec![hw])?,
        table: None,
        table_rows: d[1],
        // the quantized rows carry their own scales
        table_scale: 1.0,
        multipliers: hash.multipliers.clone(),
        head_vocab: hash.head_vocab_sizes.clone(),
        head_offset: hash.head_offsets.clone(),
    })
}

/// `token_embd` on the k-quant gather (Q4_K in the UD exports).
pub(super) fn load_embed(
    exec: &GpuExecutor,
    map: &MappedGguf,
    c: &Qwen4ExpConfig,
) -> Result<Embed, GpuModelError> {
    let name = "token_embd.weight";
    want_dims(map, name, &[c.hidden, c.vocab])?;
    let (info, bytes) = map
        .tensor_bytes(name)
        .map_err(|e| unsupported(format!("{name}: {e}")))?;
    if info.ggml_type == GgmlType::Bf16 {
        return Ok(Embed::Bf16(QuantTensor {
            bytes: exec.to_device_u8(bytes).map_err(GpuModelError::from)?,
            ty: GgmlType::Bf16,
            dims: vec![c.hidden, c.vocab],
        }));
    }
    let w: RepackedKQ = exec
        .try_repack_kquant(map, name)
        .map_err(|e| unsupported(format!("{name}: {e}")))?
        .ok_or_else(|| {
            unsupported(format!(
                "{name}: type {:?} has no k-quant gather",
                info.ggml_type
            ))
        })?;
    Ok(Embed::Kq(w))
}

pub(super) fn load_head(
    exec: &GpuExecutor,
    map: &MappedGguf,
    c: &Qwen4ExpConfig,
) -> Result<DensePlane, GpuModelError> {
    dense(exec, map, "output.weight", c.vocab, c.hidden)
}
