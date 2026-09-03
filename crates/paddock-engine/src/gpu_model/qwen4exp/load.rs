//! Checkpoint-dir loader - every residency byte-exact to the file (charter:
//! bf16 parity first). Big GEMM planes keep their bf16 bytes on device;
//! small vectors widen to f32 (exact - bf16 is the top half of f32); routed
//! experts ride `nvf4_moe_upload` with the checkpoint's own nibbles.

use std::sync::Arc;

use crate::gpu::{DeviceTensor, GpuError, GpuExecutor, Nvf4MoePlane, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::{bf16_bytes, bf16_to_f32, f32_tensor};
use paddock_models::ggml_type::GgmlType;
use paddock_models::modelopt::nvfp4_view;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::*;
use super::{DenseClass, DensePlane, dense_class_from_env};

fn dt(exec: &GpuExecutor, v: Vec<f32>, dims: Vec<usize>) -> Result<DeviceTensor, GpuError> {
    Ok(DeviceTensor {
        buf: exec.to_device(&v)?,
        dims,
    })
}

/// bf16 plane resident as shipped: checkpoint bytes on device, dims [k, n]
/// (in_dim-major, the QuantTensor convention the bf16 GEMV/GEMM lanes read).
fn bf16_plane(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    n: usize,
    k: usize,
) -> Result<QuantTensor, GpuModelError> {
    let raw = bf16_bytes(st, name, n * k)?;
    Ok(QuantTensor {
        bytes: exec.to_device_u8(raw).map_err(GpuModelError::from)?,
        ty: GgmlType::Bf16,
        dims: vec![k, n],
    })
}

/// f32 exact widen of a small tensor.
fn f32_dt(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    dims: Vec<usize>,
) -> Result<DeviceTensor, GpuModelError> {
    let n: usize = dims.iter().product();
    dt(exec, f32_tensor(st, name, n)?, dims).map_err(GpuModelError::from)
}

/// Gemma (1+w) fold for the norms the pack's PLAIN-w kernels consume
/// (attention q/k per-head RMSNorm). The checkpoint stores raw `w`, and
/// `rmsnorm_batch` computes `x*inv*w` - so `w+1` on the host is the reference
/// (`examples/q38fn_host_forward.rs` folds the same way before calling the
/// plain-w reference). The hyper-connection and PLE norms do not come through
/// here: their kernel (`q4x_group_norm_1p`) applies the (1+w) FMA itself.
fn f32_dt_1p(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    dims: Vec<usize>,
) -> Result<DeviceTensor, GpuModelError> {
    let n: usize = dims.iter().product();
    let v: Vec<f32> = f32_tensor(st, name, n)?
        .into_iter()
        .map(|w| w + 1.0)
        .collect();
    dt(exec, v, dims).map_err(GpuModelError::from)
}

/// conv1d [rows, 1, k] -> device f32 [rows, k] (squeeze the middle axis).
fn conv_plane(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    rows: usize,
    k: usize,
) -> Result<DeviceTensor, GpuModelError> {
    f32_dt(exec, st, name, vec![rows, k]).map_err(|e| match e {
        GpuModelError::Unsupported(m) => GpuModelError::Unsupported(format!("{m} (conv [r,1,k])")),
        other => other,
    })
}

/// Load one dense projection in the elected class. bf16 is the parity class
/// and the default; the 8-bit lane is opt-in (see `DenseClass`).
fn dense(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    n: usize,
    k: usize,
) -> Result<DensePlane, GpuModelError> {
    match dense_class_from_env() {
        DenseClass::Bf16 => Ok(DensePlane::Bf16(bf16_plane(exec, st, name, n, k)?)),
        DenseClass::F8Row => {
            if !exec.has_bf16_to_f8row() {
                return Err(GpuModelError::Unsupported(
                    "PADDOCK_Q38FN_DENSE=f8row but the pack has no bf16_to_f8row".into(),
                ));
            }
            // Straight from the checkpoint's own bf16 - no Q8 hop, which would
            // double-quantize (the edge `bf16_to_f8row` exists to close).
            let raw = bf16_bytes(st, name, n * k)?;
            let plane = exec.bf16_to_f8row(raw, k, n).map_err(GpuModelError::from)?;
            Ok(DensePlane::F8Row {
                plane,
                in_dim: k,
                out_dim: n,
            })
        }
    }
}

/// The lm_head rides the same class election as every other dense plane -
/// it is 1.27 GB of the per-token traffic on its own.
pub(crate) fn dense_head(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    name: &str,
    n: usize,
    k: usize,
) -> Result<DensePlane, GpuModelError> {
    dense(exec, st, name, n, k)
}

pub(crate) fn hc_weights(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    pfx: &str,
    inject: bool,
) -> Result<HcW, GpuModelError> {
    let hw = c.hc_width();
    let (lr, hc) = (c.hc_lowrank, c.hc_count);
    let down_name = format!("{pfx}.input_mix_weight_down.weight");
    let inj_name = format!("{pfx}.block_inject_weight.weight");
    // Fold the inject rows onto the down plane when we may: both projections
    // read the same normalized state, so this turns two launches into one at
    // batch 1 (and two ROW-SEGMENT reads of one residency above it). Refused
    // when the dense class is 8-bit - appending f32 inject weights to a
    // quantized plane would quantize them, a numerics change a launch fold is
    // not entitled to make - and when the checkpoint does not store the
    // inject as bf16, since the concatenation is a BYTE concatenation.
    let inj_is_bf16 = st
        .bytes(&inj_name)
        .map(|(t, _)| t.dtype == StDtype::Bf16)
        .unwrap_or(false);
    let fold = inject && dense_class_from_env() == DenseClass::Bf16 && inj_is_bf16;
    let norm = f32_dt(exec, st, &format!("{pfx}.hc_norm.weight"), vec![hw])?;
    let up = dense(
        exec,
        st,
        &format!("{pfx}.input_mix_weight_up.weight"),
        hw,
        lr,
    )?;
    if fold {
        let mut raw = bf16_bytes(st, &down_name, lr * hw)?.to_vec();
        raw.extend_from_slice(bf16_bytes(st, &inj_name, hc * hw)?);
        let plane = QuantTensor {
            bytes: exec.to_device_u8(&raw).map_err(GpuModelError::from)?,
            ty: GgmlType::Bf16,
            dims: vec![hw, lr + hc],
        };
        return Ok(HcW {
            norm,
            down: DensePlane::Bf16(plane),
            lowrank: lr,
            inject_rows: hc,
            up,
            inject: None,
        });
    }
    Ok(HcW {
        norm,
        down: dense(exec, st, &down_name, lr, hw)?,
        lowrank: lr,
        inject_rows: 0,
        up,
        inject: if inject {
            Some(f32_dt(exec, st, &inj_name, vec![hc, hw])?)
        } else {
            None
        },
    })
}

/// One routed-expert plane across all 512 experts, nibbles as shipped.
fn moe_plane(
    exec: &GpuExecutor,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    pfx: &str,
    role: &str,
    rows: usize,
    in_dim: usize,
) -> Result<Nvf4MoePlane, GpuModelError> {
    let mut cat_p: Vec<u8> = Vec::with_capacity(c.n_expert * rows * in_dim / 2);
    let mut cat_s: Vec<u8> = Vec::with_capacity(c.n_expert * rows * in_dim / 16);
    let mut s2 = Vec::with_capacity(c.n_expert);
    for e in 0..c.n_expert {
        let v = nvfp4_view(st, &format!("{pfx}.mlp.experts.{e}.{role}"))
            .map_err(|err| GpuModelError::Unsupported(format!("{pfx} expert {e} {role}: {err}")))?;
        if (v.n, v.k) != (rows, in_dim) {
            return Err(GpuModelError::Unsupported(format!(
                "{pfx} expert {e} {role} is [{}, {}], expected [{rows}, {in_dim}]",
                v.n, v.k
            )));
        }
        cat_p.extend_from_slice(v.packed);
        cat_s.extend_from_slice(v.scales);
        s2.push(v.scale2);
    }
    exec.nvf4_moe_upload(&cat_p, &cat_s, &s2, c.n_expert, rows, in_dim)
        .map_err(GpuModelError::from)
}

/// Load one decoder layer (everything except the PLE table - see `load_ple`).
pub fn load_layer(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    li: usize,
) -> Result<Qwen4ExpLayer, GpuModelError> {
    let p = format!("model.language_model.layers.{li}");
    let h = c.hidden;
    let mixer = match c.blocks[li] {
        Qwen4ExpBlock::Gdn => {
            let g = format!("{p}.linear_attn");
            MixerW::Gdn(GdnW {
                qkv: dense(
                    exec,
                    st,
                    &format!("{g}.in_proj_qkv.weight"),
                    c.gdn_qkv_rows(),
                    h,
                )?,
                z: dense(
                    exec,
                    st,
                    &format!("{g}.in_proj_z.weight"),
                    c.gdn_z_rows(),
                    h,
                )?,
                // a then b, which is delta_gate_ab's fused layout
                ab: {
                    let mut v =
                        f32_tensor(st, &format!("{g}.in_proj_a.weight"), c.gdn_v_heads * h)?;
                    v.extend(f32_tensor(
                        st,
                        &format!("{g}.in_proj_b.weight"),
                        c.gdn_v_heads * h,
                    )?);
                    dt(exec, v, vec![2 * c.gdn_v_heads, h])?
                },
                conv: conv_plane(
                    exec,
                    st,
                    &format!("{g}.conv1d.weight"),
                    c.gdn_qkv_rows(),
                    c.gdn_conv,
                )?,
                a_log: f32_dt(exec, st, &format!("{g}.A_log"), vec![c.gdn_v_heads])?,
                // pd_delta_gate computes g = ssm_a * softplus(a + dt_bias), so
                // it wants the -exp(A_log) fold the reference does inline. Kept
                // BESIDE the raw A_log (which stays byte-exact residency) -
                // 48 floats, and a derived plane is cheaper than a kernel arm.
                ssm_a: dt(
                    exec,
                    f32_tensor(st, &format!("{g}.A_log"), c.gdn_v_heads)?
                        .into_iter()
                        .map(|a| -a.exp())
                        .collect(),
                    vec![c.gdn_v_heads],
                )?,
                dt_bias: f32_dt(exec, st, &format!("{g}.dt_bias"), vec![c.gdn_v_heads])?,
                norm: f32_dt(exec, st, &format!("{g}.norm.weight"), vec![c.gdn_v_dim])?,
                out: dense(exec, st, &format!("{g}.out_proj.weight"), h, c.gdn_z_rows())?,
            })
        }
        Qwen4ExpBlock::Attention => {
            let a = format!("{p}.self_attn");
            let kv = c.n_kv_heads * c.head_dim;
            MixerW::Attn(AttnW {
                q: dense(exec, st, &format!("{a}.q_proj.weight"), c.attn_q_rows(), h)?,
                k: dense(exec, st, &format!("{a}.k_proj.weight"), kv, h)?,
                v: dense(exec, st, &format!("{a}.v_proj.weight"), kv, h)?,
                o: dense(exec, st, &format!("{a}.o_proj.weight"), h, c.attn_o_in())?,
                q_norm: f32_dt_1p(exec, st, &format!("{a}.q_norm.weight"), vec![c.head_dim])?,
                k_norm: f32_dt_1p(exec, st, &format!("{a}.k_norm.weight"), vec![c.head_dim])?,
                idx_qk: bf16_plane(
                    exec,
                    st,
                    &format!("{a}.indexer.index_qk_proj.weight"),
                    (c.idx_heads + c.idx_kv_heads) * c.idx_head_dim,
                    h,
                )?,
                idx_q_norm: f32_dt(
                    exec,
                    st,
                    &format!("{a}.indexer.q_layernorm.weight"),
                    vec![c.idx_head_dim],
                )?,
                idx_k_norm: f32_dt(
                    exec,
                    st,
                    &format!("{a}.indexer.k_layernorm.weight"),
                    vec![c.idx_head_dim],
                )?,
            })
        }
    };
    let moe = MoeW {
        // the shared expert's scalar gate rides as row n_expert
        router: {
            let mut v = f32_tensor(st, &format!("{p}.mlp.gate.weight"), c.n_expert * h)?;
            v.extend(f32_tensor(
                st,
                &format!("{p}.mlp.shared_expert_gate.weight"),
                h,
            )?);
            dt(exec, v, vec![c.n_expert + 1, h])?
        },
        gate: moe_plane(exec, st, c, &p, "gate_proj", c.moe_ff, h)?,
        up: moe_plane(exec, st, c, &p, "up_proj", c.moe_ff, h)?,
        down: moe_plane(exec, st, c, &p, "down_proj", h, c.moe_ff)?,
        sh_gate: dense(
            exec,
            st,
            &format!("{p}.mlp.shared_expert.gate_proj.weight"),
            c.shared_ff,
            h,
        )?,
        sh_up: dense(
            exec,
            st,
            &format!("{p}.mlp.shared_expert.up_proj.weight"),
            c.shared_ff,
            h,
        )?,
        sh_down: dense(
            exec,
            st,
            &format!("{p}.mlp.shared_expert.down_proj.weight"),
            h,
            c.shared_ff,
        )?,
    };
    Ok(Qwen4ExpLayer {
        attn_hc: hc_weights(exec, st, c, &format!("{p}.attn_hyper_connection"), true)?,
        mlp_hc: hc_weights(exec, st, c, &format!("{p}.mlp_hyper_connection"), true)?,
        mixer,
        moe,
        ple: None, // filled by load_ple on PLE layers (51 GB table)
    })
}

/// PLE projections + hash buffers (no table - see `load_ple_table`).
pub fn load_ple_projections(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    li: usize,
) -> Result<PleW, GpuModelError> {
    let p = format!("model.language_model.layers.{li}.ple");
    let hw = c.hc_width();
    let emb = format!("{p}.ple_embedding");
    let i64s = |name: &str, want: usize| -> Result<Vec<i64>, GpuModelError> {
        let (t, b) = st
            .bytes(name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        if t.dtype != StDtype::I64 || b.len() != want * 8 {
            return Err(GpuModelError::Unsupported(format!(
                "{name}: want I64 x{want}"
            )));
        }
        Ok(b.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect())
    };
    // table scale: bf16 scalar in this repo (F32 in the FP8 repo) - widen
    let table_scale = {
        let name = format!("{emb}.ngram_embedding.weight_scale");
        let (t, b) = st
            .bytes(&name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        match t.dtype {
            StDtype::Bf16 => bf16_to_f32(b)[0],
            StDtype::F32 => f32::from_le_bytes(b[..4].try_into().unwrap()),
            other => {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: dtype {other:?}"
                )));
            }
        }
    };
    Ok(PleW {
        key: dense(exec, st, &format!("{p}.key_proj.weight"), hw, c.hidden)?,
        value: dense(
            exec,
            st,
            &format!("{p}.value_proj.weight"),
            c.hidden,
            c.hidden,
        )?,
        conv: conv_plane(exec, st, &format!("{p}.conv1d.weight"), hw, c.ple_conv)?,
        norm_key: f32_dt(exec, st, &format!("{p}.norm_key.weight"), vec![hw])?,
        norm_query: f32_dt(exec, st, &format!("{p}.norm_query.weight"), vec![hw])?,
        norm_conv: f32_dt(exec, st, &format!("{p}.norm_conv.weight"), vec![hw])?,
        table: None,
        table_rows: 0,
        table_scale,
        multipliers: i64s(&format!("{emb}.layer_multipliers"), c.ngram_size)?,
        head_vocab: i64s(&format!("{emb}.ngram_heads_vocab_sizes"), c.ple_heads())?,
        head_offset: i64s(&format!("{emb}.ngram_heads_offsets"), c.ple_heads())?,
    })
}

/// Upload the 51.2 GB n-gram table (128 fp8 shards, concatenated in order).
pub fn load_ple_table(
    exec: &Arc<GpuExecutor>,
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    li: usize,
    ple: &mut PleW,
) -> Result<(), GpuModelError> {
    let emb = format!("model.language_model.layers.{li}.ple.ple_embedding");
    let width = c.ple_embed / c.ple_heads();
    let mut rows = 0usize;
    let mut host: Vec<u8> = Vec::new();
    for sh in 0..c.ngram_split {
        let name = format!("{emb}.ngram_embedding.shard_{sh}.weight");
        let (t, b) = st
            .bytes(&name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        if t.dtype != StDtype::F8E4m3 || t.shape.len() != 2 || t.shape[1] != width {
            return Err(GpuModelError::Unsupported(format!(
                "{name}: want F8 [*, {width}], got {:?} {:?}",
                t.dtype, t.shape
            )));
        }
        rows += t.shape[0];
        host.extend_from_slice(b);
    }
    ple.table = Some(exec.to_device_u8(&host).map_err(GpuModelError::from)?);
    ple.table_rows = rows;
    Ok(())
}
