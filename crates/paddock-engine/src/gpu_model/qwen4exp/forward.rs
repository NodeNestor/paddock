//! Qwen3.8-Flash-Next forward graph -  stage 3.
//!
//! Prompt-at-once (every token of the prompt in one pass per layer), which is
//! the shape the parity gate needs and the shape a chunked-prefill serving
//! lane grows from. What it is not yet: incremental decode off carried state,
//! CUDA-graph capture, continuous batching, or the QSA sparse walk - those are
//! the perf/serving rungs, each with its own gate.
//!
//! The gate this file answers to is `examples/q38fn_host_forward.rs`, the
//! host-exact forward that holds ARBITER parity (docs/qwen38-flash-next-
//! bringup.md stage 2). Every op here is either an existing pack lane
//! whose semantics were source-verified for this family, or one of the new
//! `q4x_*` slots - no formula is re-derived in this file.
//!
//! PLE residency: the 51 GB n-gram table stays on the host and rows are
//! gathered per token (10 KB/token) - the same split vLLM's PLE-offload lane
//! makes. `load_ple_table` (device-resident) exists for the day the memory
//! budget prefers it.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;

use crate::gpu::{DeviceTensor, GpuExecutor, KvDtype, QuantTensor};
use crate::gpu_model::gpt_oss::GpuModelError;
use crate::gpu_model::st_load::bf16_bytes;
use paddock_kernels::reference::qwen4exp as rq;
use paddock_models::ggml_type::GgmlType;
use paddock_models::qwen4exp::{Qwen4ExpBlock, Qwen4ExpConfig};
use paddock_models::safetensors::{ShardedSafetensors, StDtype};

use super::load::{dense_head, hc_weights, load_layer, load_ple_projections};
use super::{DensePlane, DenseStage, HcW, MixerW, PleW, Qwen4ExpLayer};

/// Attention KV element type. f16 is the narrowest class the pack's attention
/// lanes take (there is no f32 KV kernel), and BF16 - the checkpoint's own
/// storage class - carries three FEWER mantissa bits than f16, so this is not
/// a precision concession. It is still the dominant deviation from the f32
/// host reference, and the full-forward gate is stated in those terms.
const KV: KvDtype = KvDtype::Fp16;

/// PLE conv dilation - a k=4 kernel over a 9-token receptive ring.
const PLE_DILATION: usize = 3;

/// Per-op dump sink, armed by `PADDOCK_Q38FN_DUMP=<dir>`. Writes the same tag
/// names `examples/q38fn_host_forward.rs --dump` writes, so the two trees diff
/// directly and a deviation localizes to one op of one layer instead of to
/// "the logits". Readback is capture-illegal, so this is a triage path only -
/// nothing reads the env var on the serving walk.
struct Dump(Option<std::path::PathBuf>);

impl Dump {
    fn arm() -> Self {
        Self(std::env::var_os("PADDOCK_Q38FN_DUMP").map(|d| {
            let p = std::path::PathBuf::from(d);
            let _ = std::fs::create_dir_all(&p);
            p
        }))
    }
    fn on(&self) -> bool {
        self.0.is_some()
    }
    /// Write host-side values already read back.
    fn put_host(&self, li: usize, tag: &str, v: &[f32]) -> Result<(), GpuModelError> {
        let Some(dir) = &self.0 else { return Ok(()) };
        let mut b = Vec::with_capacity(v.len() * 4);
        for x in v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(dir.join(format!("L{li}.{tag}.bin")), b)
            .map_err(|err| GpuModelError::Unsupported(format!("dump write: {err}")))?;
        Ok(())
    }

    /// Read `len` elements back and write them as raw little-endian f32.
    fn put(
        &self,
        e: &GpuExecutor,
        li: usize,
        tag: &str,
        buf: &CudaSlice<f32>,
        len: usize,
    ) -> Result<(), GpuModelError> {
        let Some(dir) = &self.0 else { return Ok(()) };
        let v = e.to_host_len(buf, len)?;
        let mut b = Vec::with_capacity(len * 4);
        for x in &v {
            b.extend_from_slice(&x.to_le_bytes());
        }
        let name = if li == usize::MAX {
            format!("{tag}.bin")
        } else {
            format!("L{li}.{tag}.bin")
        };
        std::fs::write(dir.join(name), b)
            .map_err(|err| GpuModelError::Unsupported(format!("dump write: {err}")))?;
        Ok(())
    }
}

/// M-RoPE section split for this family (text uses all four axes equal, so the
/// split only matters for the vision rung; it is the checkpoint's own
/// `[11,11,10]` plus the zero extra axis).
const MROPE_SECTIONS: [u32; 4] = [11, 11, 10, 0];

pub struct Qwen4ExpGpu {
    exec: Arc<GpuExecutor>,
    cfg: Qwen4ExpConfig,
    /// kept open for the host-side PLE n-gram gather
    st: ShardedSafetensors,
    layers: Vec<Qwen4ExpLayer>,
    embed: QuantTensor,
    lm_head: DensePlane,
    final_mix: HcW,
    max_tokens: usize,
    sc: Scratch,
    /// per-layer GDN recurrent state `[v_heads][k_dim][v_dim]`, None on attn layers
    recur: Vec<Option<CudaSlice<f32>>>,
    /// per-layer KV caches `[max_tokens, kv_dim]`, None on GDN layers
    kv_k: Vec<Option<CudaSlice<u8>>>,
    kv_v: Vec<Option<CudaSlice<u8>>>,
    /// per-GDN-layer conv window: the last `k-1` PRE-conv rows, oldest first
    /// (`conv_step`'s contract - it shifts the window itself)
    gdn_win: Vec<Option<CudaSlice<f32>>>,
    /// the PLE conv's window: the last `(k-1)*dilation` pre-conv rows of
    /// `norm_conv(gv)`, oldest first. `q4x_conv_dil_step` is stateless by
    /// design (graph-safe), so this side advances it.
    ple_win: Option<CudaSlice<f32>>,
    /// next position to write - the cursor prefill leaves behind and decode
    /// advances. 0 means "no sequence started".
    pos: usize,
    /// the request's token stream with the 2-token EOS priming already on the
    /// front, carried so a decode step can hash its n-gram window.
    stream: Vec<i64>,
    /// The captured decode tick. Every per-token INPUT (token id, positions,
    /// the PLE n-gram rows) is staged into address-stable buffers before the
    /// replay, and every kernel in the tick reads its position from the device
    /// - so one capture is valid at every position, exactly as in the qwen3.5
    ///   lane. `None` until the first decode step builds it, or forever under
    ///   `PADDOCK_Q38FN_NO_GRAPH`.
    decode_graph: Option<crate::gpu::CapturedGraph>,
    /// staging the 8-bit classes need on their batch > 1 arm
    stage: DenseStage,
    /// Whether decode ticks may be captured at all. Defaults to the env gate;
    /// `set_graph_capture` lets one process A/B the two paths, which is how
    /// the capture gate proves the graph and the eager walk agree.
    graph_capture: bool,
}

/// Every device buffer the walk touches, allocated once at `max_tokens`.
/// Address-stable by construction - the graph-capture rung depends on it.
struct Scratch {
    d_tok: CudaSlice<u32>,
    d_pos: CudaSlice<u32>,
    d_mrope: CudaSlice<u32>,
    d_slots: CudaSlice<u32>,
    d_x: CudaSlice<f32>,
    d_h: CudaSlice<f32>,
    d_xn: CudaSlice<f32>,
    d_m: CudaSlice<f32>,
    d_gate: CudaSlice<f32>,
    d_bi: CudaSlice<f32>,
    d_inj: CudaSlice<f32>,
    d_mix: CudaSlice<f32>,
    // GDN
    d_qkv: CudaSlice<f32>,
    d_zg: CudaSlice<f32>,
    d_ab: CudaSlice<f32>,
    d_g: CudaSlice<f32>,
    d_beta: CudaSlice<f32>,
    d_conv: CudaSlice<f32>,
    d_dq: CudaSlice<f32>,
    d_dk: CudaSlice<f32>,
    d_dv: CudaSlice<f32>,
    d_dattn: CudaSlice<f32>,
    d_core: CudaSlice<f32>,
    // attention
    d_qg: CudaSlice<f32>,
    d_q: CudaSlice<f32>,
    d_agate: CudaSlice<f32>,
    d_k: CudaSlice<f32>,
    d_v: CudaSlice<f32>,
    d_qn: CudaSlice<f32>,
    d_kn: CudaSlice<f32>,
    d_attn: CudaSlice<f32>,
    d_sinks: CudaSlice<f32>,
    // MoE
    d_logits: CudaSlice<f32>,
    d_zero_bias: CudaSlice<f32>,
    d_idx: CudaSlice<u32>,
    d_topw: CudaSlice<f32>,
    d_act: CudaSlice<f32>,
    d_shg: CudaSlice<f32>,
    d_shu: CudaSlice<f32>,
    d_shd: CudaSlice<f32>,
    d_shgate: CudaSlice<f32>,
    // PLE
    d_emb: CudaSlice<f32>,
    d_pkey: CudaSlice<f32>,
    d_pval: CudaSlice<f32>,
    d_pkn: CudaSlice<f32>,
    d_pqn: CudaSlice<f32>,
    d_pgv: CudaSlice<f32>,
    d_pconv: CudaSlice<f32>,
    /// scratch for the PLE window shift (an in-buffer copy would overlap)
    d_pwin_tmp: CudaSlice<f32>,
    // head
    d_fin: CudaSlice<f32>,
    d_out: CudaSlice<f32>,
}

impl Qwen4ExpGpu {
    /// Load the whole text model. `max_tokens` sizes every scratch plane and
    /// the KV caches - a longer prompt is a loud refusal, never a silent
    /// truncation.
    pub fn load(
        exec: &Arc<GpuExecutor>,
        dir: &std::path::Path,
        max_tokens: usize,
    ) -> Result<Self, GpuModelError> {
        if !exec.has_delta_gate_ab() {
            return Err(GpuModelError::Unsupported(
                "pack has no delta_gate_ab - the folded GDN a||b plane needs it".into(),
            ));
        }
        if !exec.has_qwen4exp_ops() {
            return Err(GpuModelError::Unsupported(
                "kernel pack has no qwen4exp family (slots 506-516) - rebuild packs/cuda".into(),
            ));
        }
        let cfg = Qwen4ExpConfig::read(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("qwen4exp config: {e}")))?;
        let st = ShardedSafetensors::open_dir(dir)
            .map_err(|e| GpuModelError::Unsupported(format!("qwen4exp shards: {e}")))?;

        let h = cfg.hidden;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            let mut layer = load_layer(exec, &st, &cfg, li)?;
            if cfg.ple_layers.contains(&li) {
                layer.ple = Some(load_ple_projections(exec, &st, &cfg, li)?);
            }
            layers.push(layer);
        }

        let embed = bf16_plane(
            exec,
            &st,
            "model.language_model.embed_tokens.weight",
            cfg.vocab,
            h,
        )?;
        let lm_head = dense_head(exec, &st, "lm_head.weight", cfg.vocab, h)?;
        let final_mix = hc_weights(
            exec,
            &st,
            &cfg,
            "model.language_model.hyper_connection_mixer",
            false,
        )?;

        let (recur, kv_k, kv_v) = alloc_state(exec, &cfg, max_tokens)?;
        let mut gdn_win = Vec::with_capacity(cfg.n_layer);
        for li in 0..cfg.n_layer {
            gdn_win.push(match cfg.blocks[li] {
                Qwen4ExpBlock::Gdn => Some(exec.alloc((cfg.gdn_conv - 1) * cfg.gdn_qkv_rows())?),
                Qwen4ExpBlock::Attention => None,
            });
        }
        let ple_win = if cfg.ple_layers.is_empty() {
            None
        } else {
            Some(exec.alloc((cfg.ple_conv - 1) * PLE_DILATION * cfg.hc_width())?)
        };
        let sc = Scratch::new(exec, &cfg, max_tokens)?;
        // the widest activation any dense plane reads is the 4-stream state
        let stage = DenseStage {
            q: exec.alloc_i8(max_tokens * cfg.hc_width())?,
            rs: exec.alloc(max_tokens)?,
        };
        Ok(Self {
            exec: exec.clone(),
            cfg,
            st,
            layers,
            embed,
            lm_head,
            final_mix,
            max_tokens,
            sc,
            recur,
            kv_k,
            kv_v,
            gdn_win,
            ple_win,
            pos: 0,
            stream: Vec::new(),
            stage,
            decode_graph: None,
            graph_capture: capture_wanted(),
        })
    }

    pub fn config(&self) -> &Qwen4ExpConfig {
        &self.cfg
    }

    /// Prefill a fresh prompt: resets every carried state, runs all prompt
    /// tokens, and leaves the cursor, conv windows, GDN recurrence and KV
    /// ready for [`Self::decode_step`]. Returns the FINAL position's logits.
    pub fn forward_prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GpuModelError> {
        let n = ids.len();
        if n == 0 || n > self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "prompt of {n} tokens; this lane is sized for 1..={}",
                self.max_tokens
            )));
        }
        self.reset()?;
        // the PLE hash reads a 2-token EOS-primed stream (vLLM `ngram_context`)
        self.stream = vec![self.cfg.bos_id as i64; 2];
        self.stream.extend(ids.iter().map(|&i| i as i64));
        let logits = self.walk(ids, Phase::Prefill)?;
        self.pos = n;
        Ok(logits)
    }

    /// Continue the live sequence by one token off the carried state - the
    /// GDN recurrence, both conv windows, the KV cache and the position
    /// cursor all pick up where the prefill (or the previous step) left them.
    pub fn decode_step(&mut self, id: u32) -> Result<Vec<f32>, GpuModelError> {
        if self.pos == 0 {
            return Err(GpuModelError::Unsupported(
                "decode_step before any prompt - call forward_prompt first".into(),
            ));
        }
        if self.pos >= self.max_tokens {
            return Err(GpuModelError::Unsupported(format!(
                "sequence reached {} tokens, the size this lane was built for",
                self.max_tokens
            )));
        }
        self.stream.push(id as i64);
        self.stage_inputs(&[id])?;
        if self.decode_graph.is_none() && self.graph_capture {
            self.capture_decode_tick()?;
        }
        match self.decode_graph.as_ref() {
            Some(g) => g
                .launch()
                .map_err(|e| crate::gpu::GpuError::Driver(format!("decode graph replay: {e}")))?,
            None => self.device_walk(1, Phase::Decode)?,
        }
        let logits = self.exec.to_host_len(&self.sc.d_out, self.cfg.vocab)?;
        self.pos += 1;
        Ok(logits)
    }

    /// Record the decode tick as a CUDA graph. Capture RECORDS without
    /// executing, so the caller still has to replay once for the step to
    /// happen. Valid at every position: each per-token input was staged into
    /// an address-stable buffer beforehand, and every kernel reads its
    /// position from the device.
    fn capture_decode_tick(&mut self) -> Result<(), GpuModelError> {
        self.exec
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| crate::gpu::GpuError::Driver(format!("begin_capture: {e}")))?;
        let walked = self.device_walk(1, Phase::Decode);
        // end the capture even if the walk failed, or the stream stays in
        // capture mode and every later launch fails with a confusing error
        let graph = crate::gpu::end_capture_no_flags(&self.exec.stream)
            .map_err(|e| crate::gpu::GpuError::Driver(format!("end_capture: {e}")));
        walked?;
        self.decode_graph = graph?;
        Ok(())
    }

    /// Greedy continuation: prefill `prompt`, then take up to `n_new` steps,
    /// stopping at any configured EOS. Returns the generated ids.
    pub fn generate_greedy(
        &mut self,
        prompt: &[u32],
        n_new: usize,
    ) -> Result<Vec<u32>, GpuModelError> {
        let mut logits = self.forward_prompt(prompt)?;
        let mut out = Vec::with_capacity(n_new);
        for _ in 0..n_new {
            let id = argmax(&logits) as u32;
            out.push(id);
            if self.cfg.eos_ids.contains(&id) {
                break;
            }
            logits = self.decode_step(id)?;
        }
        Ok(out)
    }

    /// Turn decode-tick graph capture on or off. Dropping an existing capture
    /// is safe at any time - it is a pure accelerator over the eager walk, and
    /// the two are gated against each other in
    /// `tests/gpu_qwen4exp_forward.rs`.
    pub fn set_graph_capture(&mut self, on: bool) {
        self.graph_capture = on;
        if !on {
            self.decode_graph = None;
        }
    }

    /// Whether the decode tick is currently running as a captured graph.
    pub fn graph_active(&self) -> bool {
        self.decode_graph.is_some()
    }

    /// The dense weight class this model actually loaded - for benchmarks and
    /// the PPL gate, so the class is never implicit in a number.
    pub fn dense_class(&self) -> &'static str {
        self.layers
            .first()
            .map(|l| l.attn_hc.down.class())
            .unwrap_or("none")
    }

    /// How many tokens the live sequence holds (0 = nothing started).
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Drop every per-sequence state. The allocations stay - a later capture
    /// rung bakes these addresses.
    fn reset(&mut self) -> Result<(), GpuModelError> {
        let state_len = self.cfg.gdn_v_heads * self.cfg.gdn_k_dim * self.cfg.gdn_v_dim;
        for r in self.recur.iter_mut().flatten() {
            self.exec.zero_region(r, 0, state_len)?;
        }
        let win_len = (self.cfg.gdn_conv - 1) * self.cfg.gdn_qkv_rows();
        for w in self.gdn_win.iter_mut().flatten() {
            self.exec.zero_region(w, 0, win_len)?;
        }
        if let Some(w) = self.ple_win.as_mut() {
            let n = (self.cfg.ple_conv - 1) * PLE_DILATION * self.cfg.hc_width();
            self.exec.zero_region(w, 0, n)?;
        }
        self.pos = 0;
        self.stream.clear();
        // the captured tick survives: it names buffers, not values, and every
        // one of them is address-stable for the life of this model
        Ok(())
    }

    /// The layer walk, shared by both phases. `ids` are the tokens to run and
    /// they start at the current cursor; the KV cache and both conv windows
    /// are read AND advanced, so a prefill of n then k decode steps is the
    /// same arithmetic as a prefill of n+k (that equality is the gate in
    /// `tests/gpu_qwen4exp_forward.rs`).
    fn walk(&mut self, ids: &[u32], phase: Phase) -> Result<Vec<f32>, GpuModelError> {
        self.stage_inputs(ids)?;
        self.device_walk(ids.len(), phase)?;
        Ok(self.exec.to_host_len(&self.sc.d_out, self.cfg.vocab)?)
    }

    /// Everything a tick reads from the host: the token ids, the position and
    /// mrope planes, the zero slot map, and the PLE n-gram rows (a pure
    /// function of the token stream, so it is known before the tick runs).
    /// Kept out of the device walk because an H2D copy from pageable memory is
    /// capture-illegal - and because staging is what makes one captured tick
    /// valid at every position.
    fn stage_inputs(&mut self, ids: &[u32]) -> Result<(), GpuModelError> {
        let n = ids.len();
        let base = self.pos;
        let pos: Vec<u32> = (0..n).map(|i| (base + i) as u32).collect();
        let mrope: Vec<u32> = (0..4).flat_map(|_| pos.iter().copied()).collect();
        self.exec.upload_u32(ids, &mut self.sc.d_tok)?;
        self.exec.upload_u32(&pos, &mut self.sc.d_pos)?;
        self.exec.upload_u32(&mrope, &mut self.sc.d_mrope)?;
        self.exec.upload_u32(&vec![0u32; n], &mut self.sc.d_slots)?;
        for li in 0..self.cfg.n_layer {
            if let Some(ple) = self.layers[li].ple.as_ref() {
                let emb = gather_ple_rows(&self.st, &self.cfg, ple, li, &self.stream, base + 2, n)?;
                self.exec.upload_f32(&emb, &mut self.sc.d_emb)?;
            }
        }
        Ok(())
    }

    /// The device-only layer walk. Contains no host reads, no allocation and
    /// no data-dependent host branching, so a decode-shaped call is capturable
    /// as-is (the `Dump` triage path is the one exception, and it refuses to
    /// coexist with capture).
    fn device_walk(&mut self, n: usize, phase: Phase) -> Result<(), GpuModelError> {
        let Self {
            exec: e,
            cfg: c,
            layers,
            embed,
            lm_head,
            final_mix,
            max_tokens,
            sc,
            recur,
            kv_k,
            kv_v,
            gdn_win,
            ple_win,
            stage,
            ..
        } = self;
        let (h, hw, hc, lr, eps) = (c.hidden, c.hc_width(), c.hc_count, c.hc_lowrank, c.eps);
        let dump = Dump::arm();

        // ---- embed -> the 4-stream hyper-connection state -----------------
        e.embed_gather_bf16(embed, &sc.d_tok, &mut sc.d_x, h, n, 1.0)?;
        for t in 0..n {
            for s in 0..hc {
                e.copy_region(&sc.d_x, t * h, &mut sc.d_h, t * hw + s * h, h)?;
            }
        }
        dump.put(e, usize::MAX, "h_embed", &sc.d_h, n * hw)?;

        // Carries whether the previous combine already left this mix's
        // normalized state in d_xn. False at entry: the first mix reads the
        // freshly broadcast embedding, which no combine produced.
        let mut pre_normed = false;
        for li in 0..c.n_layer {
            let layer = &layers[li];
            if let Some(ple) = layer.ple.as_ref() {
                // rows already staged into d_emb by `stage_inputs`
                ple_pass(
                    e,
                    c,
                    ple,
                    sc,
                    stage,
                    n,
                    phase,
                    ple_win.as_mut().expect("ple window"),
                )?;
                debug_assert!(
                    !pre_normed,
                    "a PLE layer must never be handed a pre-normed state"
                );
                if dump.on() {
                    dump.put(e, li, "ple_gv", &sc.d_pgv, n * hw)?;
                    dump.put(e, li, "ple_conv", &sc.d_pconv, n * hw)?;
                    dump.put(e, li, "h_ple", &sc.d_h, n * hw)?;
                }
            }

            let attn_inj = hc_mix_pass(e, c, &layer.attn_hc, sc, stage, n, pre_normed)?;
            if dump.on() {
                dump.put(e, li, "attn_bi", &sc.d_bi, n * h)?;
                dump_inj(e, &dump, li, "attn_inj", sc, attn_inj, n, hc)?;
            }
            match c.blocks[li] {
                Qwen4ExpBlock::Gdn => {
                    let MixerW::Gdn(w) = &layer.mixer else {
                        return Err(GpuModelError::Unsupported(
                            "gdn layer has an attn mixer".into(),
                        ));
                    };
                    gdn_pass(
                        e,
                        c,
                        w,
                        sc,
                        stage,
                        recur[li].as_mut().expect("gdn state"),
                        gdn_win[li].as_mut().expect("gdn conv window"),
                        n,
                        phase,
                    )?;
                }
                Qwen4ExpBlock::Attention => {
                    let MixerW::Attn(w) = &layer.mixer else {
                        return Err(GpuModelError::Unsupported(
                            "attn layer has a gdn mixer".into(),
                        ));
                    };
                    attn_pass(
                        e,
                        c,
                        w,
                        sc,
                        stage,
                        kv_k[li].as_mut().expect("kv k"),
                        kv_v[li].as_mut().expect("kv v"),
                        *max_tokens,
                        n,
                        phase,
                    )?;
                }
            }
            dump.put(e, li, "mix_out", &sc.d_mix, n * h)?;
            // the mlp mix always reads this combine's own output
            let mlp_pre = combine(e, sc, attn_inj, Some(&layer.mlp_hc.norm), n, hc, h, eps)?;
            dump.put(e, li, "h_mid", &sc.d_h, n * hw)?;

            let mlp_inj = hc_mix_pass(e, c, &layer.mlp_hc, sc, stage, n, mlp_pre)?;
            if dump.on() {
                dump.put(e, li, "mlp_bi", &sc.d_bi, n * h)?;
                dump_inj(e, &dump, li, "mlp_inj", sc, mlp_inj, n, hc)?;
            }
            moe_pass(e, c, &layer.moe, sc, stage, n)?;
            dump.put(e, li, "moe_out", &sc.d_mix, n * h)?;
            // Whoever reads the state next: the following layer's attention
            // mix, or the final mixer. Not fusable when the next layer carries
            // a PLE - that layer ADDS to the state before its mix reads it, so
            // a norm taken here would be of the wrong thing.
            let next_norm = if li + 1 < c.n_layer {
                if layers[li + 1].ple.is_some() {
                    None
                } else {
                    Some(&layers[li + 1].attn_hc.norm)
                }
            } else {
                Some(&final_mix.norm)
            };
            pre_normed = combine(e, sc, mlp_inj, next_norm, n, hc, h, eps)?;
            dump.put(e, li, "h_out", &sc.d_h, n * hw)?;
        }

        // ---- final mixer (no inject) -> lm_head on the last position -----
        if !pre_normed {
            e.q4x_group_norm_1p(&sc.d_h, &final_mix.norm.buf, &mut sc.d_xn, n, hc, h, eps)?;
        }
        final_mix.down.matmul(e, &sc.d_xn, &mut sc.d_m, n, stage)?;
        e.q4x_scale_silu(&mut sc.d_m, n * lr, 1.0 / hc as f32)?;
        final_mix.up.matmul(e, &sc.d_m, &mut sc.d_gate, n, stage)?;
        e.q4x_hc_mix(&sc.d_xn, &sc.d_gate, &mut sc.d_bi, n, hc, h)?;
        e.copy_region(&sc.d_bi, (n - 1) * h, &mut sc.d_fin, 0, h)?;
        dump.put(e, usize::MAX, "fin", &sc.d_fin, h)?;
        lm_head.matmul(e, &sc.d_fin, &mut sc.d_out, 1, stage)?;
        dump.put(e, usize::MAX, "logits", &sc.d_out, c.vocab)?;
        Ok(())
    }
}

/// Which shape the walk runs in. Prefill sees the whole span at once and can
/// use the sequence-form convs and the tiled prefill attention; decode sees
/// one token and reads its history out of the carried windows and the KV.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Prefill,
    Decode,
}

/// Graph capture is on unless killed, and never while the triage dump is
/// armed - the dump reads device memory back mid-walk, which is
/// capture-illegal and would poison the whole tick.
fn capture_wanted() -> bool {
    std::env::var_os("PADDOCK_Q38FN_NO_GRAPH").is_none()
        && std::env::var_os("PADDOCK_Q38FN_DUMP").is_none()
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best
}

/// Apply the combine against whichever place the mix left its inject, FUSING
/// the grouped norm that consumes the result when there is one - which is
/// every combine except the one whose output a PLE layer modifies before the
/// next mix reads it. Returns whether `d_xn` now holds that mix's normalized
/// state.
#[allow(clippy::too_many_arguments)]
fn combine(
    e: &GpuExecutor,
    sc: &mut Scratch,
    inj: Inj,
    next_norm: Option<&DeviceTensor>,
    n: usize,
    hc: usize,
    hidden: usize,
    eps: f32,
) -> Result<bool, GpuModelError> {
    // d_h, d_mix, d_m/d_inj and d_xn are disjoint fields, so the borrows below
    // are all simultaneous without any shuffling.
    match (inj, next_norm) {
        (Inj::InM(off), Some(nw)) => {
            e.q4x_combine_norm(
                &mut sc.d_h,
                &sc.d_mix,
                &sc.d_m,
                off,
                &nw.buf,
                &mut sc.d_xn,
                n,
                hc,
                hidden,
                eps,
            )?;
            Ok(true)
        }
        (Inj::Separate, Some(nw)) => {
            e.q4x_combine_norm(
                &mut sc.d_h,
                &sc.d_mix,
                &sc.d_inj,
                0,
                &nw.buf,
                &mut sc.d_xn,
                n,
                hc,
                hidden,
                eps,
            )?;
            Ok(true)
        }
        (Inj::InM(off), None) => {
            e.q4x_hc_combine_at(&mut sc.d_h, &sc.d_mix, &sc.d_m, off, n, hc, hidden)?;
            Ok(false)
        }
        (Inj::Separate, None) => {
            e.q4x_hc_combine(&mut sc.d_h, &sc.d_mix, &sc.d_inj, n, hc, hidden)?;
            Ok(false)
        }
    }
}

/// Triage dump of the inject logits, wherever the fold put them.
#[allow(clippy::too_many_arguments)]
fn dump_inj(
    e: &GpuExecutor,
    dump: &Dump,
    li: usize,
    tag: &str,
    sc: &Scratch,
    inj: Inj,
    n: usize,
    hc: usize,
) -> Result<(), GpuModelError> {
    match inj {
        Inj::InM(off) => {
            let v = e.to_host_len(&sc.d_m, off + n * hc)?;
            dump.put_host(li, tag, &v[off..off + n * hc])
        }
        Inj::Separate => dump.put(e, li, tag, &sc.d_inj, n * hc),
    }
}

/// One hyper-connection mix: grouped (1+w) norm -> low-rank down -> scale+silu
/// -> up -> gated reduce, plus the raw inject logits. Leaves `d_bi` = block
/// input and `d_inj` = inject logits.
#[allow(clippy::too_many_arguments)]
fn hc_mix_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &HcW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    n: usize,
    pre_normed: bool,
) -> Result<Inj, GpuModelError> {
    let (h, hc, lr) = (c.hidden, c.hc_count, c.hc_lowrank);
    // `pre_normed`: the preceding combine already produced this mix's
    // normalized state as part of its own single pass (slot 517).
    if !pre_normed {
        e.q4x_group_norm_1p(&sc.d_h, &w.norm.buf, &mut sc.d_xn, n, hc, h, c.eps)?;
    }
    let inj = if w.inject_rows > 0 && n == 1 {
        // One launch for both projections: the inject logits come out as the
        // tail of the low-rank output and are read there, so folding the
        // launch does not cost a copy back.
        w.down.matmul(e, &sc.d_xn, &mut sc.d_m, 1, stage)?;
        Inj::InM(lr)
    } else if w.inject_rows > 0 {
        w.down.matmul_rows(e, 0, lr, &sc.d_xn, &mut sc.d_m, n)?;
        w.down.matmul_rows(e, lr, hc, &sc.d_xn, &mut sc.d_inj, n)?;
        Inj::Separate
    } else {
        w.down.matmul(e, &sc.d_xn, &mut sc.d_m, n, stage)?;
        let wi = w.inject.as_ref().expect("unfolded block hc carries inject");
        e.matvec_f32_raw(&wi.buf, hc * h, hc, &sc.d_xn, &mut sc.d_inj, n)?;
        Inj::Separate
    };
    e.q4x_scale_silu(&mut sc.d_m, n * lr, 1.0 / hc as f32)?;
    w.up.matmul(e, &sc.d_m, &mut sc.d_gate, n, stage)?;
    e.q4x_hc_mix(&sc.d_xn, &sc.d_gate, &mut sc.d_bi, n, hc, h)?;
    Ok(inj)
}

/// Where a mix pass left its inject logits: folded into the tail of the
/// low-rank output at an element offset, or in its own plane.
#[derive(Clone, Copy)]
enum Inj {
    InM(usize),
    Separate,
}

/// PLE n-gram layer: device projections off the host-gathered rows, per-stream
/// gate, dilated conv, then `H += gv + conv`.
#[allow(clippy::too_many_arguments)]
fn ple_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    ple: &PleW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    n: usize,
    phase: Phase,
    win: &mut CudaSlice<f32>,
) -> Result<(), GpuModelError> {
    let (h, hw, hc, eps) = (c.hidden, c.hc_width(), c.hc_count, c.eps);
    let wrows = (c.ple_conv - 1) * PLE_DILATION;
    ple.key.matmul(e, &sc.d_emb, &mut sc.d_pkey, n, stage)?;
    ple.value.matmul(e, &sc.d_emb, &mut sc.d_pval, n, stage)?;
    e.q4x_group_norm_1p(&sc.d_pkey, &ple.norm_key.buf, &mut sc.d_pkn, n, hc, h, eps)?;
    e.q4x_group_norm_1p(&sc.d_h, &ple.norm_query.buf, &mut sc.d_pqn, n, hc, h, eps)?;
    e.q4x_ple_gate(&sc.d_pkn, &sc.d_pqn, &sc.d_pval, &mut sc.d_pgv, n, hc, h)?;
    // the conv rides norm_conv(gv); d_pkn is free again, reuse it as the source
    e.q4x_group_norm_1p(&sc.d_pgv, &ple.norm_conv.buf, &mut sc.d_pkn, n, hc, h, eps)?;
    match phase {
        Phase::Prefill => {
            e.q4x_conv_dil(
                &sc.d_pkn,
                &ple.conv.buf,
                &mut sc.d_pconv,
                n,
                hw,
                c.ple_conv,
                PLE_DILATION,
            )?;
            // leave the window holding the last `wrows` PRE-conv rows,
            // oldest-first, so a decode step continues the same convolution.
            // A prompt shorter than the window lands at the TAIL - the head
            // keeps the zeros `reset` left, which is exactly the zero left-pad
            // the sequence form applies.
            if n >= wrows {
                e.copy_region(&sc.d_pkn, (n - wrows) * hw, win, 0, wrows * hw)?;
            } else {
                e.copy_region(&sc.d_pkn, 0, win, (wrows - n) * hw, n * hw)?;
            }
        }
        Phase::Decode => {
            e.q4x_conv_dil_step(
                &sc.d_pkn,
                win,
                &ple.conv.buf,
                &mut sc.d_pconv,
                hw,
                c.ple_conv,
                PLE_DILATION,
            )?;
            // advance: drop the oldest row, append this token's pre-conv row.
            // `q4x_conv_dil_step` is stateless deliberately (graph-safe), so the
            // shift lives here; it goes through a scratch row because a
            // device-to-device copy inside one buffer would overlap.
            e.copy_region(win, hw, &mut sc.d_pwin_tmp, 0, (wrows - 1) * hw)?;
            e.copy_region(&sc.d_pwin_tmp, 0, win, 0, (wrows - 1) * hw)?;
            e.copy_region(&sc.d_pkn, 0, win, (wrows - 1) * hw, hw)?;
        }
    }
    e.add(&mut sc.d_h, &sc.d_pgv, n * hw)?;
    e.add(&mut sc.d_h, &sc.d_pconv, n * hw)?;
    Ok(())
}

/// The n-gram ids are a pure function of the token stream, so they are computed
/// host-side and the 16 x 160 fp8 rows are gathered from the still-mapped
/// shards and widened. Returns `[n, ple_embed]` f32.
#[allow(clippy::too_many_arguments)]
fn gather_ple_rows(
    st: &ShardedSafetensors,
    c: &Qwen4ExpConfig,
    ple: &PleW,
    li: usize,
    stream: &[i64],
    first: usize,
    n: usize,
) -> Result<Vec<f32>, GpuModelError> {
    let width = c.ple_embed / c.ple_heads();
    let emb_p = format!("model.language_model.layers.{li}.ple.ple_embedding");
    // take the shard row split from shard 0's own shape, so a re-sharded
    // checkpoint cannot silently read the wrong row
    let rows_per_shard = {
        let name = format!("{emb_p}.ngram_embedding.shard_0.weight");
        let (t, _) = st
            .bytes(&name)
            .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
        t.shape[0]
    };
    // the caller carries the 2-token EOS priming on the front of `stream`
    // (vLLM's `ngram_context`), so a decode step hashes the same window a
    // prefill of the whole sequence would have.
    let eos = c.bos_id as i64;
    let mut out = vec![0f32; n * c.ple_embed];
    for t in 0..n {
        let w3 = rq::ple_window(stream, first + t, eos);
        let row_ids = rq::ple_ngram_ids(
            &w3,
            &ple.multipliers,
            &ple.head_vocab,
            &ple.head_offset,
            c.heads_per_ngram,
        );
        for (hh, &rid) in row_ids.iter().enumerate() {
            let rid = rid as usize;
            let (sh, local) = (rid / rows_per_shard, rid % rows_per_shard);
            let name = format!("{emb_p}.ngram_embedding.shard_{sh}.weight");
            let (tinfo, sb) = st
                .bytes(&name)
                .ok_or_else(|| GpuModelError::Unsupported(format!("{name}: missing")))?;
            if tinfo.dtype != StDtype::F8E4m3 {
                return Err(GpuModelError::Unsupported(format!(
                    "{name}: dtype {:?}, want F8E4m3",
                    tinfo.dtype
                )));
            }
            let row = &sb[local * width..(local + 1) * width];
            let dst = t * c.ple_embed + hh * width;
            for (i, &byte) in row.iter().enumerate() {
                out[dst + i] = rq::e4m3_to_f32(byte) * ple.table_scale;
            }
        }
    }
    Ok(out)
}

/// Gated DeltaNet mixer. Writes `d_mix` `[n, hidden]` and advances `state`.
#[allow(clippy::too_many_arguments)]
fn gdn_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::GdnW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    state: &mut CudaSlice<f32>,
    win: &mut CudaSlice<f32>,
    n: usize,
    phase: Phase,
) -> Result<(), GpuModelError> {
    let (h, hv, kd, vd) = (c.hidden, c.gdn_v_heads, c.gdn_k_dim, c.gdn_v_dim);
    let (qkv_rows, km1) = (c.gdn_qkv_rows(), c.gdn_conv - 1);
    w.qkv.matmul(e, &sc.d_bi, &mut sc.d_qkv, n, stage)?;
    w.z.matmul(e, &sc.d_bi, &mut sc.d_zg, n, stage)?;
    // one plane, one launch: rows [0,h) are alpha and [h,2h) beta, which is
    // delta_gate_ab's own layout
    e.matvec_f32_raw(&w.ab.buf, h, 2 * hv, &sc.d_bi, &mut sc.d_ab, n)?;
    match phase {
        Phase::Prefill => {
            e.causal_conv1d_silu(
                &sc.d_qkv,
                &w.conv.buf,
                &mut sc.d_conv,
                n,
                qkv_rows,
                c.gdn_conv,
            )?;
            // window = the last k-1 PRE-conv rows, oldest first (conv_step's
            // contract); a short prompt lands at the tail over the reset zeros
            if n >= km1 {
                e.copy_region(&sc.d_qkv, (n - km1) * qkv_rows, win, 0, km1 * qkv_rows)?;
            } else {
                e.copy_region(&sc.d_qkv, 0, win, (km1 - n) * qkv_rows, n * qkv_rows)?;
            }
        }
        // conv_step shifts the window itself
        Phase::Decode => e.conv_step(
            win,
            &sc.d_qkv,
            &w.conv.buf,
            &mut sc.d_conv,
            qkv_rows,
            c.gdn_conv,
        )?,
    }
    e.q4x_gdn_split_widen(
        &sc.d_conv,
        &mut sc.d_dq,
        &mut sc.d_dk,
        &mut sc.d_dv,
        n,
        c.gdn_k_heads,
        hv,
        kd,
        vd,
    )?;
    // g = ssm_a * softplus(a + dt_bias) with ssm_a = -exp(A_log) folded at
    // load; beta = sigmoid(b). The same expressions as reference::gdn_gates.
    e.delta_gate_ab(
        &sc.d_ab,
        &w.ssm_a.buf,
        &w.dt_bias.buf,
        &mut sc.d_g,
        &mut sc.d_beta,
        n,
        hv,
    )?;
    e.gated_delta_recurrent(
        &sc.d_dq,
        &sc.d_dk,
        &sc.d_dv,
        &sc.d_g,
        &sc.d_beta,
        state,
        &mut sc.d_dattn,
        n,
        hv,
        kd,
    )?;
    e.q4x_gdn_gated_norm(
        &sc.d_dattn,
        &sc.d_zg,
        &w.norm.buf,
        &mut sc.d_core,
        n * hv,
        vd,
        c.eps,
    )?;
    w.out.matmul(e, &sc.d_core, &mut sc.d_mix, n, stage)?;
    Ok(())
}

/// Gated full attention, dense path. The QSA sparse walk is a later rung and
/// is exact anyway while the visible window stays inside the indexer budget.
#[allow(clippy::too_many_arguments)]
fn attn_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::AttnW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    kc: &mut CudaSlice<u8>,
    vc: &mut CudaSlice<u8>,
    max_ctx: usize,
    n: usize,
    phase: Phase,
) -> Result<(), GpuModelError> {
    let (nh, nkv, hd) = (c.n_heads, c.n_kv_heads, c.head_dim);
    let (kv_dim, q_dim) = (nkv * hd, nh * hd);
    let yarn = yarn_params(c);
    w.q.matmul(e, &sc.d_bi, &mut sc.d_qg, n, stage)?;
    e.split_qg(&sc.d_qg, &mut sc.d_q, &mut sc.d_agate, n, nh, hd)?;
    w.k.matmul(e, &sc.d_bi, &mut sc.d_k, n, stage)?;
    w.v.matmul(e, &sc.d_bi, &mut sc.d_v, n, stage)?;
    // q_norm/k_norm carry the +1 already (Gemma (1+w), folded at load)
    e.rmsnorm_batch(&sc.d_q, &w.q_norm.buf, &mut sc.d_qn, hd, c.eps, n * nh)?;
    e.rmsnorm_batch(&sc.d_k, &w.k_norm.buf, &mut sc.d_kn, hd, c.eps, n * nkv)?;
    e.mrope(
        &mut sc.d_qn,
        &sc.d_mrope,
        n,
        nh,
        hd,
        c.rotary_dim,
        yarn,
        MROPE_SECTIONS,
    )?;
    e.mrope(
        &mut sc.d_kn,
        &sc.d_mrope,
        n,
        nkv,
        hd,
        c.rotary_dim,
        yarn,
        MROPE_SECTIONS,
    )?;
    e.kv_append_batch(
        &sc.d_kn,
        kc,
        &sc.d_pos,
        Some(&sc.d_slots),
        kv_dim,
        max_ctx,
        n,
        KV,
    )?;
    e.kv_append_batch(
        &sc.d_v,
        vc,
        &sc.d_pos,
        Some(&sc.d_slots),
        kv_dim,
        max_ctx,
        n,
        KV,
    )?;
    let scale = 1.0 / (hd as f32).sqrt();
    match phase {
        Phase::Prefill => e.attn_prefill(
            &sc.d_qn,
            kc,
            vc,
            &sc.d_sinks,
            &mut sc.d_attn,
            &sc.d_pos,
            &sc.d_slots,
            nh,
            nkv,
            hd,
            max_ctx,
            kv_dim,
            0,
            n,
            scale,
            KV,
        )?,
        // one query row against the whole carried cache
        Phase::Decode => e.attn_decode_batch(
            &sc.d_qn,
            kc,
            vc,
            &sc.d_sinks,
            &mut sc.d_attn,
            &sc.d_pos,
            Some(&sc.d_slots),
            nh,
            nkv,
            hd,
            max_ctx,
            kv_dim,
            0,
            n,
            scale,
            KV,
        )?,
    }
    e.mul_sigmoid(&mut sc.d_attn, &sc.d_agate, n * q_dim)?;
    w.o.matmul(e, &sc.d_attn, &mut sc.d_mix, n, stage)?;
    Ok(())
}

/// 512-expert top-10 NVFP4 MoE + the bf16 sigmoid-gated shared expert.
#[allow(clippy::too_many_arguments)]
fn moe_pass(
    e: &GpuExecutor,
    c: &Qwen4ExpConfig,
    w: &super::MoeW,
    sc: &mut Scratch,
    stage: &mut DenseStage,
    n: usize,
) -> Result<(), GpuModelError> {
    let (h, k, sff) = (c.hidden, c.n_active, c.shared_ff);
    // Router: softmax over all experts, top-k, renormalized over the picks -
    // which is exactly moe_topk_batch's local softmax over the selected logits
    // (the global denominator cancels). Bias is zero for this family.
    // one launch covers the router AND the shared expert's scalar gate (row
    // n_expert). At batch 1 the topk reads logits[0..n_expert] in place; above
    // it the two are row-segment reads of the same residency, because a fused
    // output is only contiguous per projection at one row.
    let fused_router = n == 1;
    if fused_router {
        e.matvec_f32_raw(
            &w.router.buf,
            h,
            c.n_expert + 1,
            &sc.d_bi,
            &mut sc.d_logits,
            1,
        )?;
    } else {
        e.matvec_f32_rows(
            &w.router.buf,
            0,
            h,
            c.n_expert,
            &sc.d_bi,
            &mut sc.d_logits,
            n,
        )?;
    }
    e.moe_topk_batch(
        &sc.d_logits,
        &sc.d_zero_bias,
        c.n_expert,
        k,
        &mut sc.d_idx,
        &mut sc.d_topw,
        n,
    )?;
    e.q4x_moe_gu_swiglu(&w.gate, &w.up, &sc.d_idx, &sc.d_bi, &mut sc.d_act, k, n)?;
    e.nvf4_moe_down_acc(
        &w.down,
        &sc.d_idx,
        &sc.d_topw,
        &sc.d_act,
        &mut sc.d_mix,
        k,
        n,
        false,
    )?;
    // shared expert: swiglu, then a per-token sigmoid scalar gate
    w.sh_gate.matmul(e, &sc.d_bi, &mut sc.d_shg, n, stage)?;
    w.sh_up.matmul(e, &sc.d_bi, &mut sc.d_shu, n, stage)?;
    e.swiglu(&mut sc.d_shg, &sc.d_shu, n * sff)?;
    w.sh_down.matmul(e, &sc.d_shg, &mut sc.d_shd, n, stage)?;
    if fused_router {
        e.q4x_add_gated_row_at(&mut sc.d_mix, &sc.d_shd, &sc.d_logits, c.n_expert, n, h)?;
    } else {
        e.matvec_f32_rows(
            &w.router.buf,
            c.n_expert,
            h,
            1,
            &sc.d_bi,
            &mut sc.d_shgate,
            n,
        )?;
        e.q4x_add_gated_row(&mut sc.d_mix, &sc.d_shd, &sc.d_shgate, n, h)?;
    }
    Ok(())
}

/// YaRN parameters for this family: plain rope, theta 1e7, no scaling.
/// `ext_factor = 0` makes the correction band inert - the pack kernel's way of
/// saying "no YaRN", which is what a `rope_parameters` dict with no scaling
/// means.
fn yarn_params(c: &Qwen4ExpConfig) -> (f32, f32, f32, f32, f32, f32) {
    let theta_scale = c.rope_theta.powf(-2.0 / c.rotary_dim as f32);
    (theta_scale, 1.0, 0.0, 1.0, 0.0, 1.0)
}

fn bf16_plane(
    exec: &Arc<GpuExecutor>,
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

#[allow(clippy::type_complexity)]
fn alloc_state(
    e: &Arc<GpuExecutor>,
    c: &Qwen4ExpConfig,
    max_tokens: usize,
) -> Result<
    (
        Vec<Option<CudaSlice<f32>>>,
        Vec<Option<CudaSlice<u8>>>,
        Vec<Option<CudaSlice<u8>>>,
    ),
    GpuModelError,
> {
    let kv_bytes = max_tokens * c.n_kv_heads * c.head_dim * KV.bytes();
    let state_len = c.gdn_v_heads * c.gdn_k_dim * c.gdn_v_dim;
    let (mut recur, mut kk, mut kv) = (Vec::new(), Vec::new(), Vec::new());
    for li in 0..c.n_layer {
        match c.blocks[li] {
            Qwen4ExpBlock::Gdn => {
                recur.push(Some(e.alloc(state_len)?));
                kk.push(None);
                kv.push(None);
            }
            Qwen4ExpBlock::Attention => {
                recur.push(None);
                kk.push(Some(e.alloc_u8(kv_bytes)?));
                kv.push(Some(e.alloc_u8(kv_bytes)?));
            }
        }
    }
    Ok((recur, kk, kv))
}

impl Scratch {
    fn new(e: &Arc<GpuExecutor>, c: &Qwen4ExpConfig, t: usize) -> Result<Self, GpuModelError> {
        let (h, hw, hc) = (c.hidden, c.hc_width(), c.hc_count);
        let kv_dim = c.n_kv_heads * c.head_dim;
        let q_dim = c.n_heads * c.head_dim;
        let vdim = c.gdn_v_heads * c.gdn_v_dim;
        let kdim = c.gdn_v_heads * c.gdn_k_dim;
        Ok(Self {
            d_tok: e.alloc_u32(t)?,
            d_pos: e.alloc_u32(t)?,
            d_mrope: e.alloc_u32(4 * t)?,
            d_slots: e.alloc_u32(t)?,
            d_x: e.alloc(t * h)?,
            d_h: e.alloc(t * hw)?,
            d_xn: e.alloc(t * hw)?,
            // + hc: the folded inject rows land in this plane's tail at batch 1
            d_m: e.alloc(t * c.hc_lowrank + c.hc_count)?,
            d_gate: e.alloc(t * hw)?,
            d_bi: e.alloc(t * h)?,
            d_inj: e.alloc(t * hc)?,
            d_mix: e.alloc(t * h)?,
            d_qkv: e.alloc(t * c.gdn_qkv_rows())?,
            d_zg: e.alloc(t * c.gdn_z_rows())?,
            d_ab: e.alloc(t * 2 * c.gdn_v_heads)?,
            d_g: e.alloc(t * c.gdn_v_heads)?,
            d_beta: e.alloc(t * c.gdn_v_heads)?,
            d_conv: e.alloc(t * c.gdn_qkv_rows())?,
            d_dq: e.alloc(t * kdim)?,
            d_dk: e.alloc(t * kdim)?,
            d_dv: e.alloc(t * vdim)?,
            d_dattn: e.alloc(t * vdim)?,
            d_core: e.alloc(t * vdim)?,
            d_qg: e.alloc(t * c.attn_q_rows())?,
            d_q: e.alloc(t * q_dim)?,
            d_agate: e.alloc(t * q_dim)?,
            d_k: e.alloc(t * kv_dim)?,
            d_v: e.alloc(t * kv_dim)?,
            d_qn: e.alloc(t * q_dim)?,
            d_kn: e.alloc(t * kv_dim)?,
            d_attn: e.alloc(t * q_dim)?,
            d_sinks: e.alloc_no_sinks(c.n_heads)?,
            // + 1 row: the folded shared-expert gate
            d_logits: e.alloc(t * (c.n_expert + 1))?,
            d_zero_bias: e.alloc(c.n_expert)?,
            d_idx: e.alloc_u32(t * c.n_active)?,
            d_topw: e.alloc(t * c.n_active)?,
            d_act: e.alloc(t * c.n_active * c.moe_ff)?,
            d_shg: e.alloc(t * c.shared_ff)?,
            d_shu: e.alloc(t * c.shared_ff)?,
            d_shd: e.alloc(t * h)?,
            d_shgate: e.alloc(t)?,
            d_emb: e.alloc(t * c.ple_embed)?,
            d_pkey: e.alloc(t * hw)?,
            d_pval: e.alloc(t * h)?,
            d_pkn: e.alloc(t * hw)?,
            d_pqn: e.alloc(t * hw)?,
            d_pgv: e.alloc(t * hw)?,
            d_pconv: e.alloc(t * hw)?,
            d_pwin_tmp: e.alloc((c.ple_conv - 1) * PLE_DILATION * hw)?,
            d_fin: e.alloc(h)?,
            d_out: e.alloc(c.vocab)?,
        })
    }
}
