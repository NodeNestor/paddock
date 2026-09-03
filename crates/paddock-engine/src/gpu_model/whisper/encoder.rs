//! Whisper audio encoder. Reference: transformers
//! `modeling_whisper.py` `WhisperEncoder`. Dataflow for one 30 s window:
//!
//!   normalized log-mel [3000, 128] (host, `crate::audio`, whisper policy)
//!   -> conv1 k3/s1/p1 (128 -> 1280) + bias + GELU-erf         [3000, 1280]
//!   -> conv2 k3/s2/p1 (1280 -> 1280) + bias + GELU-erf        [1500, 1280]
//!   -> + the stored sinusoid position table (rows 0..1500)
//!   -> 32 × pre-LN blocks [LN -> q(+bias)/k(no bias)/v(+bias) -> full
//!     bidirectional attention over all 1500 frames -> out(+bias) -> +res
//!     -> LN -> fc1(+bias) -> GELU-erf -> fc2(+bias) -> +res]
//!   -> final LN
//!   -> [1500, 1280] encoder states, which the decoder turns into the decoder's
//!     static per-layer cross-attention K/V.
//!
//! Chassis is the qwen3_asr audio tower's: f32 activations over f16 weight
//! planes, one f16 staging buffer per GEMM, `vision_attn_at` for the
//! unmasked attention (hd 64 rides its mma path). Two structural
//! differences from that tower, both from whisper being fixed-geometry:
//! attention is over the whole window (no 104-token windows, so one launch
//! per layer), and the conv stem runs once over all 3000 frames rather than
//! in bounded chunk groups - the window is a constant 30 s, so the scratch
//! is a constant ~135 MB and needs no grouping.
//!
//! SERVING SHAPE. Both of the bring-up path's per-window host
//! costs are gone, because the transcriber thread that calls this is the same
//! thread that runs every other request's decode steps - anything serial here
//! is a stall for everyone:
//!   - the scratch is RESIDENT (`EncScratch`, allocated once). It used to be
//!     ~135 MB of `cudaMalloc`/free per window, and mid-serve allocation is a
//!     known serve-killer.
//!   - conv1's im2col is a GPU gather off the uploaded mel, not a host loop
//!     into a 4.6 MB staging vector. Identical values (the gather table is
//!     the same tap arithmetic), a third of the host->device bytes, and no
//!     per-window host pass over 1.15 M floats.
//!     Both index tables are constants of the geometry, so they are built once
//!     with the scratch rather than per call.
//!
//! Padding semantics, the seam that cost the Qwen3-ASR bring-up a bug:
//! conv1's p=1 pads the MEL with zeros (the gather table below points its
//! out-of-range taps at an always-zero pad row), and conv2's p=1 pads its own
//! INPUT - the post-GELU activations - also with zeros, not with GELU(bias).
//! Torch's `F.conv1d(padding=1)` zero-pads the tensor it is given, so the
//! always-zero pad row both gathers gather from is exactly right; deriving a
//! pad value from the bias would be the bug.

use cudarc::driver::CudaSlice;
use cudarc::driver::sys::CUstreamCaptureMode;
use half::f16;

use crate::audio::MelFeatures;
use crate::gpu::GpuError;
use crate::gpu_model::gpt_oss::GpuModelError;

use super::{GpuWhisper, SendGraph};

/// One encoded window: `[n_frames, d_model]` device-resident encoder states.
pub struct EncoderOutput {
    pub states: CudaSlice<f32>,
    pub n_frames: usize,
}

/// Every buffer one window's forward pass touches, allocated once. Sized by
/// the fixed 30 s geometry, so there is nothing per-request to grow.
pub(crate) struct EncScratch {
    /// mel frames plus one always-zero pad row, [(2n)+1, bins]. Uploads land
    /// in the front rows only, so the pad row keeps the zeros it was born
    /// with - that is what conv1's out-of-range taps gather.
    mel: CudaSlice<f32>,
    /// conv1's im2col gather table [(2n)*3] and conv2's [n*3], both constant
    im1: CudaSlice<u32>,
    im2: CudaSlice<u32>,
    /// gather landing, sized by the wider of the two stems
    gath: CudaSlice<f32>,
    /// conv1 landing plus its own always-zero pad row, [(2n)+1, d]
    c1: CudaSlice<f32>,
    /// the widest f16 GEMM input in flight anywhere in the pass
    s16: CudaSlice<f16>,
    /// the residual stream and the per-block landings
    x: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    /// the fused q|k|v GEMM landing, [n, 3*d] (- one M=3d GEMM
    /// where three M=d ones left half the tc5p clusters idle)
    qkv: CudaSlice<f32>,
    q: CudaSlice<f32>,
    k: CudaSlice<f32>,
    v: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    ff: CudaSlice<f32>,
    /// the pre-norm landing every GEMM eats, kept separate from `s16` so the
    /// wide fc1 staging cannot clobber a live norm
    n16: CudaSlice<f16>,
    bytes: u64,
    /// The captured admission tick: stem + 32 blocks + final LN + every
    /// layer's cross-K/V, ~520 launches replayed as one. Whisper's window is
    /// a CONSTANT 30 s, so unlike the decode tick there is only ever one
    /// shape to record - the mel and the target slot ride resident buffers
    /// uploaded before the replay. The launch train was 4.9 ms of host time
    /// per window (1.27 s of a 7.5 s c32 battery) with the GPU idling
    /// through it.
    /// keyed by the pass's audio count b - each batch width is its own
    /// fixed launch train
    graphs: std::collections::HashMap<usize, SendGraph>,
    /// a GEMM route may allocate on first sight of a shape, and an
    /// allocation during capture is a hard driver error - so each batch
    /// width's first pass runs eagerly and only its second is recorded.
    warmed: std::collections::HashSet<usize>,
    /// batch cap the buffers were laid out for (PADDOCK_WHISPER_ENC_BATCH)
    bmax: usize,
    /// audio count of the currently staged pass (set by upload_windows)
    cur_b: usize,
    /// the sinusoid table replicated bmax times, so positions add as one
    /// b*n-row launch (there is no offset-add primitive to loop instead)
    pos_rep: CudaSlice<f32>,
}

impl EncScratch {
    #[allow(dead_code)]
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The last encode's output states, `[b * n_audio_ctx, d_model]` f32,
    /// audio-major - valid until the next encode overwrites them.
    pub(crate) fn states(&self) -> &CudaSlice<f32> {
        &self.norm
    }

    /// Audio count of the currently staged pass.
    pub(crate) fn batch_staged(&self) -> usize {
        self.cur_b
    }

    /// Drop the recorded admission ticks - for when a buffer they address is
    /// about to go away. The next admission records fresh ones.
    pub(crate) fn forget_graph(&mut self) {
        self.graphs.clear();
    }
}

/// Batch cap for one whisper admission pass: how many pending
/// 30 s windows encode as one audio-major pass. Default 2 - the measured
/// tc5p arithmetic (P35/P37) has wo at 6.7us/audio vs the 7.61 frozen bar
/// at b=2, and c8's 8-deep concurrency makes pairs common. The dump lane
/// stays b=1 (its oracle tap reads one window's states).
pub(crate) fn enc_batch_env() -> usize {
    if paddock_models::dev_var_os!("PADDOCK_WHISPER_DUMP_ENC").is_some() {
        return 1;
    }
    paddock_models::dev_var!("PADDOCK_WHISPER_ENC_BATCH")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(2)
        .clamp(1, 8)
}

impl GpuWhisper {
    /// Allocate the encoder scratch and its two constant gather tables.
    /// Idempotent - every later window reuses them.
    pub(crate) fn ensure_enc_scratch(&mut self) -> Result<(), GpuModelError> {
        if self.enc.is_some() {
            return Ok(());
        }
        let exec = self.exec.clone();
        let (d, bins, ffn) = (self.hp.d_model, self.mel.bins, self.hp.enc_ffn);
        let n = self.hp.n_audio_ctx;
        let t_in = 2 * n;

        // Batched admission: buffers and gather tables laid out
        // audio-major for up to bmax windows; a pass uses the front b.
        let bmax = enc_batch_env();

        // conv1: row t's k-vector is [mel[t-1] | mel[t] | mel[t+1]], three
        // contiguous bin-runs (the load-time weight permute bought exactly
        // this). conv2: stride 2, same three taps off the conv1 landing.
        // Out-of-range taps address the pad row - the conv's zero padding -
        // which sits past all windows at bmax*t_in so every audio's taps
        // stay inside its own [ai*t_in, (ai+1)*t_in) rows.
        let mut im1 = Vec::with_capacity(bmax * t_in * 3);
        let mut im2 = Vec::with_capacity(bmax * n * 3);
        for ai in 0..bmax {
            for t in 0..t_in {
                for kp in 0..3usize {
                    let src = t as isize + kp as isize - 1;
                    im1.push(if src >= 0 && (src as usize) < t_in {
                        (ai * t_in + src as usize) as u32
                    } else {
                        (bmax * t_in) as u32
                    });
                }
            }
        }
        for ai in 0..bmax {
            for t in 0..n {
                for kp in 0..3usize {
                    let src = 2 * t as isize + kp as isize - 1;
                    im2.push(if src >= 0 && (src as usize) < t_in {
                        (ai * t_in + src as usize) as u32
                    } else {
                        (bmax * t_in) as u32
                    });
                }
            }
        }

        // the sinusoid table replicated per audio: one gather at init (the
        // identity-table trick - no new kernel, no per-pass work)
        let mut pos_rep = exec.alloc(bmax * n * d)?;
        {
            let idt: Vec<u32> = (0..bmax * n).map(|i| (i % n) as u32).collect();
            let idt = exec.to_device_u32(&idt)?;
            exec.gather_rows_avg(&self.enc_pos, &idt, &mut pos_rep, bmax * n, 1, d)?;
        }

        let stage_max = bmax * (t_in * 3 * bins).max(3 * n * d).max(n * ffn);
        let gath_max = bmax * (t_in * 3 * bins).max(n * 3 * d);
        let sc = EncScratch {
            bytes: ((bmax * t_in + 1) * bins * 4
                + (im1.len() + im2.len()) * 4
                + gath_max * 4
                + (bmax * t_in + 1) * d * 4
                + stage_max * 2
                + bmax * (11 * n * d * 4 + n * ffn * 4 + n * d * 2)) as u64,
            mel: exec.alloc(bmax * t_in * bins + bins)?,
            im1: exec.to_device_u32(&im1)?,
            im2: exec.to_device_u32(&im2)?,
            gath: exec.alloc(gath_max)?,
            c1: exec.alloc(bmax * t_in * d + d)?,
            s16: exec.alloc_f16(stage_max)?,
            x: exec.alloc(bmax * n * d)?,
            norm: exec.alloc(bmax * n * d)?,
            qkv: exec.alloc(bmax * n * 3 * d)?,
            q: exec.alloc(bmax * n * d)?,
            k: exec.alloc(bmax * n * d)?,
            v: exec.alloc(bmax * n * d)?,
            attn: exec.alloc(bmax * n * d)?,
            ff: exec.alloc(bmax * n * ffn)?,
            n16: exec.alloc_f16(bmax * n * d)?,
            graphs: std::collections::HashMap::new(),
            warmed: std::collections::HashSet::new(),
            bmax,
            cur_b: 1,
            pos_rep,
        };
        tracing::info!(
            mib = sc.bytes / (1 << 20),
            batch_cap = bmax,
            "whisper encoder scratch resident (30 s windows, reused)"
        );
        self.enc = Some(sc);
        Ok(())
    }

    /// Validate one window's mel and land it in the resident input buffer.
    /// Returns the encoder frame count. The upload is deliberately outside
    /// the captured tick: a host->device copy from pageable memory cannot be
    /// captured, and it is the one genuinely per-window input.
    pub(crate) fn upload_window(&mut self, mel: &MelFeatures) -> Result<usize, GpuModelError> {
        self.upload_windows(&[mel])
    }

    /// Stage 1..=bmax windows audio-major for one batched pass.
    pub(crate) fn upload_windows(&mut self, mels: &[&MelFeatures]) -> Result<usize, GpuModelError> {
        let bins = self.mel.bins;
        let n = self.hp.n_audio_ctx;
        // conv2's stride 2 is the only downsample: the window is exactly
        // twice the encoder's position count.
        let t_in = 2 * n;
        self.ensure_enc_scratch()?;
        let exec = self.exec.clone();
        let sc = self.enc.as_mut().expect("ensure_enc_scratch ran");
        if mels.is_empty() || mels.len() > sc.bmax {
            return Err(GpuModelError::Unsupported(format!(
                "whisper encoder: {} windows staged, batch cap is {}",
                mels.len(),
                sc.bmax
            )));
        }
        for (ai, mel) in mels.iter().enumerate() {
            if mel.data.len() != t_in * bins {
                return Err(GpuModelError::Unsupported(format!(
                    "whisper encoder: mel is {} values, want {t_in} frames x {bins} bins - feed \
                     whole 30 s windows (crate::audio::whisper_features)",
                    mel.data.len()
                )));
            }
            exec.upload_f32_at(&mel.data, &mut sc.mel, ai * t_in * bins)?;
        }
        sc.cur_b = mels.len();
        Ok(n)
    }

    /// The encoder forward itself, off the already-uploaded window; the states
    /// land in `enc.norm` as `[n_audio_ctx, d_model]` f32.
    ///
    /// f32 states, not f16: they are the oracle gate's comparison
    /// surface, and the cross-K/V staging widens from them anyway.
    ///
    /// Allocation-free and host-read-free by construction - it is captured.
    pub(crate) fn encode_body(&mut self) -> Result<(), GpuModelError> {
        let (d, bins) = (self.hp.d_model, self.mel.bins);
        let n = self.hp.n_audio_ctx;
        let t_in = 2 * n;
        let exec = self.exec.clone();
        let ffn = self.hp.enc_ffn;
        let scale = 1.0 / (self.hp.head_dim as f32).sqrt();
        let eps = self.hp.eps;
        let sc = self.enc.as_mut().expect("ensure_enc_scratch ran");
        // audio count of the staged pass: every op below is
        // row-batched, so the whole body scales by b as pure row counts
        let b = sc.cur_b;

        // ---- conv1: gather the three taps per mel frame, stride 1 ----
        let k1 = 3 * bins;
        exec.gather_rows_avg(&sc.mel, &sc.im1, &mut sc.gath, b * t_in * 3, 1, bins)?;
        exec.convert_f32_f16(&sc.gath, &mut sc.s16, b * t_in * k1)?;
        exec.matvec_batch_f16(&self.conv1_w, &sc.s16, &mut sc.c1, b * t_in)?;
        exec.bias_add(&mut sc.c1, &self.conv1_b, b * t_in, d)?;
        exec.gelu_erf(&mut sc.c1, b * t_in * d)?;

        // ---- conv2: same three taps off the conv1 landing, stride 2 ----
        exec.gather_rows_avg(&sc.c1, &sc.im2, &mut sc.gath, b * n * 3, 1, d)?;
        exec.convert_f32_f16(&sc.gath, &mut sc.s16, b * n * 3 * d)?;
        exec.matvec_batch_f16(&self.conv2_w, &sc.s16, &mut sc.x, b * n)?;
        exec.bias_add(&mut sc.x, &self.conv2_b, b * n, d)?;
        exec.gelu_erf(&mut sc.x, b * n * d)?;

        // ---- positions: the replicated sinusoid table, added whole ----
        exec.add(&mut sc.x, &sc.pos_rep, b * n * d)?;

        // ---- 32 pre-LN blocks over the full window ----
        //
        // The epilogues run through the whisper decode lane's fused kernels
        // each is bit-identical to the op sequence it replaces -
        // gated in tests/gpu_whisper_kernels.rs - and they are row-batched,
        // so the encoder's 1500-row shapes use them unchanged. That matters
        // here for the same reason it did in the decoder: at 1500x1280 the
        // unfused chain paid four separate DRAM round trips per residual
        // seam, and traced at 3.4 ms of the encoder's 15.3 ms per window.
        let l0 = &self.enc_layers[0].attn.ln;
        exec.whisper_ln_f16(&sc.x, &l0.w, &l0.b, &mut sc.n16, b * n, d, eps)?;
        let n_layer = self.enc_layers.len();
        for li in 0..n_layer {
            let blk = &self.enc_layers[li];
            let a = &blk.attn;
            // q, k and v as one fused GEMM + a split that folds both biases
            // at 1500x1280 the three separate GEMMs each left
            // half the tc5p clusters idle (3x12.60us vs 19.09 fused), and
            // the two full-width bias_add launches ride the split for free.
            exec.matvec_batch_f16(&a.wqkv, &sc.n16, &mut sc.qkv, b * n)?;
            exec.whisper_enc_qkv_split(
                &sc.qkv,
                Some(&a.bq),
                Some(&a.bv),
                &mut sc.q,
                &mut sc.k,
                &mut sc.v,
                d,
                b * n,
            )?;
            // the batched form keeps each audio attending only to itself:
            // grid.z strides q/k/v by n*heads*hd, which is exactly the
            // audio-major layout the split just wrote
            exec.vision_attn_x(
                &sc.q,
                &sc.k,
                &sc.v,
                &mut sc.attn,
                n,
                n,
                self.hp.n_enc_heads,
                self.hp.head_dim,
                b,
                scale,
            )?;
            exec.convert_f32_f16(&sc.attn, &mut sc.s16, b * n * d)?;
            exec.matvec_batch_f16(&a.wo, &sc.s16, &mut sc.norm, b * n)?;
            let m = &blk.mlp;
            exec.whisper_res_ln_f16(
                &mut sc.x,
                &sc.norm,
                &a.bo,
                &m.ln.w,
                &m.ln.b,
                &mut sc.n16,
                b * n,
                d,
                eps,
            )?;

            exec.matvec_batch_f16(&m.fc1_w, &sc.n16, &mut sc.ff, b * n)?;
            exec.whisper_bias_gelu_f16(&sc.ff, &m.fc1_b, &mut sc.s16, b * n, ffn)?;
            exec.matvec_batch_f16(&m.fc2_w, &sc.s16, &mut sc.norm, b * n)?;
            if li + 1 < n_layer {
                let next = &self.enc_layers[li + 1].attn.ln;
                exec.whisper_res_ln_f16(
                    &mut sc.x,
                    &sc.norm,
                    &m.fc2_b,
                    &next.w,
                    &next.b,
                    &mut sc.n16,
                    b * n,
                    d,
                    eps,
                )?;
            } else {
                // the last seam feeds the encoder's own final norm, which
                // stays f32 (see the states note above)
                exec.bias_add(&mut sc.norm, &m.fc2_b, b * n, d)?;
                exec.add(&mut sc.x, &sc.norm, b * n * d)?;
            }
        }
        exec.layernorm(
            &sc.x,
            &self.enc_ln.w,
            &self.enc_ln.b,
            &mut sc.norm,
            b * n,
            d,
            self.hp.eps,
        )?;
        Ok(())
    }

    /// Replay the admission tick (encode + every layer's cross-K/V) for the
    /// slot already staged in `d_one`, recording it first if the shapes have
    /// been warmed but not yet captured. Same three-state ladder as the
    /// decode tick: launch -> warm eagerly -> capture.
    /// Returns true when the pass REPLAYED on `replay_on` (P38 side-stream
    /// overlap); false when it ran eagerly / was captured on the main stream.
    pub(crate) fn admit_replay(
        &mut self,
        replay_on: Option<&std::sync::Arc<cudarc::driver::CudaStream>>,
    ) -> Result<bool, GpuModelError> {
        // the oracle tap reads the states back on the host, which capture
        // forbids and a replay would skip - the dump lane stays eager
        let dumping = paddock_models::dev_var_os!("PADDOCK_WHISPER_DUMP_ENC").is_some();
        let sc = self.enc.as_ref().expect("scratch allocated");
        // each batch width b is its own fixed launch train:
        // graphs and warm state key on it
        let b = sc.cur_b;
        if let Some(g) = sc.graphs.get(&b) {
            let r = match replay_on {
                Some(s) => g.0.launch_on(s),
                None => g.0.launch(),
            };
            return r
                .map(|_| replay_on.is_some())
                .map_err(|e| GpuError::Driver(format!("whisper encode graph launch: {e}")).into());
        }
        if !sc.warmed.contains(&b) || dumping {
            self.admit_body()?;
            self.enc.as_mut().expect("scratch").warmed.insert(b);
            return Ok(false);
        }
        let exec = self.exec.clone();
        exec.stream
            .synchronize()
            .map_err(|e| GpuError::Driver(format!("whisper pre-capture sync: {e}")))?;
        exec.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| GpuError::Driver(format!("whisper encode begin_capture: {e}")))?;
        let rec = self.admit_body();
        let graph = crate::gpu::end_capture_no_flags(&exec.stream)
            .map_err(|e| GpuError::Driver(format!("whisper encode end_capture: {e}")));
        rec?; // a record failure is only surfaceable after capture ends cleanly
        let graph = graph?
            .ok_or_else(|| GpuError::Driver("whisper encode capture produced no graph".into()))?;
        graph
            .launch()
            .map_err(|e| GpuError::Driver(format!("whisper encode graph launch: {e}")))?;
        self.enc
            .as_mut()
            .expect("scratch")
            .graphs
            .insert(b, SendGraph(graph));
        Ok(false)
    }

    /// How many pending windows one admission pass may batch.
    pub fn enc_batch_cap(&self) -> usize {
        self.enc.as_ref().map_or_else(enc_batch_env, |s| s.bmax)
    }

    /// Encode one window into a buffer of its own - the probe example and the
    /// oracle gate, which want the states to outlive the next encode. Serving
    /// never calls this: it runs the whole admission tick as one graph
    /// (`encode_into`).
    pub fn encode(&mut self, mel: &MelFeatures) -> Result<EncoderOutput, GpuModelError> {
        let n = self.upload_window(mel)?;
        self.encode_body()?;
        let d = self.hp.d_model;
        let exec = self.exec.clone();
        // Debug tap for the encoder-output oracle gate (the pattern):
        // one [n, d_model] f32 block appended per encode call, compared
        // against transformers out of tree
        if let Ok(path) = paddock_models::dev_var!("PADDOCK_WHISPER_DUMP_ENC") {
            let sc = self.enc.as_ref().expect("scratch");
            let host = exec.to_host_len(&sc.norm, n * d)?;
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| GpuError::Driver(format!("encoder dump {path}: {e}")))?;
            let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
            f.write_all(&bytes)
                .map_err(|e| GpuError::Driver(format!("encoder dump {path}: {e}")))?;
            tracing::info!(frames = n, path, "dumped whisper encoder states");
        }
        let mut states = exec.alloc(n * d)?;
        let sc = self.enc.as_ref().expect("scratch");
        exec.copy_slice(&sc.norm, 0, n * d, &mut states)?;
        Ok(EncoderOutput {
            states,
            n_frames: n,
        })
    }
}
