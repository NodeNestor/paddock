//! Word timing, the device half: the teacher-forced pass that produces the
//! cross-attention `align` turns into times.
//!
//! `align` is pure and knows nothing about a GPU; this is everything it needs
//! fed to it. The split is deliberate - the maths is what can be wrong in
//! subtle ways and it is tested against hand-worked cases, while this file is
//! plumbing that either runs or does not.
//!
//! Why A SECOND pass at ALL. The decode already computed exactly this
//! attention, once per token, and threw it away. Keeping it would mean every
//! transcription materialising an attention row per alignment head per step -
//! and a second captured graph variant to write it - to serve a feature most
//! requests never ask for. So word timing re-runs the decode instead, with the
//! tokens it already produced fed back in (teacher forcing), which is also what
//! both references do.
//!
//! AND the RE-RUN is A PREFILL, not A DECODE. Teacher forcing knows every input
//! up front, so the whole sequence can go through in CHUNKS of rows rather than
//! one token at a time - which matters because whisper decode at one row is
//! purely weight-bandwidth-bound (measured: 4.7 ms a step, and the model is
//! 3.1 GB at f16 on a ~700 GB/s card), so a chunk of rows costs barely more
//! than one row. Measured on a 145-token window: 959 ms token-at-a-time,
//! 120 ms at 32 rows a chunk. Both references do the same thing - one forward
//! pass over the full sequence. The chunk is bounded rather than being the
//! whole sequence because cross-attention re-reads the encoder planes per ROW;
//! see `CHUNK_ROWS`.
//!
//! Causality across a chunk is not an extra mechanism: `whisper_qkv_split`
//! writes every row's K/V into the slot at its own position before
//! `whisper_dec_attn` runs, and the attention reads keys `0..=pos` per row - so
//! a row never sees a later row's key even though it is already in the cache.
//!
//! The ROW OFF-BY-ONE, which is the part that is easy to get wrong. The
//! attention at input position *i* is the model reading the audio in order to
//! emit token *i+1*. So the row captured while feeding `<|notimestamps|>` is
//! the boundary before the first text token, the row captured while feeding
//! text token *j* is the boundary before token *j+1*, and the row for the last
//! text token is the end of the transcript. `n` text tokens therefore give
//! `n+1` BOUNDARIES, not `n` starts, and token *j* spans `[b[j], b[j+1])`.
//! Both OpenAI (`timing.py`, slicing `[len(sot_sequence) : -1]`) and
//! whisper.cpp (`whisper_exp_compute_token_level_timestamps_dtw`, dropping
//! `sot_sequence_length` from the front and one from the back) land on exactly
//! this row set; they differ only in whether the task token is in the prompt at
//! all, and we keep it because it is in the prompt the transcript was actually
//! decoded under.
//!
//! Because the trailing `<|endoftext|>` row is discarded by both references, we
//! never feed it - the sequence run here is precisely the decode's own prompt
//! plus its own text tokens.

use cudarc::driver::CudaSlice;

use crate::gpu::{GpuError, GpuExecutor, KvDtype};
use crate::gpu_model::gpt_oss::GpuModelError;

use super::GpuWhisper;
use super::align;

/// Samples of 16 kHz audio per encoder frame: whisper's 160-sample hop, halved
/// again by the conv stem's stride-2. The 0.02 s `align::SECONDS_PER_FRAME`
/// prices, expressed in samples.
const SAMPLES_PER_FRAME: usize = 320;

/// Median filter width over the time axis. OpenAI's `medfilt_width`, and
/// whisper.cpp's default - 7 frames is 140 ms, wide enough to kill the
/// single-frame spikes attention is full of and narrow enough to leave a real
/// word standing. Note it also ERASES any band under 4 frames wide (a value
/// survives only where it holds 4 of the 7), which is a real floor on how short
/// a token this can time.
const MEDFILT_WIDTH: usize = 7;

/// Rows per teacher-forced chunk.
///
/// The weight stream is shared by every row in a chunk, so widening is nearly
/// free - until cross-attention, which re-reads the window's encoder planes
/// once per ROW and stops being free. Swept on a 145-token 30 s window,
/// A6000, f16 KV (capture only, ms): 8 -> 235, 16 -> 165, 32 -> 120,
/// 64 -> 100, 128 -> 92.
///
/// 32 and not 128 because the row scratch is resident once it widens: ~14 MB
/// at 32 rows against ~55 MB at 128 (mostly the logits plane), for a 28 ms
/// difference on a whole window. On a product whose scarce resource is memory
/// that is the wrong way round.
const CHUNK_ROWS: usize = 32;

/// How many encoder frames actually cover audio, for a window holding
/// `audio_samples` samples.
///
/// The window is always padded to 30 s, so the encoder always produces its full
/// `n_enc` frames - but the tail of a short clip is padding, and letting the DTW
/// path wander into it lets the last token claim silence it was never spoken
/// over. Both references clip here for the same reason.
///
/// Rounds up: a partial frame still holds audio.
pub(crate) fn used_frames(audio_samples: usize, n_enc: usize) -> usize {
    audio_samples.div_ceil(SAMPLES_PER_FRAME).clamp(1, n_enc)
}

/// The alignment-head capture buffers for one teacher-forced pass.
///
/// Allocated per call rather than pooled: this runs off the latency path, after
/// a window's decode has finished, and sizing it to the actual token count keeps
/// a typical 30 s window at a few MB instead of the ~27 MB a worst-case
/// resident buffer would hold forever.
pub(crate) struct XattnDump {
    /// One entry per decoder layer, `None` for a layer no alignment head lives
    /// in - indexing by layer keeps the decode loop's per-layer test to one
    /// array read.
    by_layer: Vec<Option<LayerDump>>,
    /// how many rows the chunk in flight carries, and which capture row it
    /// starts at; set by the pass before each chunk
    rows: usize,
    row0: usize,
    n_used: usize,
}

/// One alignment layer's landing.
struct LayerDump {
    /// which heads of this layer to dump, device-side for the kernel
    ids: CudaSlice<u32>,
    /// `[n_rows, ids, n_used]` - this layer's whole capture, written straight
    /// by the kernel at the chunk's row offset. Per LAYER and not one shared
    /// plane because the kernel's own output is `[row][head][frame]` for the
    /// heads it was handed, so a layer's chunk is contiguous exactly here and
    /// nowhere else. The regrouping into `align`'s `[head][row][frame]` is the
    /// host transpose at the end, over a few MB, once.
    acc: CudaSlice<f32>,
}

impl XattnDump {
    /// Dump layer `li`'s alignment heads for the chunk in flight, if it has
    /// any. Called from inside the decode loop with the cross-attention query
    /// the layer is about to attend with.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn capture(
        &mut self,
        exec: &GpuExecutor,
        li: usize,
        q: &CudaSlice<f32>,
        qbias: Option<&CudaSlice<f32>>,
        k: &CudaSlice<u8>,
        slots: &CudaSlice<u32>,
        kv_stride: usize,
        n_heads: usize,
        head_dim: usize,
        scale: f32,
        kv: KvDtype,
    ) -> Result<(), GpuError> {
        let (n_used, rows, row0) = (self.n_used, self.rows, self.row0);
        let Some(l) = self.by_layer[li].as_mut() else {
            return Ok(());
        };
        let n = l.ids.len();
        // `n_used` and not the full plane: the softmax has to be over the
        // frames that survive the clip, not over 30 s of padding it then gets
        // truncated out of. Renormalising after the fact is not the same
        // number, because the per-frame normalisation that follows does not
        // commute with a per-token rescale.
        exec.whisper_xattn_probs(
            q,
            qbias,
            k,
            slots,
            &l.ids,
            &mut l.acc,
            row0 * n * n_used,
            kv_stride,
            n_used,
            n_heads,
            head_dim,
            n,
            rows,
            scale,
            kv,
        )
    }
}

impl GpuWhisper {
    /// Recover the timing of one window's transcript: boundary times, in
    /// seconds from the START of the WINDOW, for the text tokens given.
    ///
    /// `text_tokens` must be text only - no timestamp tokens, no `<|eot|>`, no
    /// prompt tokens. That is a hard requirement rather than a filter applied
    /// here deliberately: the caller groups these same tokens into words, and a
    /// silent filter would let the two disagree about which token is which.
    ///
    /// Returns `n + 1` boundaries for `n` tokens (see the module note on the
    /// row off-by-one) - token `j` spans `[out[j], out[j+1])` - or an empty vec
    /// for an empty transcript.
    ///
    /// PRECONDITION: `slot`'s cross-attention planes still hold this window.
    /// The pass re-decodes into the slot, so it also CLOBBERS that slot's
    /// self-attention KV - which is harmless once the window is transcribed,
    /// and is why this runs after the window rather than alongside it.
    pub fn token_boundaries(
        &mut self,
        slot: usize,
        lang_tok: u32,
        text_tokens: &[u32],
        audio_samples: usize,
    ) -> Result<Vec<f32>, GpuModelError> {
        if text_tokens.is_empty() {
            return Ok(Vec::new());
        }
        let (n_enc, ctx) = (self.hp.n_audio_ctx, self.hp.n_text_ctx);
        let (n_heads, hd) = (self.hp.n_dec_heads, self.hp.head_dim);
        let n_layer = self.dec_layers.len();
        // the dump kernel reads `q` as `[row, head, hd]` off the same buffer the
        // decode's cross-attention treats as `[row, d]` - that only lines up
        // while the head geometry tiles d exactly
        debug_assert_eq!(
            self.hp.d_model,
            n_heads * hd,
            "whisper: head geometry disagrees with d_model"
        );
        if self.decode.as_ref().is_none_or(|b| slot >= b.cap()) {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: word timing on slot {slot} outside the decode pool"
            )));
        }
        if let Some(&t) = text_tokens.iter().find(|&&t| t >= self.tokens.eot) {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: word timing wants text tokens only, got the special token {t}"
            )));
        }

        // sot + language + task + <|notimestamps|>, then the transcript. The
        // canonical alignment prompt: `<|notimestamps|>` is fed even when the
        // request asked for timestamps, because the reference matrices are
        // defined over this sequence and a timestamped decode's own token
        // stream is not comparable to them.
        let (transcribe, no_ts) = self.prompt_tail();
        let mut seq = Vec::with_capacity(4 + text_tokens.len());
        seq.extend_from_slice(&[self.tokens.sot, lang_tok, transcribe, no_ts]);
        seq.extend_from_slice(text_tokens);
        const PREFIX: usize = 3; // sot, language, task - rows nothing consumes
        if seq.len() > ctx {
            return Err(GpuModelError::Unsupported(format!(
                "whisper: word timing needs {} decoder positions, the served context is {ctx}",
                seq.len()
            )));
        }
        let n_rows = seq.len() - PREFIX; // = text_tokens.len() + 1 boundaries

        let heads = align::heads_for(n_layer, n_heads, self.mel.bins, self.hp.n_vocab)
            .unwrap_or_else(|| align::fallback_heads(n_layer, n_heads));
        // Sorted by (layer, head) so a layer's heads are CONTIGUOUS in the
        // global head order - which is what makes the host transpose at the end
        // a straight run of copies rather than a scatter.
        let mut sel = heads.heads.clone();
        sel.sort_unstable();
        let n_sel = sel.len();
        let n_used = used_frames(audio_samples, n_enc);
        tracing::debug!(
            heads = n_sel,
            source = heads.source,
            tokens = text_tokens.len(),
            frames = n_used,
            // "attempting", not "capturing": this line is emitted before the
            // pack call it precedes, so on a build whose pack lacked
            // `whisper_xattn_probs` the log showed the timing pass working on
            // a runner where it could not. A line that announces
            // an attempt must not read as a result.
            "whisper word timing: attempting cross-attention capture"
        );

        let exec = self.exec.clone();
        let mut by_layer: Vec<Option<LayerDump>> = (0..n_layer).map(|_| None).collect();
        // global head order -> (layer, index within the layer's dump), the map
        // the host transpose reads back through
        let mut where_head: Vec<(usize, usize)> = Vec::with_capacity(n_sel);
        for (li, cell) in by_layer.iter_mut().enumerate() {
            let ids: Vec<u32> = sel
                .iter()
                .filter(|&&(l, _)| l == li)
                .map(|&(_, h)| h as u32)
                .collect();
            if ids.is_empty() {
                continue;
            }
            where_head.extend((0..ids.len()).map(|i| (li, i)));
            *cell = Some(LayerDump {
                acc: exec.alloc(n_rows * ids.len() * n_used)?,
                ids: exec.to_device_u32(&ids)?,
            });
        }
        let mut dump = XattnDump {
            by_layer,
            rows: 0,
            row0: 0,
            n_used,
        };

        let t_capture = std::time::Instant::now();
        // The pool's tick has to be able to carry a whole chunk of rows; only
        // the row scratch widens, never the slot planes.
        let chunk = CHUNK_ROWS.min(n_rows);
        self.ensure_decode_rows(chunk.max(PREFIX))?;
        // The prompt prefix produces rows nothing consumes, so it goes through
        // in one undumped pass.
        let slots_v = vec![slot as u32; chunk.max(PREFIX)];
        {
            let pos: Vec<u32> = (0..PREFIX as u32).collect();
            self.decode.as_mut().expect("checked above").feed_rows(
                &exec,
                &slots_v[..PREFIX],
                &seq[..PREFIX],
                &pos,
            )?;
            self.step_body(PREFIX, false, None)?;
        }
        for p0 in (PREFIX..seq.len()).step_by(chunk) {
            let p1 = (p0 + chunk).min(seq.len());
            let pos: Vec<u32> = (p0 as u32..p1 as u32).collect();
            self.decode.as_mut().expect("checked above").feed_rows(
                &exec,
                &slots_v[..p1 - p0],
                &seq[p0..p1],
                &pos,
            )?;
            dump.rows = p1 - p0;
            dump.row0 = p0 - PREFIX;
            // Eager deliberately. A graph would only pay back over many replays
            // of the same shape, and a whole window is a handful of chunks -
            // the last one a different width again.
            self.step_body(p1 - p0, false, Some(&mut dump))?;
        }

        // per-layer `[row][head-in-layer][frame]` -> align's `[head][row][frame]`
        let mut w = vec![0.0f32; n_sel * n_rows * n_used];
        for (g, &(li, hi)) in where_head.iter().enumerate() {
            let l = dump.by_layer[li]
                .as_ref()
                .expect("head map points at a dumped layer");
            let nh = l.ids.len();
            let flat = exec.to_host_len(&l.acc, n_rows * nh * n_used)?;
            for r in 0..n_rows {
                let src = (r * nh + hi) * n_used;
                let dst = (g * n_rows + r) * n_used;
                w[dst..dst + n_used].copy_from_slice(&flat[src..src + n_used]);
            }
        }
        let capture_ms = t_capture.elapsed().as_secs_f32() * 1e3;
        let t_dtw = std::time::Instant::now();

        align::normalize_over_tokens(&mut w, n_sel, n_rows, n_used);
        align::median_filter_rows(&mut w, n_sel * n_rows, n_used, MEDFILT_WIDTH);
        let cost = align::mean_heads_negated(&w, n_sel, n_rows, n_used);
        let path = align::dtw_path(&cost, n_rows, n_used);
        let frames = align::token_start_frames(&path, n_rows);
        // Split because the two halves scale differently and the fix for each
        // is different: the capture is chunked GPU decode, the post-pass is
        // O(heads * tokens * frames) on the host.
        tracing::debug!(
            capture_ms,
            dtw_ms = t_dtw.elapsed().as_secs_f32() * 1e3,
            "whisper word timing done"
        );
        Ok(frames
            .iter()
            .map(|&f| f as f32 * align::SECONDS_PER_FRAME)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn used_frames_rounds_up_and_clamps() {
        // 1 s of 16 kHz audio is 50 encoder frames
        assert_eq!(used_frames(16_000, 1500), 50);
        // a partial frame still holds audio, so it counts
        assert_eq!(used_frames(16_001, 1500), 51);
        // a full 30 s window saturates and never overruns the plane
        assert_eq!(used_frames(480_000, 1500), 1500);
        assert_eq!(used_frames(999_999, 1500), 1500);
        // silence still has to leave the DTW somewhere to go
        assert_eq!(used_frames(0, 1500), 1);
    }
}
