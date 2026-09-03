//! Forced-alignment service seam  - the encoder.rs shape: a
//! dedicated CUDA thread owning the model, oneshot request/response, no
//! decode loop, no streaming. One request is one tower encode plus one causal
//! prefill reading the head at the `<timestamp>` rows; the reply is the raw
//! argmax time-bin per row. The runner owns everything linguistic - word
//! splitting, tokenization, the LIS monotonicity repair, ms conversion - so
//! this thread stays a pure compute seam.
//!
//! Deliberately no cross-request coalescing (encoder.rs's whole second half):
//! alignment is an enrichment pass that runs once per settled utterance, not
//! a throughput surface. If a batch shape ever matters here, the encoder's
//! submit/collect pipeline is the pattern to lift.

use std::sync::mpsc::{Sender, channel};

use tokio::sync::oneshot;

use crate::audio::MelFeatures;
use crate::gpu_model::qwen3_asr::GpuQwen3Asr;

/// One alignment request, fully packed by the runner: the token sequence with
/// the audio-pad run and the interleaved `<timestamp>` slots already in
/// place.
pub struct AlignReq {
    pub ids: Vec<u32>,
    pub mel: MelFeatures,
    /// row index where the audio-pad run starts
    pub splice_at: usize,
    /// pad-run length - must equal the tower's token count for this clip
    pub n_audio: usize,
    /// ascending row indices of every `<timestamp>` token
    pub ts_rows: Vec<usize>,
}

enum Job {
    Run(AlignReq, oneshot::Sender<Result<Vec<u32>, String>>),
}

/// Handle to the aligner thread. Cloneable; jobs queue FIFO.
#[derive(Clone)]
pub struct Aligner {
    tx: Sender<Job>,
}

impl Aligner {
    /// Spawn the aligner thread. `build` constructs the model on that thread
    /// (CUDA context binding) and may fail; spawn blocks until build finishes
    /// and propagates the error - same contract as `Encoder::spawn`.
    pub fn spawn<F>(build: F) -> Result<Self, String>
    where
        F: FnOnce() -> Result<GpuQwen3Asr, String> + Send + 'static,
    {
        let (tx, rx) = channel::<Job>();
        let (ready_tx, ready_rx) = channel::<Result<(), String>>();
        std::thread::Builder::new()
            .name("paddock-aligner".into())
            .spawn(move || {
                let mut model = match build() {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(()));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                while let Ok(Job::Run(req, reply)) = rx.recv() {
                    let r = model
                        .align_bins(&req.ids, &req.mel, req.splice_at, req.n_audio, &req.ts_rows)
                        .map_err(|e| e.to_string());
                    let _ = reply.send(r);
                }
            })
            .map_err(|e| e.to_string())?;
        ready_rx.recv().map_err(|e| e.to_string())??;
        Ok(Self { tx })
    }

    /// Run one alignment; resolves with the argmax time-bin per `<timestamp>`
    /// row, in `ts_rows` order.
    pub async fn align(&self, req: AlignReq) -> Result<Vec<u32>, String> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(Job::Run(req, tx))
            .map_err(|_| "aligner thread gone".to_string())?;
        rx.await
            .map_err(|_| "aligner dropped the request".to_string())?
    }
}
