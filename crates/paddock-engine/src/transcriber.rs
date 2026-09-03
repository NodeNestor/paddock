//! Transcription serving seam: a dedicated thread owning a `GpuWhisper` and
//! scheduling many transcriptions across its decode slots.
//!
//! Mirrors `Encoder`'s thread-owns-the-CUDA-context shape - whisper is an
//! encoder-DECODER, which does not fit the `Generator` trait the decoder-only
//! families share: there is no prompt KV to page, no continuous batch to join
//! in the LLM sense, and the audio is the whole input.
//!
//! SCHEDULING (the bring-up lane was one clip at a time, FIFO,
//! which pinned throughput flat at ~2.1 req/s from c1 to c32 while latency
//! grew linearly - the queue was the engine). The unit of work here is a
//! 30 s WINDOW, not a request: whisper's encoder is fixed-size, so a long
//! clip is several windows, and windows are what occupy slots. Each loop
//! turn:
//!
//!   1. drain newly arrived jobs (blocking only when nothing is in flight),
//!   2. fill every free slot by ENCODING one admittable window into it,
//!   3. run one decode step across every active slot at once,
//!   4. retire slots that hit `<|endoftext|>` (or a limit) - running the word
//!      timing pass first, while the slot still holds that window's encoder
//!      planes - and answer any request whose windows are all in.
//!
//! Admission has exactly one ordering constraint, and it is a correctness
//! one: when the caller did not force a language, window 0 must resolve it
//! before its siblings start, because whisper detects per window and a
//! mid-clip flip is a transcript bug, not multilingual support. With a
//! forced language - what every benchmark and most callers do - all of a
//! clip's windows are independent and admit together.
//!
//! The mel frontend is not here: jobs arrive with their windows already
//! transformed, computed on the caller's blocking pool. Host DSP on this
//! thread would serialize every concurrent transcription behind each other's
//! frontend, which is the same defect the Qwen3-ASR path already fixed.

use std::collections::HashMap;
use std::sync::mpsc::{Sender, channel};

use tokio::sync::oneshot;

use crate::audio::MelFeatures;
use crate::audio::guards::{self, Repetition, Stop};
use crate::gpu_model::whisper::{self, GpuWhisper, LangProb, TimeScale};

/// How much probability mass a caller's candidate set carries, as a whole,
/// before any audio is heard.
///
/// A prior, not a filter, and the difference is the whole design (
/// the set biases the
/// posterior, it does not forbid anything, so a language nobody named can
/// still win if the audio says so. That is what separates this from Azure's
/// LID, which "returns one of the candidate languages provided even if those
/// languages weren't in the audio", and from Deepgram's, which silently drops
/// speech in a language you did not name.
///
/// 0.5 = half the mass on what the caller said, half spread over everything
/// else. The arithmetic that number buys, on a 99-language checkpoint with
/// two hinted languages: an out-of-set language needs to be ~48x more likely
/// than an in-set one to still win ((0.5/2) / (0.5/97)). Enough to fix the
/// failure this was written for - Qwen3-ASR heard Swedish and wrote fluent
/// German - and nowhere near enough to hide a clear out-of-set answer.
///
/// PROVISIONAL. It is the one constant here with no measurement behind it,
/// is explicit that no threshold exists to inherit
/// (faster-whisper's 0.5 has a mechanical PR rationale and no accuracy
/// study). Stage B measures the calibration curve and this moves; until then
/// it rides on the wire with every answer it shaped, so it is never a number
/// only the server knows.
pub const DEFAULT_LANGUAGE_PRIOR: f32 = 0.5;

/// The prior can be strong but never absolute - at 1.0 the "soft hint" would
/// be a hard filter wearing a different name, and a filter cannot report an
/// out-of-set language honestly because it has already zeroed it.
const MAX_LANGUAGE_PRIOR: f32 = 0.99;

/// What the caller said about the language of the audio.
///
/// Three states, and they are genuinely different questions: a FORCED code is
/// an instruction (no detection runs at all), HINTS are a prior over
/// detection, and neither is "let the model decide with nothing to go on".
#[derive(Default, Clone, Debug)]
pub struct LanguageAsk {
    /// bare code ("sv"). Set = the decode uses exactly this and the detector
    /// never runs.
    pub forced: Option<String>,
    /// The caller's candidate languages - OpenAI's own `languages` array.
    /// Ignored when `forced` is set (an instruction outranks a hint).
    pub hints: Vec<String>,
    /// Mass on the hinted set as a whole; 0 disables the prior entirely.
    pub strength: f32,
}

impl LanguageAsk {
    /// A forced language and nothing else - what every caller sending plain
    /// `language=sv` asks for.
    pub fn forced(code: Option<String>) -> Self {
        Self {
            forced: code,
            hints: Vec::new(),
            strength: 0.0,
        }
    }

    /// Fold the candidate set into a posterior.
    ///
    /// Bayes, spelled out: `p'(l) ∝ p(l) · w(l)` with the hinted languages
    /// sharing `strength` and everything else sharing the remainder. Returns
    /// the re-sorted distribution and, when the prior actually CHANGED the
    /// winner, what the audio alone preferred - because a hint that quietly
    /// overturns the model's own answer is exactly the kind of invisible
    /// decision this whole feature exists to remove.
    pub fn apply(&self, mut post: Vec<LangProb>) -> (Vec<LangProb>, Option<LangProb>) {
        let strength = self.strength.clamp(0.0, MAX_LANGUAGE_PRIOR);
        let n = post.len();
        let hinted = |c: &str| self.hints.iter().any(|h| h == c);
        let k = post.iter().filter(|e| hinted(&e.code)).count();
        // Nothing to say: no hints, no strength, hints nobody here can serve,
        // or hints covering the whole map. Each leaves the posterior as the
        // audio left it, which is the right answer rather than a no-op to
        // apologise for.
        if strength <= 0.0 || k == 0 || k == n {
            return (post, None);
        }
        let unbiased = post.first().cloned();
        let inside = strength / k as f32;
        let outside = (1.0 - strength) / (n - k) as f32;
        let mut z = 0.0f32;
        for e in &mut post {
            e.p *= if hinted(&e.code) { inside } else { outside };
            z += e.p;
        }
        if z > 0.0 {
            for e in &mut post {
                e.p /= z;
            }
        }
        post.sort_by(|a, b| b.p.total_cmp(&a.p));
        let moved = unbiased.filter(|u| post.first().is_some_and(|t| t.code != u.code));
        (post, moved)
    }
}

/// What the loaded checkpoint can tell a caller before any request runs.
///
/// Read once on the transcriber thread at startup, because the model never
/// leaves it - and because a client should learn which languages this
/// checkpoint knows from `/v1/models`, not from a 400 (or worse, from a
/// transcript in a language it was never able to produce).
pub struct AsrCard {
    pub time_scale: TimeScale,
    /// bare language codes, in the checkpoint's own order
    pub languages: Vec<String>,
}

/// One 30 s window's decode: the raw token ids (the caller owns
/// detokenization - it has the tokenizer), what the model thought of each,
/// and whether it heard speech at all.
pub struct Window {
    pub tokens: Vec<u32>,
    /// parallel to `tokens` - the chosen token's own log-probability
    pub logprobs: Vec<f32>,
    /// parallel to `tokens` - what the model nearly picked instead, as
    /// `(id, log p)`. The gap to `logprobs` is the margin, which is what
    /// separates a genuine two-way call from a merely diffuse one.
    pub runners: Vec<Option<(u32, f32)>>,
    /// `<|nospeech|>`'s probability at the window's first decode step, which
    /// is where OpenAI defines it. High on a window of silence or music.
    pub no_speech_prob: f32,
    /// Word-timing boundaries, seconds from the start of the CLIP, when the
    /// job asked for them - otherwise empty.
    ///
    /// One more than the window's text tokens: entry `j` is the boundary
    /// before text token `j`, so token `j` spans `[b[j], b[j+1])` (see
    /// `whisper::timing` for why the count is n+1). "Text tokens" means
    /// `tokens` with the timestamp ids filtered out, in order - the same
    /// filter every consumer of `tokens` already applies.
    pub boundaries: Vec<f32>,
    /// Why the decode ended. Anything but `Eot` is a decode that
    /// was stopped from outside and has to reach the caller.
    pub stop: Stop,
    /// Mean logprob over the window's generated tokens - the figure the
    /// silence rule tests, kept here because `logprobs` is CLEARED when that
    /// rule fires and the number is what explains why.
    pub avg_logprob: f32,
    /// Tail entropy at the end of the decode, in nats; `f32::INFINITY` on a
    /// window too short to judge (see `guards::Repetition::value`).
    pub entropy: f32,
    /// The silence rule fired: this window holds no speech, `tokens` has been
    /// emptied, and it contributes nothing to the transcript.
    ///
    /// The tokens are dropped here rather than left for each consumer to
    /// filter, because there are four of them (the file lane, its stream, the
    /// live socket, word timing) and a suppression that three of them honour
    /// is a suppression that does not exist.
    pub suppressed: bool,
}

impl Default for Window {
    fn default() -> Self {
        Self {
            tokens: Vec::new(),
            logprobs: Vec::new(),
            runners: Vec::new(),
            no_speech_prob: 0.0,
            boundaries: Vec::new(),
            stop: Stop::default(),
            avg_logprob: 0.0,
            // "nothing decoded yet" is not "maximally repetitive"
            entropy: f32::INFINITY,
            suppressed: false,
        }
    }
}

/// One finished transcription: the resolved language plus a decode per 30 s
/// window, in clip order.
pub struct Transcript {
    pub language: String,
    pub windows: Vec<Window>,
    /// whether the decode ran without `<|notimestamps|>` - i.e. whether the
    /// token stream can be expected to carry timestamp tokens at all
    pub timestamps: bool,
    /// The language posterior the decode was settled from, best first - the
    /// whole distribution, after any candidate-set prior.
    ///
    /// Empty when the caller forced a language, and that absence is the
    /// honest answer rather than a gap: with a forced code no detection runs
    /// at all, so there is no distribution to report and a fabricated one
    /// would read as a measurement.
    pub language_probs: Vec<LangProb>,
    /// What the audio alone preferred, present only when the caller's
    /// candidate set outranked it. A hint that changes the answer has to say
    /// so - see `LanguageAsk::apply`.
    pub language_prior_moved: Option<LangProb>,
}

/// Decode progress, for a caller that wants the transcript as it ARRIVES
/// rather than at the end.
///
/// The ordering is the whole subtlety, and it is inherent rather than a
/// choice: windows occupy slots in PARALLEL, so window 3 routinely finishes
/// before window 1 and these events interleave across `window`. Reassembling
/// them into clip order is the consumer's job - this thread deliberately does
/// not buffer, because holding window 3 until window 1 lands would mean
/// keeping a slot's worth of tokens here for the one caller that wanted them
/// early.
pub enum Progress {
    /// The clip's language, sent once and before any token: either the
    /// caller's forced code or window 0's detection.
    Language(String),
    /// A window's decode has started, carrying the one thing that is known
    /// about it before a single token exists: how sure the model is that it
    /// holds no speech.
    ///
    /// It is here for the streaming consumer's sake. The silence
    /// rule needs the finished decode's average logprob, so a window can only
    /// be SUPPRESSED at the end - and a delta stream cannot take bytes back.
    /// This is the early warning: above `guards::NO_SPEECH_THOLD` a window
    /// might yet be dropped, so its tokens must not be shown until it closes.
    /// Below it, suppression is arithmetically impossible (the rule is an AND)
    /// and the window streams live, which is every window of ordinary audio.
    WindowOpen { window: usize, no_speech_prob: f32 },
    /// One decoded token, in the window that produced it.
    Token {
        window: usize,
        id: u32,
        logprob: f32,
    },
    /// Nothing more will arrive for that window (`<|endoftext|>`, the token
    /// budget, the context ceiling, or a guard). `suppressed` means the
    /// silence rule fired and everything this window streamed must be
    /// discarded rather than joined into the transcript.
    WindowDone { window: usize, suppressed: bool },
}

pub struct TranscribeJob {
    /// One entry per 30 s window, in clip order, mel-transformed by the
    /// caller (see the module note on where the frontend runs).
    pub windows: Vec<MelFeatures>,
    /// Parallel to `windows`: false = the caller's VAD found no speech in that
    /// span, so do not encode or decode it at all. Empty means
    /// "every window has speech", which is what an ungated caller sends and
    /// what keeps this an addition rather than a behaviour change.
    ///
    /// The gate lives out there, not here, because the samples do - this
    /// thread receives mel blocks and a mel block cannot say how loud the room
    /// was. It still costs the caller its frontend pass; skipping the MEL too
    /// is a further saving nobody has needed yet.
    pub speech: Vec<bool>,
    /// What the caller said about the language: a forced code, a soft
    /// candidate set, or nothing. `LanguageAsk::default()` runs whisper's own
    /// detection over the unbiased posterior.
    pub language: LanguageAsk,
    /// Context tokens for the `<|startofprev|>` pre-roll - the API's `prompt`,
    /// already tokenized by the caller (the tokenizer lives out there with the
    /// detokenizer). TEXT tokens only: the marker is this thread's to add, and
    /// whisper's own truncation rule is applied here too.
    ///
    /// It goes in front of every window of the clip, not just the first. The
    /// field is a style and vocabulary hint - names, an acronym list, the
    /// spelling the caller wants - and those are as true of window 5 as of
    /// window 0. Conditioning each window on the previous window's text is a
    /// different feature (whisper's `condition_on_previous_text`), and it
    /// would serialize windows that currently decode side by side.
    pub prompt: Vec<u32>,
    /// Ask the model to emit its timestamp tokens by dropping
    /// `<|notimestamps|>` from the prompt. OPT-IN, because it changes the
    /// prompt and therefore the decode: a caller who only wants text keeps
    /// the prompt every WER gate was measured on.
    pub timestamps: bool,
    /// Recover per-token times from cross-attention. INDEPENDENT of
    /// `timestamps`: the two are different mechanisms reading the same clip, and
    /// word timing needs no grammar - it re-runs each window under the canonical
    /// `<|notimestamps|>` alignment prompt whatever the transcript was decoded
    /// under.
    ///
    /// Its own opt-in because it costs a second forward pass per window
    /// (~170 ms on a full 30 s window, A6000) - the latency OpenAI's own docs
    /// warn about for this granularity.
    pub words: bool,
    pub max_tokens: usize,
    /// Where to mirror decode progress, for `stream=true`. Absent on the
    /// ordinary path, and a closed receiver is not an error - a client that
    /// hung up mid-stream must not abort a decode the slots are already
    /// paying for.
    pub progress: Option<tokio::sync::mpsc::UnboundedSender<Progress>>,
    pub reply: oneshot::Sender<Result<Transcript, String>>,
}

/// Handle to a whisper model running on its own CUDA thread.
#[derive(Clone)]
pub struct Transcriber {
    tx: Sender<TranscribeJob>,
}

/// One in-flight request.
struct Req {
    windows: Vec<MelFeatures>,
    /// see `TranscribeJob::speech`; empty = every window has speech
    speech: Vec<bool>,
    /// resolved language: the forced code up front, or window 0's detection
    lang: Option<String>,
    /// what the caller said about the language, kept for the detection pass
    ask: LanguageAsk,
    /// window 0's posterior, after the prior; empty on a forced language
    lang_probs: Vec<LangProb>,
    /// the unbiased top when the prior overturned it (see `LanguageAsk::apply`)
    lang_moved: Option<LangProb>,
    /// the whole pre-roll, marker included, ready to feed: empty when no
    /// context was asked for, `[<|startofprev|>, ...context]` otherwise
    pre: Vec<u32>,
    timestamps: bool,
    words: bool,
    out: Vec<Window>,
    admitted: usize,
    done: usize,
    max_tokens: usize,
    progress: Option<tokio::sync::mpsc::UnboundedSender<Progress>>,
    /// Whether any window of this clip has reached a slot yet.
    ///
    /// The language constraint is "the first window that actually DECODES
    /// settles it", which used to be spelled `admitted == 0`. Under VAD gating
    /// that is wrong: a clip whose lead-in is silence advances `admitted` past
    /// windows nothing ever ran, and every remaining window would then wait
    /// forever for a language no window was allowed to detect.
    decoded_any: bool,
    /// whether `Progress::Language` has gone out - it is sent once, at the
    /// first moment the code is known, which is admission for a forced
    /// language and window 0's first step otherwise
    lang_sent: bool,
    /// A word-timing failure on one window, held until the request is
    /// answered. It cannot fail the request on the spot - the other windows
    /// are still decoding - and it must not be dropped either: the caller
    /// ASKED for word times, so answering with a transcript whose words have
    /// no times is the silent-failure shape this endpoint refuses.
    err: Option<String>,
    reply: Option<oneshot::Sender<Result<Transcript, String>>>,
}

impl Req {
    /// Does window `w` hold speech? Always true for a caller that sent no
    /// gate, which is the ungated default.
    fn has_speech(&self, w: usize) -> bool {
        self.speech.get(w).copied().unwrap_or(true)
    }

    fn emit(&self, p: Progress) {
        if let Some(tx) = &self.progress {
            let _ = tx.send(p);
        }
    }

    /// Announce the language the first time it is known. Called from both
    /// places that can settle it so neither has to know about the other.
    fn announce_language(&mut self) {
        if self.lang_sent || self.progress.is_none() {
            return;
        }
        if let Some(code) = self.lang.clone() {
            self.lang_sent = true;
            self.emit(Progress::Language(code));
        }
    }
}

/// One window occupying a decode slot.
struct Run {
    req: u64,
    win: usize,
    slot: usize,
    /// 0/1/2 feed the prompt tail, 3+ generate. Whisper's contract is
    /// `<|startoftranscript|><|lang|><|transcribe|><|notimestamps|>`, and the
    /// sot step's own logits are the language detector. With timestamps asked
    /// for, the prompt STOPS at `<|transcribe|>` - phase 2's own argmax is
    /// then already the window's first emitted token.
    phase: u8,
    /// How much of the context pre-roll is still ahead of this window,
    /// counting the `<|startoftranscript|>` that closes it. Zero on every
    /// request without a `prompt`, and zero from the sot step onward - which
    /// is what `phase == 0` means to everything downstream, so both the
    /// language detection and `no_speech_prob` test it as well.
    pre: usize,
    /// token fed at the next step
    feed: u32,
    /// decoder position the next step lands at (also the steps taken)
    pos: usize,
    /// the repetition guard's rolling tail over this window's tokens
    rep: Repetition,
    out: Window,
    finished: bool,
}

impl Transcriber {
    /// Spawn the transcriber thread. `build` constructs the model on that
    /// thread (CUDA context binding) and may fail; spawn blocks until the
    /// build finishes and propagates the error. `slots` is the decode-slot
    /// ceiling - each one owns a full set of cross-attention planes, so it
    /// is a real VRAM decision, not a queue depth.
    /// Returns the handle and the loaded checkpoint's `AsrCard` - the model
    /// itself never leaves this thread, but its timestamp geometry and its
    /// language map have to reach the caller (one parses segments, the other
    /// publishes and validates `language`), and reading both once at startup
    /// beats a round trip per request.
    /// `metrics`, when given, gets the memory rows of `/api/stats`. An ASR
    /// runner published `"engine": null` before even though the
    /// checkpoint is resident on the card; only the memory rows are filled
    /// (a transcriber's tok/s and phase are not the generative counters).
    pub fn spawn<F>(
        build: F,
        slots: usize,
        metrics: Option<std::sync::Arc<crate::metrics::EngineMetrics>>,
    ) -> Result<(Self, AsrCard), String>
    where
        F: FnOnce() -> Result<GpuWhisper, String> + Send + 'static,
    {
        let (tx, rx) = channel::<TranscribeJob>();
        let (ready_tx, ready_rx) = channel::<Result<AsrCard, String>>();

        std::thread::Builder::new()
            .name("paddock-transcriber".into())
            .spawn(move || {
                let mut model = match build().and_then(|mut m| {
                    m.prepare_batch(slots).map_err(|e| e.to_string())?;
                    Ok(m)
                }) {
                    Ok(m) => {
                        let _ = ready_tx.send(Ok(AsrCard {
                            time_scale: m.time_scale(),
                            languages: m.languages(),
                        }));
                        m
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        return;
                    }
                };
                if let Some(mt) = metrics.as_deref() {
                    use std::sync::atomic::Ordering::Relaxed;
                    // weights_bytes is a per-tensor sum, so prepare_batch's
                    // state above does not contaminate it.
                    mt.weights_mem_bytes.store(model.weights_bytes(), Relaxed);
                    if let Some(b) = model.device_mem_used() {
                        mt.model_mem_bytes.store(b, Relaxed);
                    }
                }
                schedule(&mut model, &rx, slots);
            })
            .map_err(|e| format!("spawn transcriber thread: {e}"))?;

        match ready_rx.recv() {
            Ok(Ok(card)) => Ok((Self { tx }, card)),
            Ok(Err(e)) => Err(e),
            Err(_) => Err("transcriber thread died during startup".into()),
        }
    }

    /// `progress` mirrors the decode as it happens; the returned
    /// `Transcript` is the authoritative result either way, so a caller that
    /// streams still gets one object to build its final answer from.
    #[allow(clippy::too_many_arguments)]
    pub async fn transcribe(
        &self,
        windows: Vec<MelFeatures>,
        speech: Vec<bool>,
        language: LanguageAsk,
        prompt: Vec<u32>,
        timestamps: bool,
        words: bool,
        max_tokens: usize,
        progress: Option<tokio::sync::mpsc::UnboundedSender<Progress>>,
    ) -> Result<Transcript, String> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(TranscribeJob {
                windows,
                speech,
                language,
                prompt,
                timestamps,
                words,
                max_tokens,
                progress,
                reply,
            })
            .map_err(|_| "transcriber thread is gone".to_owned())?;
        rx.await
            .map_err(|_| "transcriber dropped the reply".to_owned())?
    }
}

/// Answer a request and drop it.
fn finish(req: &mut Req, lang: String) {
    if let Some(tx) = req.reply.take() {
        if let Some(e) = req.err.take() {
            let _ = tx.send(Err(e));
            return;
        }
        let windows = std::mem::take(&mut req.out);
        let _ = tx.send(Ok(Transcript {
            language: lang,
            windows,
            timestamps: req.timestamps,
            language_probs: std::mem::take(&mut req.lang_probs),
            language_prior_moved: req.lang_moved.take(),
        }));
    }
}

/// One finished window's word-timing boundaries, in CLIP time.
///
/// Runs while the window's slot is still its slot - see the call site. Word
/// timing is the one thing here that reads the encoder planes after the decode,
/// so it is the one thing that cares when the slot is recycled.
fn word_times(
    model: &mut GpuWhisper,
    run: &Run,
    req: &Req,
    scale: &TimeScale,
) -> Result<Vec<f32>, String> {
    // The alignment prompt is the canonical one, so the language token has to be
    // the language the window was decoded under: an unforced clip settled it at
    // window 0 and every window of the clip shares it (see the module note on
    // why a mid-clip flip is a bug, not multilingual support).
    let code = req.lang.as_deref().unwrap_or_default();
    let lang_tok = model
        .lang_token(code)
        .ok_or_else(|| format!("whisper: language {code:?} is not in this checkpoint's map"))?;
    // TEXT tokens only. `token_boundaries` refuses a special token rather than
    // filtering one, deliberately: the caller groups these same tokens into words,
    // and a filter applied in two places is a filter that can disagree with
    // itself about which token is which.
    let text: Vec<u32> = run
        .out
        .tokens
        .iter()
        .copied()
        .filter(|&t| !scale.is_timestamp(t))
        .collect();
    // The window's real length, which the mel block carries because a whisper
    // feature block is always the padded 30 s and cannot say it otherwise.
    let n_samples = req.windows.get(run.win).map_or(0, |w| w.n_samples);
    let b = model
        .token_boundaries(run.slot, lang_tok, &text, n_samples)
        .map_err(|e| e.to_string())?;
    // window-relative -> clip time, the same offset `split_segments` applies to
    // a timestamp token
    let off = run.win as f32 * scale.window_s;
    Ok(b.iter().map(|t| t + off).collect())
}

/// Fail everything still in flight - a device error is not per-request.
fn fail_all(reqs: &mut HashMap<u64, Req>, active: &mut Vec<Run>, free: &mut Vec<usize>, e: &str) {
    for r in reqs.values_mut() {
        if let Some(tx) = r.reply.take() {
            let _ = tx.send(Err(e.to_owned()));
        }
    }
    reqs.clear();
    for run in active.drain(..) {
        free.push(run.slot);
    }
}

fn schedule(model: &mut GpuWhisper, rx: &std::sync::mpsc::Receiver<TranscribeJob>, slots: usize) {
    let (sot, eot) = model.contract_tokens();
    let (transcribe, no_ts) = model.prompt_tail();
    let sot_prev = model.sot_prev_token();
    let ctx = model.text_ctx();
    let scale = model.time_scale();
    let mut reqs: HashMap<u64, Req> = HashMap::new();
    let mut order: Vec<u64> = Vec::new(); // admission is FIFO across requests
    let mut free: Vec<usize> = (0..slots).rev().collect();
    let mut active: Vec<Run> = Vec::new();
    let mut next_id = 0u64;

    let mut take = |job: TranscribeJob, reqs: &mut HashMap<u64, Req>, order: &mut Vec<u64>| {
        // A job with no windows has nothing to admit and nothing to retire, so
        // parking it in `reqs` would leave it there forever - and now that the
        // intake gate waits on `reqs` being empty, forever means a spin rather
        // than a block. Answer it here instead. The serving path always cuts
        // at least one window, so this is the direct-API guard.
        if job.windows.is_empty() {
            let _ = job.reply.send(Ok(Transcript {
                language: job.language.forced.unwrap_or_default(),
                windows: Vec::new(),
                timestamps: job.timestamps,
                // no window decoded, so nothing detected anything
                language_probs: Vec::new(),
                language_prior_moved: None,
            }));
            return;
        }
        let id = next_id;
        next_id += 1;
        let n = job.windows.len();
        // Whisper's own rule (decoding.py): a context prompt keeps at most
        // half the decoder context minus one, and it is the OLDEST context
        // that goes - a prompt is a tail of previous text, so its end is the
        // part that matters.
        // A context prompt is TEXT. Whisper's control block starts at
        // `<|endoftext|>` and runs to the last timestamp, and a tokenizer will
        // happily turn a literal "<|notimestamps|>" in a caller's prompt into
        // the real thing - which would silently rewrite the decode contract
        // from inside the field that is supposed to be a spelling hint.
        if let Some(&bad) = job.prompt.iter().find(|&&t| t >= eot) {
            let _ = job.reply.send(Err(format!(
                "`prompt` contains whisper control token {bad}; a context prompt is plain text"
            )));
            return;
        }
        let mut pre = Vec::new();
        if !job.prompt.is_empty() {
            let room = (ctx / 2).saturating_sub(1);
            let keep = job.prompt.len().min(room);
            pre.push(sot_prev);
            pre.extend_from_slice(&job.prompt[job.prompt.len() - keep..]);
        }
        let mut req = Req {
            windows: job.windows,
            speech: job.speech,
            lang: job.language.forced.clone(),
            ask: job.language,
            lang_probs: Vec::new(),
            lang_moved: None,
            pre,
            timestamps: job.timestamps,
            words: job.words,
            out: (0..n).map(|_| Window::default()).collect(),
            admitted: 0,
            done: 0,
            max_tokens: job.max_tokens,
            progress: job.progress,
            decoded_any: false,
            lang_sent: false,
            err: None,
            reply: Some(job.reply),
        };
        // a forced language is known before a single step runs, so the stream
        // can say so immediately rather than at window 0's first token
        req.announce_language();
        reqs.insert(id, req);
        order.push(id);
    };

    // Burst accounting: whisper's wall splits into "encode a window" and
    // "step every live slot", and which of the two binds is the whole
    // scheduling question. Totals are emitted when the engine goes idle, so
    // a bench run reports its own split without instrumenting the hot loop.
    let (mut enc_ms, mut step_ms, mut n_enc, mut n_step) = (0f64, 0f64, 0u64, 0u64);
    // Runs whose admission pass is still on the encoder side stream (P38):
    // they join `active` only after `encode_sync`, one tick later - the tick
    // their encode overlaps.
    let mut pending: Vec<Run> = Vec::new();
    loop {
        // ---- 1. intake ----
        // BLOCK only when there is genuinely nothing to do - nothing decoding
        // AND no request still holding an unadmitted window. `active.is_empty()`
        // alone was a deadlock, and a nasty one: a clip longer than 30 s is
        // several windows, so on a ONE-SLOT pool (`--max-batch 1`) window 0
        // retires, `active` empties, and the thread parks on `recv()` with
        // window 1 never admitted. The request then hangs until some UNRELATED
        // request arrives and wakes the loop. Found by a transcript
        // regression run: every clip under 30 s exact, the one
        // 41.7 s clip "timed out" after 600 s. A `reqs` entry only leaves the
        // map when its last window is answered, so a non-empty map with an
        // empty active set always means there is a window waiting.
        if active.is_empty() && reqs.is_empty() {
            if n_step > 0 {
                tracing::debug!(
                    encode_s = enc_ms / 1000.0,
                    windows = n_enc,
                    decode_s = step_ms / 1000.0,
                    steps = n_step,
                    "whisper burst"
                );
                enc_ms = 0.0;
                step_ms = 0.0;
                n_enc = 0;
                n_step = 0;
            }
            match rx.recv() {
                Ok(job) => take(job, &mut reqs, &mut order),
                Err(_) => return, // every handle dropped
            }
        }
        while let Ok(job) = rx.try_recv() {
            take(job, &mut reqs, &mut order);
        }

        // ---- 2a. VAD-gated windows never reach a slot  ----
        //
        // Retired where they stand, in admission order, before the slot loop -
        // which is what makes the gate free rather than merely cheap: no
        // encoder pass, no decode steps, and the slot stays available to a
        // window that has speech in it.
        //
        // In admission order, because the language constraint still has to
        // hold: a clip whose lead-in is silence must settle its language on
        // the first window that actually runs, and draining the head here is
        // what advances `admitted` to find it (see `Req::decoded_any`).
        //
        // Every skipped window is REPORTED. The caller's audio produced
        // nothing over that span because we chose not to look at it, and that
        // is a decision they have to be able to see.
        let mut drained: Vec<u64> = Vec::new();
        for id in order.iter().copied() {
            let Some(req) = reqs.get_mut(&id) else {
                continue;
            };
            while req.admitted < req.windows.len() && !req.has_speech(req.admitted) {
                let w = req.admitted;
                req.admitted += 1;
                req.out[w] = Window {
                    stop: Stop::Vad,
                    ..Default::default()
                };
                req.done += 1;
                req.emit(Progress::WindowDone {
                    window: w,
                    suppressed: true,
                });
            }
            if req.done == req.windows.len() {
                let lang = req.lang.clone().unwrap_or_default();
                finish(req, lang);
                drained.push(id);
            }
        }
        for id in drained {
            reqs.remove(&id);
            order.retain(|&o| o != id);
        }

        // ---- 2a'. merge last tick's admissions (P38): their encode pass
        // overlapped that tick on the side stream; sync and seat them ----
        if !pending.is_empty() {
            if let Err(e) = model.encode_sync() {
                let e = e.to_string();
                fail_all(&mut reqs, &mut active, &mut free, &e);
                order.clear();
                pending.clear();
            } else {
                active.append(&mut pending);
            }
        }

        // ---- 2b. admission: encode pending windows into free slots, up to
        // the encoder's batch cap per pass (- one audio-major
        // encoder pass re-fills the 1-unit-fill GEMM shapes) ----
        let enc_cap = model.enc_batch_cap().max(1);
        while !free.is_empty() {
            // pick up to min(free, cap) admittable windows. A window is
            // admittable when nothing of its clip has decoded yet, or when
            // the clip's language is already settled (see the module note);
            // updating the clip's state between picks preserves that rule
            // inside one batch - an unsettled clip contributes at most its
            // first window.
            let mut picks = Vec::new();
            let mut fi = free.len();
            while picks.len() < enc_cap && fi > 0 {
                let pick = order.iter().copied().find(|id| {
                    reqs.get(id).is_some_and(|r| {
                        r.admitted < r.windows.len() && (!r.decoded_any || r.lang.is_some())
                    })
                });
                let Some(id) = pick else { break };
                let req = reqs.get_mut(&id).expect("picked id exists");
                fi -= 1;
                picks.push((id, req.admitted, free[fi]));
                req.admitted += 1;
                req.decoded_any = true;
            }
            if picks.is_empty() {
                break;
            }
            let slots: Vec<usize> = picks.iter().map(|p| p.2).collect();
            let mels: Vec<_> = picks
                .iter()
                .map(|&(id, win, _)| &reqs.get(&id).expect("picked id exists").windows[win])
                .collect();
            let t0 = std::time::Instant::now();
            let enc = model.encode_into_batch(&slots, &mels);
            enc_ms += t0.elapsed().as_secs_f64() * 1000.0;
            n_enc += picks.len() as u64;
            if let Err(e) = enc {
                let e = e.to_string();
                fail_all(&mut reqs, &mut active, &mut free, &e);
                order.clear();
                break;
            }
            for &(id, win, slot) in &picks {
                let req = reqs.get(&id).expect("picked id exists");
                let pre = req.pre.len();
                let first = req.pre.first().copied().unwrap_or(sot);
                free.pop();
                pending.push(Run {
                    req: id,
                    win,
                    slot,
                    phase: 0,
                    pre,
                    feed: first,
                    pos: 0,
                    // whisper emits plain transcript text (its timestamp
                    // tokens all differ), so both tests apply here
                    rep: Repetition::text(),
                    out: Window::default(),
                    finished: false,
                });
            }
        }
        if !pending.is_empty() && (active.is_empty() || !model.enc_overlap()) {
            // seat the admissions immediately when there is nothing to
            // overlap with, or when overlap is off (killed by env, or the
            // pack cannot route around mmaf - the encode is stream-ordered
            // ahead of this tick either way, so the sync is free)
            if let Err(e) = model.encode_sync() {
                let e = e.to_string();
                fail_all(&mut reqs, &mut active, &mut free, &e);
                order.clear();
                pending.clear();
            } else {
                active.append(&mut pending);
            }
        }
        if active.is_empty() {
            continue;
        }

        // ---- 3. one decode step across every active slot ----
        let ids: Vec<u32> = active.iter().map(|r| r.slot as u32).collect();
        let feeds: Vec<u32> = active.iter().map(|r| r.feed).collect();
        let poss: Vec<u32> = active.iter().map(|r| r.pos as u32).collect();
        // Whisper's timestamp grammar, rebuilt per step from what each row has
        // sampled. Only assembled when something in the batch wants times -
        // otherwise the step keeps its original launch chain and its own
        // captured graph (see `step_batch`).
        //
        // PHASE >= 2 only, and that is not a detail. The filter's opening rule
        // masks everything below `<|0.00|>`, which at the PROMPT steps would
        // mask the language tokens and `<|nospeech|>` along with them - it
        // broke language detection and zeroed every no_speech_prob when this
        // first ran on every step. The reference calls the boundary
        // `sample_begin`: the grammar starts where the model starts choosing,
        // which here is the step that consumes `<|transcribe|>`.
        let on_now = |r: &Run| r.phase >= 2 && reqs.get(&r.req).is_some_and(|q| q.timestamps);
        let rules: Option<Vec<u32>> = active.iter().any(on_now).then(|| {
            active
                .iter()
                .flat_map(|r| whisper::ts_state(&r.out.tokens, &scale, on_now(r)))
                .collect()
        });
        // Overlap route: a non-empty `pending` here means this
        // tick runs while its admissions' encode replay is in flight on the
        // side stream - the step must replay the mmaf-off graph variant
        // (P39: mmaf × tc5p corrupts under true concurrency; every other
        // decode lane measured clean). With overlap gated off `pending` was
        // seated above, so this is false and the step keeps its usual graph.
        model.set_enc_inflight(!pending.is_empty());
        let t0 = std::time::Instant::now();
        let stepped = model.step_batch(&ids, &feeds, &poss, rules.as_deref());
        step_ms += t0.elapsed().as_secs_f64() * 1000.0;
        n_step += 1;
        let step = match stepped {
            Ok(n) => n,
            Err(e) => {
                let e = e.to_string();
                fail_all(&mut reqs, &mut active, &mut free, &e);
                order.clear();
                continue;
            }
        };
        let next = &step.next;

        // language detection needs the whole logits row, so it is its own
        // (rare) pass - only unforced window 0s, once each
        let mut detect_err = None;
        for (b, run) in active.iter().enumerate() {
            // `pre > 0` is a context step, whose logits are about the prompt
            // text, not about the audio - the detector reads the row the
            // `<|startoftranscript|>` step produced and nothing else
            if run.phase != 0 || run.pre > 0 || reqs.get(&run.req).is_none_or(|r| r.lang.is_some())
            {
                continue;
            }
            match model
                .logits_row(b)
                .and_then(|row| model.language_posterior(&row))
            {
                Ok(post) => {
                    if let Some(r) = reqs.get_mut(&run.req) {
                        // The caller's candidate set enters here and nowhere
                        // else: as a prior over the finished posterior, never
                        // as a mask on the logits. Masking would make an
                        // out-of-set language unreportable rather than merely
                        // unlikely, which is the Azure failure (§3.5).
                        let (ranked, moved) = r.ask.apply(post);
                        r.lang = ranked.first().map(|l| l.code.clone());
                        r.lang_probs = ranked;
                        r.lang_moved = moved;
                        r.announce_language();
                    }
                }
                Err(e) => detect_err = Some(e.to_string()),
            }
        }
        if let Some(e) = detect_err {
            fail_all(&mut reqs, &mut active, &mut free, &e);
            order.clear();
            continue;
        }

        // ---- 4. advance each slot ----
        let mut lang_err = None;
        for (b, run) in active.iter_mut().enumerate() {
            run.pos += 1;
            let Some(req) = reqs.get(&run.req) else {
                run.finished = true;
                continue;
            };
            // The context pre-roll runs ahead of everything: its steps feed
            // `<|startofprev|>` and the prompt text, nothing they sample is
            // read, and the window's contract prompt starts once it is spent.
            if run.pre > 0 {
                let k = req.pre.len() - run.pre;
                run.feed = req.pre.get(k + 1).copied().unwrap_or(sot);
                run.pre -= 1;
                continue;
            }
            // Phase 0 is the `<|startoftranscript|>` step, which is where
            // OpenAI defines no_speech_prob - the model has heard the whole
            // window and not yet committed to a first word.
            if run.phase == 0 {
                run.out.no_speech_prob = step.nospeech.get(b).copied().unwrap_or(0.0);
                req.emit(Progress::WindowOpen {
                    window: run.win,
                    no_speech_prob: run.out.no_speech_prob,
                });
            }
            // With timestamps asked for the prompt ends at `<|transcribe|>`,
            // so phase 2's argmax is already the first emitted token and the
            // `<|notimestamps|>` step never happens.
            let generating = run.phase >= 3 || (run.phase == 2 && req.timestamps);
            if generating {
                // stop conditions in the serial lane's order: eot first, then
                // the token budget, then the served context - and now the
                // repetition guard, which is the only one of the four that can
                // fire while the model still has plenty of both left
                let tok = next[b];
                if tok == eot {
                    run.finished = true;
                } else {
                    let lp = step.logprob.get(b).copied().unwrap_or(0.0);
                    run.out.tokens.push(tok);
                    run.out.logprobs.push(lp);
                    run.out
                        .runners
                        .push(step.runner_up.get(b).copied().flatten());
                    req.emit(Progress::Token {
                        window: run.win,
                        id: tok,
                        logprob: lp,
                    });
                    // A looping decode is confident and unbounded: nothing else
                    // here stops it before 448, and the tokens after the tail
                    // collapses are the ones worth never computing.
                    let looping = run.rep.push(tok);
                    if looping {
                        run.out.stop = Stop::Repetition;
                    } else if run.out.tokens.len() >= req.max_tokens {
                        run.out.stop = Stop::Budget;
                    } else if run.pos + 1 >= ctx {
                        run.out.stop = Stop::Context;
                    }
                    run.finished = run.out.stop != Stop::Eot;
                    run.feed = tok;
                    run.phase = 3;
                }
                continue;
            }
            match run.phase {
                0 => {
                    let code = req.lang.as_deref().unwrap_or_default();
                    match model.lang_token(code) {
                        Some(t) => {
                            run.feed = t;
                            run.phase = 1;
                        }
                        None => {
                            lang_err = Some(format!(
                                "whisper: language {code:?} is not in this checkpoint's map"
                            ));
                            run.finished = true;
                        }
                    }
                }
                1 => {
                    run.feed = transcribe;
                    run.phase = 2;
                }
                _ => {
                    run.feed = no_ts;
                    run.phase = 3;
                }
            }
        }
        if let Some(e) = lang_err {
            fail_all(&mut reqs, &mut active, &mut free, &e);
            order.clear();
            continue;
        }

        // ---- 5. retire ----
        let mut i = 0;
        while i < active.len() {
            if !active[i].finished {
                i += 1;
                continue;
            }
            let mut run = active.remove(i);
            // ---- the silence rule, before anything reads the tokens ----
            //
            // whisper's own failure mode: a window holding no speech gets a
            // fluent sentence written over it. The rule is whisper.cpp's
            // (`guards::is_no_speech`) and it needs the finished decode, which
            // is why it lands here and not at the `<|startoftranscript|>` step
            // where `no_speech_prob` was read.
            //
            // The tokens are DROPPED, not flagged, and that asymmetry with the
            // repetition cut is deliberate: a suppressed window has a positive
            // signal that its audio held nothing, so the honest transcript for
            // it is empty. A looping one had real speech in front of the loop,
            // so its text is kept and the caller is told where it stopped
            // being trustworthy.
            run.out.avg_logprob = guards::avg_logprob(&run.out.logprobs);
            run.out.entropy = run.rep.value();
            if guards::is_no_speech(run.out.no_speech_prob, run.out.avg_logprob) {
                run.out.suppressed = true;
                run.out.tokens.clear();
                run.out.logprobs.clear();
                run.out.runners.clear();
            }
            // WORD TIMING runs here, in the gap between "this window is
            // transcribed" and "this slot is somebody else's". The alignment
            // pass reads the cross-attention planes this window was encoded
            // into, which the next admission overwrites - and it clobbers the
            // slot's self-attention KV, which is why it cannot run any earlier.
            // It is a second forward pass on this thread, so every other live
            // slot waits for it; that is the latency this granularity costs and
            // the reason it is opt-in.
            if reqs.get(&run.req).is_some_and(|r| r.words) && !run.out.tokens.is_empty() {
                // The alignment pass drives `step_body` EAGERLY - no graph,
                // no mmaf-off route - so an admission replay still in flight
                // on the side stream would re-create the P39 overlap race
                // here. Join it first; word timing is opt-in and rare, and
                // the sync is a no-op whenever nothing is outstanding
                // (`pending` seats next iteration off the same, now-idle,
                // event).
                if !pending.is_empty()
                    && let Err(e) = model.encode_sync()
                {
                    let e = e.to_string();
                    fail_all(&mut reqs, &mut active, &mut free, &e);
                    order.clear();
                    pending.clear();
                    break;
                }
                let out = {
                    let req = reqs.get(&run.req).expect("checked above");
                    word_times(model, &run, req, &scale)
                };
                match out {
                    Ok(b) => run.out.boundaries = b,
                    // held on the request rather than raised now: the clip's
                    // other windows are still in flight, and `finish` is where
                    // one answer goes out
                    Err(e) => {
                        if let Some(r) = reqs.get_mut(&run.req) {
                            r.err.get_or_insert(e);
                        }
                    }
                }
            }
            free.push(run.slot);
            let Some(req) = reqs.get_mut(&run.req) else {
                continue;
            };
            let suppressed = run.out.suppressed;
            req.out[run.win] = run.out;
            req.done += 1;
            req.emit(Progress::WindowDone {
                window: run.win,
                suppressed,
            });
            if req.done == req.windows.len() {
                let lang = req.lang.clone().unwrap_or_default();
                finish(req, lang);
                reqs.remove(&run.req);
                order.retain(|&id| id != run.req);
            }
        }
    }
}

#[cfg(test)]
mod prior_tests {
    use super::*;

    fn post(v: &[(&str, f32)]) -> Vec<LangProb> {
        v.iter()
            .enumerate()
            .map(|(i, (c, p))| LangProb {
                code: (*c).to_owned(),
                id: i as u32,
                p: *p,
            })
            .collect()
    }

    fn ask(hints: &[&str], strength: f32) -> LanguageAsk {
        LanguageAsk {
            forced: None,
            hints: hints.iter().map(|s| (*s).to_owned()).collect(),
            strength,
        }
    }

    fn total(v: &[LangProb]) -> f32 {
        v.iter().map(|e| e.p).sum()
    }

    /// No hints, no change. The default request must reach the decode with
    /// the model's own answer untouched.
    #[test]
    fn no_hints_leaves_the_posterior_alone() {
        let p = post(&[("de", 0.5), ("sv", 0.3), ("en", 0.2)]);
        let (out, moved) = LanguageAsk::default().apply(p.clone());
        assert_eq!(out, p);
        assert!(moved.is_none());
        // and an explicit set with zero strength is the same thing
        let (out, moved) = ask(&["sv"], 0.0).apply(p.clone());
        assert_eq!(out, p);
        assert!(moved.is_none());
    }

    /// A whisper-sized map: the named languages on top, the rest sharing what
    /// is left. Sized deliberately - the prior's strength is the ratio between
    /// the in-set and out-of-set weights, and on a 4-entry map hinting 2 of
    /// them is exactly uniform and therefore no prior at all.
    fn map99(named: &[(&str, f32)]) -> Vec<LangProb> {
        let rest = 1.0 - named.iter().map(|(_, p)| p).sum::<f32>();
        let filler = 99 - named.len();
        let mut v: Vec<(String, f32)> = named.iter().map(|(c, p)| ((*c).to_owned(), *p)).collect();
        for i in 0..filler {
            v.push((format!("x{i}"), rest / filler as f32));
        }
        v.iter()
            .enumerate()
            .map(|(i, (c, p))| LangProb {
                code: c.clone(),
                id: i as u32,
                p: *p,
            })
            .collect()
    }

    /// The failure this exists for: the audio's own argmax is wrong, the
    /// caller said which languages they speak, and the prior turns it over -
    /// visibly.
    #[test]
    fn a_hint_can_overturn_the_argmax_and_says_so() {
        // the shape measured in practice: Swedish speech, German on top
        let p = map99(&[("de", 0.40), ("sv", 0.25), ("nl", 0.20), ("en", 0.05)]);
        let (out, moved) = ask(&["sv", "en"], DEFAULT_LANGUAGE_PRIOR).apply(p);
        assert_eq!(out[0].code, "sv");
        assert_eq!(
            moved.expect("the audio preferred something else").code,
            "de"
        );
        assert!((total(&out) - 1.0).abs() < 1e-5);
    }

    /// ...and when it AGREES with the audio it is silent: nothing was
    /// overturned, so there is nothing to report.
    #[test]
    fn a_hint_that_agrees_reports_nothing() {
        let p = map99(&[("sv", 0.6), ("de", 0.3), ("en", 0.05)]);
        let (out, moved) = ask(&["sv", "en"], DEFAULT_LANGUAGE_PRIOR).apply(p);
        assert_eq!(out[0].code, "sv");
        assert!(moved.is_none());
    }

    /// SOFT, and this is the test that makes the word mean something: a
    /// language nobody named still wins when the audio is clear enough about
    /// it. A filter could not do this, and Azure's LID - which "returns one of
    /// the candidate languages provided even if those languages weren't in the
    /// audio" - is what it would become.
    #[test]
    fn a_clear_out_of_set_language_still_wins() {
        // 99 languages, 2 hinted, strength 0.5: the prior odds are
        // (0.5/2)/(0.5/97) ~= 48x, so an out-of-set language needs to beat an
        // in-set one by more than that
        let mut v: Vec<(String, f32)> = Vec::new();
        v.push(("ja".to_owned(), 0.90));
        v.push(("sv".to_owned(), 0.008));
        for i in 0..97 {
            v.push((format!("x{i}"), 0.092 / 97.0));
        }
        let p: Vec<LangProb> = v
            .iter()
            .enumerate()
            .map(|(i, (c, q))| LangProb {
                code: c.clone(),
                id: i as u32,
                p: *q,
            })
            .collect();
        let (out, moved) = ask(&["sv", "en"], DEFAULT_LANGUAGE_PRIOR).apply(p);
        assert_eq!(
            out[0].code, "ja",
            "a 112x likelihood ratio was overruled by a hint"
        );
        assert!(moved.is_none(), "nothing moved, so nothing to report");
    }

    /// The other side of the same knob: a MARGINAL out-of-set win is what the
    /// prior is for, and it does get overturned.
    #[test]
    fn a_marginal_out_of_set_win_is_overturned() {
        let mut v = vec![("de".to_owned(), 0.30), ("sv".to_owned(), 0.20)];
        for i in 0..97 {
            v.push((format!("x{i}"), 0.50 / 97.0));
        }
        let p: Vec<LangProb> = v
            .iter()
            .enumerate()
            .map(|(i, (c, q))| LangProb {
                code: c.clone(),
                id: i as u32,
                p: *q,
            })
            .collect();
        let (out, moved) = ask(&["sv", "en"], DEFAULT_LANGUAGE_PRIOR).apply(p);
        assert_eq!(out[0].code, "sv");
        assert_eq!(moved.unwrap().code, "de");
    }

    /// A hint naming languages this checkpoint does not have is a no-op, not
    /// an error and not a silent renormalisation of nothing.
    #[test]
    fn hints_nobody_here_can_serve_change_nothing() {
        let p = post(&[("de", 0.5), ("sv", 0.3), ("en", 0.2)]);
        let (out, moved) = ask(&["xx", "zz"], DEFAULT_LANGUAGE_PRIOR).apply(p.clone());
        assert_eq!(out, p);
        assert!(moved.is_none());
        // and hinting every language is the same as hinting none
        let (out, _) = ask(&["de", "sv", "en"], DEFAULT_LANGUAGE_PRIOR).apply(p.clone());
        assert_eq!(out, p);
    }

    /// The strength is clamped below 1: at 1.0 the "soft hint" would zero
    /// every out-of-set language, which is a filter wearing a hint's name - and
    /// a filter cannot report an out-of-set answer honestly because it has
    /// already thrown it away.
    #[test]
    fn the_prior_can_be_strong_but_never_absolute() {
        let p = post(&[("de", 0.9999), ("sv", 0.0001)]);
        let (out, _) = ask(&["sv"], 1.0).apply(p);
        assert!(
            out.iter().all(|e| e.p > 0.0),
            "an out-of-set language was zeroed"
        );
        assert!((total(&out) - 1.0).abs() < 1e-5);
    }
}
