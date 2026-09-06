//! Loading a model into a running engine, and the served-model handle the
//! routes use (engine + tokenizer + id + stop tokens).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use paddock_engine::encoder::Encoder;
use paddock_engine::generator::Generator;
use paddock_engine::service::Engine;
use paddock_engine::transcriber::Transcriber;
use paddock_models::mapped::MappedGguf;
use paddock_tokenizer::GgufTokenizer;

/// Everything the completion routes need for one served model.
/// What speculation the LOAD actually wired - self-reported over the admin
/// surface so the manager's catalog-side prediction (describe_spec) defers
/// to the process that did the attaching. `heads` = the family drafts from
/// heads in the weights (qwen nextn, nemotron MTP); `drafter` = the attached
/// companion's file stem, which the manager maps back to its catalog label.
#[derive(Debug, Clone, Default)]
pub struct SpecReport {
    pub heads: bool,
    pub drafter: Option<String>,
}

/// Families whose speculative heads live in the weights file. Family-level
/// truth: a stripped export (nextn removed) would still claim heads here -
/// acceptable for a provenance badge, and the engine refuses spec rounds on
/// such a file anyway.
fn arch_has_infile_heads(arch: &str) -> bool {
    matches!(
        arch,
        "qwen35" | "qwen35moe" | "nemotron_h_moe" | "nemotron-h"
    )
}

pub struct ServingModel {
    pub id: String,
    /// speculation mechanisms this load wired (see [`SpecReport`])
    pub spec: SpecReport,
    /// `general.architecture` (or the HF `model_type`) - the checkpoint's own
    /// family name. `id` is the SERVED name and a user can rename it, so this
    /// is what anything keyed on family reads: today the elected sampling
    /// profile, which must not follow a renamed file.
    pub arch: String,
    /// Sampling this CHECKPOINT publishes in its own header, for the case
    /// `arch` cannot decide: granite 4.1 and 4.2 are both `granite` but only
    /// 4.2 publishes (temperature 1.0, top_p 0.95). Consulted only where the
    /// arch-keyed table has no row - see `paddock_models::sampling`.
    pub published_sampling: Option<paddock_models::sampling::Elected>,
    pub engine: Engine,
    pub tokenizer: Arc<GgufTokenizer>,
    /// BOS to prepend, or None when the model's tokenizer says not to.
    pub bos: Option<u32>,
    /// tokens that end generation (eos + family-specific stops)
    pub stop_tokens: Vec<u32>,
    /// the model's Jinja chat template, if it ships one.
    pub chat_template: Option<String>,
    /// task tags that template expands (granite-vision's `<chart2csv>` and
    /// friends) - extracted once at load, empty for every other model. A
    /// client can't reach them without knowing they exist.
    pub task_tags: Vec<crate::chat_template::TaskTag>,
    /// which assistant-output syntax to parse (tool calls, reasoning)
    pub dialect: crate::parsers::Dialect,
    /// what reasoning control this checkpoint's own template implements -
    /// measured from it at load (`crate::reasoning::probe`), because neither
    /// `arch` nor `dialect` can tell Qwen3.8's three-rung ladder apart from
    /// Qwen3.6's on/off switch: they report the same value for both.
    pub reasoning: crate::reasoning::ReasoningCaps,
    /// an mmproj was loaded - image content parts are accepted
    pub supports_vision: bool,
    /// an AUDIO mmproj was loaded - /v1/audio/transcriptions serves
    pub supports_audio: bool,
    /// which host frontend this model's audio goes through. Only meaningful
    /// when `supports_audio`; the two contracts share no geometry, so the
    /// wrong one produces a plausible-looking plane of the wrong width.
    pub audio_frontend: AudioFrontend,
    /// the token the chat template emits once per image (`<|image_pad|>`),
    /// where the vision embeddings get injected
    pub image_pad_id: Option<u32>,
    /// the token the chat template emits once per audio clip
    /// (`<|audio_pad|>` on Qwen3-ASR, `<|audio|>` on granite-speech), where
    /// the tower embeddings get spliced
    pub audio_pad_id: Option<u32>,
    /// Set when the served checkpoint's own chat template is written against a
    /// string content with the audio marker written into it (granite-speech,
    /// both variants), so the OpenAI wire's parts list has to be flattened
    /// before rendering. See `chat_template::inline_audio_content`.
    pub audio_inline_marker: Option<String>,
    /// This checkpoint can be INSTRUCTED to emit word end times in its
    /// transcript (granite-speech-plus).
    ///
    /// Not a property of the frontend - the two granite-speech siblings share
    /// their mel geometry exactly and differ only in what they were trained to
    /// answer. So it rides as its own flag, and the transcription handler
    /// refuses `word` granularity by name on the sibling that cannot, rather
    /// than sending an instruction that would come back as a plain transcript
    /// with no tags in it.
    pub audio_word_times: bool,
    /// The deepseek2-ocr family is serving: the `ocr` request object, the
    /// canonical prompt vocabulary, the sliding-ngram sampling default and
    /// the grounding-region parse all key off this (see `crate::deepseek_ocr`).
    pub ocr: bool,
    /// The paddleocr family is serving: its `ocr` request object maps modes
    /// to the checkpoint's six task prompts (see `crate::paddle_ocr`).
    pub paddleocr: bool,
    /// This model only reads documents (deepseek2-ocr, paddleocr): its decoder
    /// was trained to transcribe pages, and a text-only prompt is
    /// out-of-distribution - it free-runs transcription-vocabulary noise to
    /// the token cap (observed live). The chat surfaces refuse
    /// text-only requests on it, and /v1/models advertises the flag so a
    /// client can gate its composer instead of discovering this by burning
    /// tokens. Broader than `ocr` (which keys deepseek's request object).
    pub document_parser: bool,
    /// per-id byte table for constrained decoding, built on first use
    vocab_cache: std::sync::OnceLock<Arc<crate::constrained::VocabBytes>>,
}

/// Which host mel contract a served audio model consumes.
///
/// The frontend runs on the runner's blocking pool rather than engine-side
/// so the runner has to know which one - and they agree on
/// nothing: 400-point transform / 128 Slaney mels / 100 frames per second
/// for Qwen3-ASR, 512-point / 80 HTK mels / pair-stacked 50 frames per
/// second for granite-speech. Feeding one to the other's tower is caught by
/// a width check rather than transcribed as noise, but the point of naming
/// the choice here is that it never gets that far.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AudioFrontend {
    /// this model takes no audio
    #[default]
    None,
    /// transformers `Qwen3ASRFeatureExtractor` (whisper-derived)
    Qwen3Asr,
    /// transformers `GraniteSpeechFeatureExtractor` (torchaudio-derived)
    GraniteSpeech,
}

impl AudioFrontend {
    /// Run this frontend over a 16 kHz mono clip.
    pub fn features(self, samples: &[f32]) -> Result<paddock_engine::audio::MelFeatures, String> {
        match self {
            AudioFrontend::Qwen3Asr => paddock_engine::audio::qwen3_asr_features(samples),
            AudioFrontend::GraniteSpeech => {
                paddock_engine::audio::granite::speech_features(samples)
            }
            AudioFrontend::None => Err("this model does not take audio input".into()),
        }
    }

    /// Prompt rows a clip of `len` samples will occupy - the same rule the
    /// engine's admission uses, so the context check the handler runs against
    /// it is the one that will actually bind.
    pub fn prompt_rows(self, len: usize) -> usize {
        match self {
            AudioFrontend::Qwen3Asr => paddock_engine::audio::audio_tokens_for_samples(len),
            AudioFrontend::GraniteSpeech => {
                paddock_engine::audio::granite::audio_tokens_for_samples(len)
            }
            AudioFrontend::None => 0,
        }
    }

    /// The longest clip, in seconds, whose audio rows still fit in `rows` -
    /// `prompt_rows` run backwards.
    ///
    /// A generative ASR lane spends the whole clip as prompt rows, so its
    /// ceiling is the context window and it is far lower than people expect:
    /// Qwen3-ASR bills 13 rows a second, which puts a 32k-token server at
    /// roughly 42 minutes and an 8k one at ten. Whisper has no such ceiling -
    /// it windows - and answers `None` from the caller.
    ///
    /// Found by bisection rather than algebra on PURPOSE. The forward rule is
    /// a chunked convolution stack whose closed form differs per family and
    /// has already changed once; bisecting the real function cannot drift from
    /// it, and this runs once per `/server` call.
    pub fn max_clip_s(self, rows: usize) -> Option<f64> {
        if self == AudioFrontend::None || rows == 0 {
            return None;
        }
        let rate = paddock_engine::audio::SAMPLE_RATE;
        // 24 h is a ceiling for the search, not a promise - anything near it
        // means the context is enormous and the answer stops being the binding
        // constraint anyway.
        let (mut lo, mut hi) = (0usize, rate * 60 * 60 * 24);
        if self.prompt_rows(hi) <= rows {
            return Some(hi as f64 / rate as f64);
        }
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.prompt_rows(mid) <= rows {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Some(lo as f64 / rate as f64)
    }
}

impl ServingModel {
    /// The vocab byte table (lazy; ~1MB, a one-time full-vocab decode).
    pub fn vocab_bytes(&self) -> Arc<crate::constrained::VocabBytes> {
        self.vocab_cache
            .get_or_init(|| Arc::new(crate::constrained::VocabBytes::build(&self.tokenizer)))
            .clone()
    }
}

/// An encoder-only served model (Qwen3 dense): serves `/v1/embeddings` and
/// `/v1/rerank`, not the generative chat/completion routes.
pub struct EmbedModel {
    pub id: String,
    pub encoder: Encoder,
    pub tokenizer: Arc<GgufTokenizer>,
    /// EOS to append so last-token pooling reads it (add_eos convention).
    pub eos: Option<u32>,
    /// reranker yes/no relevance tokens (present when the vocab has them).
    pub yes_id: Option<u32>,
    pub no_id: Option<u32>,
    /// whether the GGUF names itself a reranker (diagnostics / model list).
    pub is_reranker: bool,
    /// The encoder's memory counters, so `/api/stats` has an `engine`
    /// block on an embedding/rerank runner too. Only the memory rows are
    /// filled - an encoder emits no tokens, so tok/s and phase are
    /// honestly zero. Before an encoder lane published
    /// `"engine": null` and the Studio's memory breakdown was blank.
    pub metrics: Arc<paddock_engine::metrics::EngineMetrics>,
}

/// A speech-to-text served model (the whisper family): serves
/// `/v1/audio/transcriptions` only. Whisper is an encoder-DECODER - it has
/// no text prompt to page and no continuous batch to join, so it rides its
/// own thread seam rather than the generative `Engine`, the same way the
/// embeddings encoder does.
pub struct AsrModel {
    pub id: String,
    pub transcriber: Transcriber,
    pub tokenizer: Arc<GgufTokenizer>,
    /// decoder position budget - the served cap on one window's tokens
    pub max_tokens: usize,
    /// the checkpoint's timestamp-token geometry, read once at load: what
    /// turns an emitted `<|2.48|>` into 2.48 s of the clip
    pub time_scale: paddock_engine::gpu_model::whisper::TimeScale,
    /// The bare language codes this checkpoint declares, from its own
    /// `lang_to_id` map. Three jobs, and none of them can be done
    /// from a baked list: it validates `language`/`languages[]` so a code the
    /// model has never heard of is a 400 rather than a decode that fails
    /// mid-flight, it is published on `/v1/models` so a client offers exactly
    /// the languages that exist here, and it bounds the transcript's own
    /// language check to what this model could have written.
    pub languages: Vec<String>,
    /// Memory counters for `/api/stats` - see `EmbedModel::metrics`.
    pub metrics: Arc<paddock_engine::metrics::EngineMetrics>,
}

/// The forced-alignment served model (Qwen3-ForcedAligner):
/// serves `/v1/audio/alignments` only. Like the embeddings encoder it is a
/// single-forward lane on its own thread seam - audio + transcript in, word
/// times out, no decode loop.
pub struct AlignModel {
    pub id: String,
    pub aligner: paddock_engine::align::Aligner,
    pub tokenizer: Arc<GgufTokenizer>,
    pub audio_start: u32,
    pub audio_pad: u32,
    pub audio_end: u32,
    pub timestamp: u32,
    /// milliseconds per predicted time-bin class (80 on the 0.6B)
    pub segment_ms: f32,
    /// the packing ceiling - audio rows + word tokens + timestamp slots
    pub max_ctx: usize,
    /// bin count × bin width = the longest clip a single call can address
    pub max_clip_s: f32,
}

/// The forced-aligner checkpoint DIRECTORY for `path`, if it is one - the
/// first (and so far only) safetensors-primary route: a dir whose config.json
/// names `Qwen3ASRForTokenClassification`. Accepts either the directory
/// itself (hand-written configs) or a `.safetensors` file inside it - the
/// manager's spawn path hands the artifact's entry-point FILE, same as every
/// GGUF model, and resolving to the parent here keeps that path uniform
/// instead of teaching the spawner about directory models.
pub fn aligner_dir(path: &Path) -> Option<std::path::PathBuf> {
    let dir = if path.is_dir() {
        path
    } else if path.extension().is_some_and(|x| x == "safetensors") {
        path.parent()?
    } else {
        return None;
    };
    paddock_models::safetensors::AlignerConfig::read(&dir.join("config.json"))
        .ok()
        .map(|_| dir.to_path_buf())
}

/// True for architectures Paddock serves as SPEECH-TO-TEXT models - the
/// server routes these to `/v1/audio/transcriptions` and nothing else.
pub fn is_asr_arch(arch: &str) -> bool {
    arch == "whisper"
}

/// True for architectures Paddock serves as ENCODERS (embeddings/rerank),
/// not as generators - the server routes these to `/v1/embeddings` + rerank.
///
/// Arch `qwen3` is ambiguous: bare, it is the Qwen3-Embedding/Reranker
/// encoder; paired with an AUDIO mmproj it is the Qwen3-ASR generative
/// family. The caller disambiguates with [`mmproj_is_audio`].
pub fn is_encoder_arch(arch: &str) -> bool {
    arch == "qwen3"
}

/// True when the companion mmproj GGUF carries an audio tower
/// (`clip.has_audio_encoder`) - the discriminator that routes arch `qwen3`
/// to the ASR generative family instead of the embeddings encoder.
pub fn mmproj_is_audio(path: &Path) -> bool {
    MappedGguf::open(path).is_ok_and(|m| {
        matches!(
            m.gguf().metadata.get("clip.has_audio_encoder"),
            Some(paddock_models::gguf::Value::Bool(true))
        )
    })
}

/// Calibration-verdict disk cache: the verdict is deterministic per
/// (model bytes, kernel pack bytes, corpus schema), so cache it and skip
/// the ~2-5 s of calibration encodes on every start. Model and pack are
/// fingerprinted by (size, FNV-64 of the first + last MiB) - cheap, and any
/// real content change (different quant, different model, rebuilt pack with
/// changed kernel numerics) moves it. Bump SCHEMA when the corpus, the
/// ladder names, or the acceptance rule change.
const CALIB_SCHEMA: u32 = 6; // v6: F8Smooth rungs + dual-alpha s-vector blobs

fn fnv1a64(chunks: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for c in chunks {
        for &b in *c {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// (size, FNV-64 over size + first MiB + last MiB) of a file.
fn file_fingerprint(path: &Path) -> std::io::Result<(u64, u64)> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    let size = f.metadata()?.len();
    let n = (1 << 20).min(size as usize);
    let mut head = vec![0u8; n];
    f.read_exact(&mut head)?;
    let mut tail = vec![0u8; n];
    f.seek(SeekFrom::End(-(n as i64)))?;
    f.read_exact(&mut tail)?;
    Ok((size, fnv1a64(&[&size.to_le_bytes(), &head, &tail])))
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CalibCacheEntry {
    schema: u32,
    model_size: u64,
    model_hash: String,
    pack_size: u64,
    pack_hash: String,
    profile: String,
    /// Base64 of the engine's SmoothQuant s-vector export - lets a warm
    /// start apply a smooth profile without re-running the stats encode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    smooth: Option<String>,
}

/// Where a model's calibration verdict lives:
/// `<data root>/cache/calib/bs-<model_hash>.json` (three-mode root).
fn calib_cache_path(model_hash: u64) -> Option<PathBuf> {
    Some(
        paddock_admin::data_root()
            .join("cache")
            .join("calib")
            .join(format!("bs-{model_hash:016x}.json")),
    )
}

/// Look up a cached verdict for (model, pack). Returns (cache_path, entry
/// fingerprints, cached profile if valid). IO errors degrade to a miss.
pub struct CalibCache {
    path: Option<PathBuf>,
    entry: CalibCacheEntry,
}

impl CalibCache {
    pub fn probe(model: &Path, pack: Option<&Path>) -> (Self, Option<(String, Option<Vec<u8>>)>) {
        let (msize, mhash) = file_fingerprint(model).unwrap_or((0, 0));
        let (psize, phash) = pack
            .and_then(|p| file_fingerprint(p).ok())
            .unwrap_or((0, 0));
        let entry = CalibCacheEntry {
            schema: CALIB_SCHEMA,
            model_size: msize,
            model_hash: format!("{mhash:016x}"),
            pack_size: psize,
            pack_hash: format!("{phash:016x}"),
            profile: String::new(),
            smooth: None,
        };
        let path = if mhash != 0 {
            calib_cache_path(mhash)
        } else {
            None
        };
        let cached = path
            .as_deref()
            .and_then(|p| std::fs::read(p).ok())
            .and_then(|b| serde_json::from_slice::<CalibCacheEntry>(&b).ok())
            .filter(|c| {
                c.schema == entry.schema
                    && c.model_size == entry.model_size
                    && c.model_hash == entry.model_hash
                    && c.pack_size == entry.pack_size
                    && c.pack_hash == entry.pack_hash
            })
            .map(|c| {
                let smooth = c.smooth.as_deref().and_then(|b| {
                    use base64::Engine as _;
                    base64::engine::general_purpose::STANDARD.decode(b).ok()
                });
                (c.profile, smooth)
            });
        (Self { path, entry }, cached)
    }

    /// Persist a verdict (best-effort: cache failures must never fail a
    /// load). `smooth` carries the engine's s-vector export for smooth
    /// profiles.
    pub fn store(mut self, profile: &str, smooth: Option<Vec<u8>>) {
        let Some(path) = self.path.take() else { return };
        self.entry.profile = profile.to_owned();
        self.entry.smooth = smooth.map(|b| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(b)
        });
        let write = || -> std::io::Result<()> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let body = serde_json::to_vec_pretty(&self.entry)?;
            std::fs::write(&path, body)
        };
        if let Err(e) = write() {
            tracing::warn!(path = %path.display(), error = %e, "calibration cache write failed");
        }
    }
}

/// Reranker calibration task: 16 queries x 12 candidate docs each (the
/// known-relevant passage among 11 distractors from the same generator as
/// [`calib_corpus`]); the metric is the yes/no judge's ranking of the
/// relevant doc. Returns (query, docs, relevant-index-in-docs) triples.
pub fn calib_rerank_corpus() -> Vec<(String, Vec<String>, usize)> {
    let (texts, n_docs, rel) = calib_corpus();
    let mut out = Vec::with_capacity(16);
    for qi in 0..16 {
        let query = texts[n_docs + qi * 4].clone();
        let target = rel[qi * 4];
        let rel_idx = (qi * 5) % 12; // deterministic, varies the answer slot
        let mut docs = Vec::with_capacity(12);
        for j in 0..12 {
            if j == rel_idx {
                docs.push(texts[target].clone());
            } else {
                // HARD negatives: consecutive corpus docs share the query's
                // subject (the corpus is subject-major), differing only in
                // verb/object - unrelated distractors saturate the judge
                // (measured 16/16 everywhere) and discriminate nothing
                let d = (target + j + 1) % n_docs;
                docs.push(texts[d].clone());
            }
        }
        out.push((query, docs, rel_idx));
    }
    out
}

/// The load-time calibration corpus: 384 distinct synthetic passages and 64
/// paraphrase queries with known source docs - the same retrieval task as
/// the offline `qwen3_embed_quality` gate (keep the two generators in sync),
/// so the load-time verdict and the offline gate agree by construction.
/// Returns (texts docs-then-queries, n_docs, per-query source index).
pub fn calib_corpus() -> (Vec<String>, usize, Vec<usize>) {
    const SUBJECTS: [&str; 8] = [
        "the migration scheduler",
        "a coral reef ecosystem",
        "the bond portfolio",
        "a convolutional network",
        "the volcanic monitoring array",
        "an ancient trade route",
        "the fermentation process",
        "a distributed cache layer",
    ];
    const VERBS: [&str; 8] = [
        "coordinates",
        "degrades under",
        "hedges against",
        "classifies",
        "detects precursors of",
        "connected",
        "converts sugars during",
        "invalidates entries after",
    ];
    const OBJECTS: [&str; 6] = [
        "seasonal workload spikes across regions",
        "sustained thermal stress and acidification",
        "interest rate shocks in emerging markets",
        "handwritten postal codes at scale",
        "major eruptions weeks in advance",
        "inland cities with coastal ports",
    ];
    let mut texts = Vec::new();
    for (i, s) in SUBJECTS.iter().enumerate() {
        for (j, v) in VERBS.iter().enumerate() {
            for (k, o) in OBJECTS.iter().enumerate() {
                let idx = (i * VERBS.len() + j) * OBJECTS.len() + k;
                texts.push(format!(
                    "Report {idx}: field observations confirm that {s} {v} {o}, \
                     which analysts consider significant for planning cycle {}.",
                    idx % 17
                ));
            }
        }
    }
    let n_docs = texts.len();
    let mut rel = Vec::with_capacity(64);
    for qi in 0..64 {
        let doc = qi * 6;
        let i = doc / (VERBS.len() * OBJECTS.len());
        let j = (doc / OBJECTS.len()) % VERBS.len();
        let k = doc % OBJECTS.len();
        texts.push(format!(
            "Which report documents that {} {} {}?",
            SUBJECTS[i], VERBS[j], OBJECTS[k]
        ));
        rel.push(doc);
    }
    (texts, n_docs, rel)
}

/// Load an encoder-only model (Qwen3 dense). CUDA only.
pub fn load_embedder(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    max_ctx: usize,
    vram_budget: Option<u64>,
) -> Result<EmbedModel, ServeError> {
    let map =
        MappedGguf::open(path).map_err(|e| ServeError::Open(path.to_path_buf(), e.to_string()))?;
    let arch = map
        .gguf()
        .architecture()
        .ok_or(ServeError::NoArch)?
        .to_owned();
    if device != "cuda" {
        return Err(ServeError::Engine(format!(
            "{arch} encoder needs cuda (got {device:?})"
        )));
    }
    let tokenizer =
        GgufTokenizer::from_gguf(map.gguf()).map_err(|e| ServeError::Tokenizer(e.to_string()))?;
    let eos = tokenizer.eos_id;
    let yes_id = tokenizer.token_to_id("yes");
    let no_id = tokenizer.token_to_id("no");
    let is_reranker = map
        .gguf()
        .metadata
        .get("general.basename")
        .and_then(|v| match v {
            // case-insensitive: real GGUFs ship "Qwen3-Reranker-..."
            paddock_models::gguf::Value::Str(s) => Some(s.to_lowercase().contains("rerank")),
            _ => None,
        })
        .unwrap_or(false);
    let tokenizer = Arc::new(tokenizer);

    let pack = pack.map(Path::to_path_buf);
    let path = path.to_path_buf();
    let metrics = Arc::new(paddock_engine::metrics::EngineMetrics::default());
    let encoder = Encoder::spawn(
        move || {
            let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
                .map_err(|e| e.to_string())?;
            note_device_cc(&exec);
            if let Some(b) = vram_budget {
                exec.set_vram_budget(b);
            }
            let exec = Arc::new(exec);
            let map = MappedGguf::open(&path).map_err(|e| e.to_string())?;
            paddock_engine::gpu_model::qwen3::GpuQwen3::load(exec, &map, max_ctx)
                .map_err(|e| e.to_string())
        },
        Some(Arc::clone(&metrics)),
    )
    .map_err(ServeError::Engine)?;

    Ok(EmbedModel {
        id,
        encoder,
        tokenizer,
        eos,
        yes_id,
        no_id,
        is_reranker,
        metrics,
    })
}

/// Load the forced aligner from its HF checkpoint directory. CUDA only.
/// Special ids are looked up by literal token text in the checkpoint's own
/// tokenizer.json and CROSS-CHECKED against config.json's stamped ids - a
/// drifted pair means the packing would silently address the wrong rows.
pub fn load_aligner(
    id: String,
    dir: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    max_ctx: usize,
    vram_budget: Option<u64>,
) -> Result<AlignModel, ServeError> {
    if device != "cuda" {
        return Err(ServeError::Engine(format!(
            "forced aligner needs cuda (got {device:?})"
        )));
    }
    let cfg = paddock_models::safetensors::AlignerConfig::read(&dir.join("config.json"))
        .map_err(|e| ServeError::Open(dir.to_path_buf(), e.to_string()))?;
    // Same constructor as the nemotron HF-dir lane; the aligner ignores the
    // generative-contract fields and looks its special ids up by literal
    // token text below.
    let tokenizer =
        GgufTokenizer::from_hf_dir(dir).map_err(|e| ServeError::Tokenizer(e.to_string()))?;
    let need = |t: &str| {
        tokenizer
            .token_to_id(t)
            .ok_or_else(|| ServeError::Tokenizer(format!("aligner vocab is missing {t}")))
    };
    let audio_start = need("<|audio_start|>")?;
    let audio_pad = need("<|audio_pad|>")?;
    let audio_end = need("<|audio_end|>")?;
    let timestamp = need("<timestamp>")?;
    if audio_pad != cfg.audio_token_id || timestamp != cfg.timestamp_token_id {
        return Err(ServeError::Tokenizer(format!(
            "aligner id drift: tokenizer says pad {audio_pad}/ts {timestamp}, config stamps {}/{}",
            cfg.audio_token_id, cfg.timestamp_token_id
        )));
    }
    let max_clip_s = cfg.n_labels as f32 * cfg.segment_ms / 1000.0;

    let pack = pack.map(Path::to_path_buf);
    let dir_owned = dir.to_path_buf();
    let aligner = paddock_engine::align::Aligner::spawn(move || {
        let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
            .map_err(|e| e.to_string())?;
        note_device_cc(&exec);
        if let Some(b) = vram_budget {
            exec.set_vram_budget(b);
        }
        paddock_engine::gpu_model::qwen3_asr::GpuQwen3Asr::load_aligner(
            Arc::new(exec),
            &dir_owned,
            max_ctx,
        )
        .map(|(m, _meta)| m)
        .map_err(|e| e.to_string())
    })
    .map_err(ServeError::Engine)?;

    Ok(AlignModel {
        id,
        aligner,
        tokenizer: Arc::new(tokenizer),
        audio_start,
        audio_pad,
        audio_end,
        timestamp,
        segment_ms: cfg.segment_ms,
        max_ctx,
        max_clip_s,
    })
}

/// Load a speech-to-text model (the whisper family). CUDA only.
///
/// `max_ctx` narrows the decoder's own learned position table (448 rows
/// trained); the runner's default serve context is far larger and simply
/// caps there - see the loader, which says so out loud rather than
/// refusing to serve.
#[allow(clippy::too_many_arguments)]
/// --kv-cache-dtype (config-normalized into this env var by lib.rs). Both
/// directions are explicit: fp8_e4m3 is the lossy opt-in for the families
/// that still default to f16, and f16 is the way back for the ones that
/// default to fp8 (gemma4 and both ASR families). Saying nothing leaves each
/// family on its own default, which it announces in its own load log.
/// The compute capability this runner serves on, so the fp8-KV gate below can
/// reach it without threading a device handle through eleven family arms.
///
/// A process singleton because a runner is one: one model, one device
/// (manager-runner doc §3). Zero means "not recorded yet", and an unrecorded
/// device never gates - guessing would be worse than the status quo.
static DEVICE_CC: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Called by every executor constructor, for the reason `make_exec` gives
/// about the VRAM budget: put it where a new family arm cannot forget it.
fn note_device_cc(exec: &paddock_engine::gpu::GpuExecutor) {
    let (major, minor) = exec.compute_capability();
    DEVICE_CC.store(major * 100 + minor, std::sync::atomic::Ordering::Relaxed);
}

fn device_cc() -> Option<(u32, u32)> {
    let v = DEVICE_CC.load(std::sync::atomic::Ordering::Relaxed);
    (v > 0).then_some((v / 100, v % 100))
}

fn apply_kv_dtype(set: impl FnOnce(paddock_engine::gpu::KvDtype)) {
    use paddock_engine::gpu::KvDtype;
    match std::env::var("PADDOCK_KV_CACHE_DTYPE").as_deref() {
        Ok("fp8_e4m3") => {
            // **fp8 KV needs sm_89.** e4m3 is a tensor-core format from Ada
            // onward; on Ampere there is no hardware for it and the cache
            // round-trip is wrong rather than merely lossy. Measured on this
            // A6000 (sm_86) with Qwen3.8-27B, temp 0, one variable changed:
            //
            //   f16 KV   -> "Paris"                     finish_reason: stop
            //   fp8 KV   -> invents a riddle nobody      finish_reason: length
            //               asked about, never closes
            //               its think block
            //
            // Nothing checked the device before (reporting
            // gibberish from a config that simply asked for fp8). The same
            // shape as the NVFP4 checkpoint gate in gpu/fp4.rs: a setting said
            // yes and nobody asked the hardware.
            //
            // Serve f16 rather than refuse to start - the model then works,
            // which is what the person wanted. Loud, because it is not what
            // they asked for and it DOUBLES the KV pool: a server that fit
            // under its budget on fp8 may now not, and that has to be
            // attributable to this line rather than a mystery.
            // The threshold itself lives in paddock_models::gpu_support beside
            // the support table, because the estimator and the Studio need the
            // same answer - the manager used to have no copy at all, so the
            // will-it-fit panel priced a width this runner then refused
            if let Some((major, minor)) = device_cc()
                && let Some(why) = paddock_models::gpu_support::fp8_kv_blocked((major, minor))
            {
                tracing::error!(
                    "kv cache: --kv-cache-dtype fp8_e4m3 asked for, but {why} (this GPU is \
                     sm_{major}{minor}). Serving f16 instead. The KV pool is twice the \
                     size it would have been - lower max_ctx or max_batch if the \
                     server no longer fits."
                );
                set(KvDtype::Fp16);
                return;
            }
            tracing::info!("kv cache: fp8-e4m3 (--kv-cache-dtype; halves KV bytes)");
            set(KvDtype::Fp8E4m3);
        }
        Ok("f16") => {
            tracing::info!("kv cache: f16 (--kv-cache-dtype; exact, doubles KV bytes)");
            set(KvDtype::Fp16);
        }
        // Nothing asked, so the FAMILY default stands - and four of them
        // default to fp8 (gemma4 both sizes, muse-glimmer, paddleocr-vl, plus
        // whisper in its own loader).
        //
        // This arm used to force those four back to f16 below sm_89,
        // under the blanket rule "fp8 KV on fp8 hardware, f16 on Ampere". That
        // rule rested on a mis-diagnosis: the garbage it was written for came
        // from the QK8/P8 e4m3-mma arms storing zeros, not from the
        // format - fp8 STORAGE is software-emulated and byte-exact on Ampere.
        // gpu_support::fp8_kv now answers yes on every die we serve, so the
        // guard below no longer fires and these defaults stand everywhere.
        //
        // Decided with the blast radius named: this changes
        // those four families on Ampere without anyone asking for it, and of
        // them only gemma4's fused KV writer is covered at fp8. whisper has its
        // own fp8 kernel gates; muse-glimmer and paddleocr-vl have neither, and
        // granite's writer is. The evidence, more importantly,
        // its limits.
        _ => {
            if let Some((major, minor)) = device_cc()
                && paddock_models::gpu_support::fp8_kv_blocked((major, minor)).is_some()
            {
                tracing::info!(
                    "kv cache: f16 (sm_{major}{minor} cannot store an fp8 KV cache, so \
                     any family default of fp8-e4m3 is overridden)"
                );
                set(KvDtype::Fp16);
            }
        }
    }
}

pub fn load_asr(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    max_ctx: usize,
    max_batch: usize,
    vram_budget: Option<u64>,
) -> Result<AsrModel, ServeError> {
    let map =
        MappedGguf::open(path).map_err(|e| ServeError::Open(path.to_path_buf(), e.to_string()))?;
    let arch = map
        .gguf()
        .architecture()
        .ok_or(ServeError::NoArch)?
        .to_owned();
    if device != "cuda" {
        return Err(ServeError::Engine(format!(
            "{arch} needs cuda (got {device:?})"
        )));
    }
    let tokenizer =
        GgufTokenizer::from_gguf(map.gguf()).map_err(|e| ServeError::Tokenizer(e.to_string()))?;
    let tokenizer = Arc::new(tokenizer);

    let pack = pack.map(Path::to_path_buf);
    let path = path.to_path_buf();
    // `max_batch` is the decode-slot count, and for whisper that is a real
    // VRAM decision rather than a queue depth: every slot holds its window's
    // cross-attention K/V for all 32 layers (~246 MiB at f16 on the large-v3
    // geometry). The scheduler runs one decode step across all live slots.
    let metrics = Arc::new(paddock_engine::metrics::EngineMetrics::default());
    let (transcriber, card) = Transcriber::spawn(
        move || {
            let exec = paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack.as_deref())
                .map_err(|e| e.to_string())?;
            note_device_cc(&exec);
            if let Some(b) = vram_budget {
                exec.set_vram_budget(b);
            }
            let exec = Arc::new(exec);
            let map = MappedGguf::open(&path).map_err(|e| e.to_string())?;
            let mut m = paddock_engine::gpu_model::whisper::GpuWhisper::load(exec, &map, max_ctx)
                .map_err(|e| e.to_string())?;
            // Whisper's KV is unlike any other family's: the CROSS planes are a
            // full 1500-frame window per slot per layer, static for the decode
            // and never shorter, so at 32 slots a decode step reads ~7.9 GB per
            // token and that one kernel is ~27% of all GPU time. f16 stays the
            // default (it is the arbiter's own class); fp8-e4m3 halves those
            // bytes for callers who ask. Must run before `prepare_batch` sizes
            // the pool - `set_kv_dtype` drops a pool sized for the other width.
            apply_kv_dtype(|d| m.set_kv_dtype(d));
            Ok(m)
        },
        max_batch.max(1),
        Some(Arc::clone(&metrics)),
    )
    .map_err(ServeError::Engine)?;

    // one window's output can never exceed the trained decoder context, and
    // the prompt takes four of those rows
    let max_tokens = max_ctx.min(448).saturating_sub(8).max(16);
    Ok(AsrModel {
        id,
        transcriber,
        metrics,
        tokenizer,
        max_tokens,
        time_scale: card.time_scale,
        languages: card.languages,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("cannot open model {0}: {1}")]
    Open(PathBuf, String),
    #[error(
        "unsupported architecture {0:?} for serving (supported: gpt-oss, qwen35, qwen35moe, gemma4, laguna, granite)"
    )]
    UnsupportedArch(String),
    #[error("model has no architecture metadata")]
    NoArch,
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("engine startup: {0}")]
    Engine(String),
}

/// Build a served model from a GGUF path. `device` must be "cuda" (GPU-only;
/// there is no CPU arm);
/// `gpu` is the resolved CUDA ordinal; `pack` is required for cuda;
/// `fp8_native` is the optional safetensors snapshot for native-fp8 planes;
/// `vram_budget` (bytes) is the hard cap the executor sizes everything
/// inside - an explicit load option like fp8_native, never env.
#[allow(clippy::too_many_arguments)]
pub fn load(
    id: String,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    max_ctx: usize,
    max_batch: usize,
    mmproj: Option<&Path>,
    mtp: Option<&Path>,
    fp8_native: Option<&Path>,
    vram_budget: Option<u64>,
) -> Result<ServingModel, ServeError> {
    // The safetensors-primary fork: a checkpoint DIRECTORY is the
    // HF-native lane - no GGUF exists in it, so everything (arch, tokenizer,
    // template, weights) comes from the checkpoint's own files. First family:
    // nemotron_h_moe served straight from the NVFP4 export.
    if path.is_dir() && path.join("config.json").exists() {
        return load_hf_dir(id, path, device, gpu, pack, max_ctx, max_batch, vram_budget);
    }
    // open once here for tokenizer + metadata; the engine reopens on its thread
    let map =
        MappedGguf::open(path).map_err(|e| ServeError::Open(path.to_path_buf(), e.to_string()))?;
    let arch = map
        .gguf()
        .architecture()
        .ok_or(ServeError::NoArch)?
        .to_owned();

    // sidecar-aware: SPM-class GGUFs (paddleocr) carry no merges, so the
    // checkpoint's tokenizer.json next to the weights is the source of truth
    let weights_dir = path.parent().unwrap_or_else(|| Path::new("."));
    let tokenizer = GgufTokenizer::from_gguf_with_sidecar(map.gguf(), weights_dir)
        .map_err(|e| ServeError::Tokenizer(e.to_string()))?;
    // only lead with BOS when the model's tokenizer asks for it
    let bos = tokenizer.add_bos.then_some(tokenizer.bos_id).flatten();

    // stop tokens: eos plus known family-specific enders
    let mut stop_tokens = Vec::new();
    if let Some(eos) = tokenizer.eos_id {
        stop_tokens.push(eos);
    }
    // The DECLARED end-of-turn token, when the file has one. Two families ship
    // it and both need it: laguna's eot 24 is `</assistant>` (its
    // eos is the BOS/EOS marker 2, which the model never emits mid-turn), and
    // muse-glimmer's is `<|eot|>` 200008 - its eos `<|end_of_text|>` likewise
    // never appears, so without this a turn ran past its own ending and the
    // model carried on writing the next one.
    // Every other served file leaves the key unset, so this changes nothing
    // for them.
    if let Some(eot) = tokenizer.eot_id
        && !stop_tokens.contains(&eot)
    {
        stop_tokens.push(eot);
    }
    for name in [
        "<|return|>",
        "<|call|>",
        "<|endoftext|>",
        "<end_of_turn>",
        "<eos>",
        "</assistant>",
    ] {
        if let Some(id) = tokenizer.token_to_id(name)
            && !stop_tokens.contains(&id)
        {
            stop_tokens.push(id);
        }
    }

    // An attached audio mmproj on the ASR arch means chat requests must
    // render the OFFICIAL audio chat template (what vLLM renders - the parity
    // target), not whatever the GGUF embedded: the checkpoint ships its
    // template in the Omni-style `chat_template.json`, which GGUF converters
    // reading `tokenizer_config.json` never see, so converted files carry the
    // generic ChatML fallback with no audio branch at all. Computed here,
    // before the struct init, because `supports_audio` below uses the same
    // predicate.
    //
    // granite-speech's GGUF does carry its official template - and the two
    // variants carry different ones (the base's bare `USER: ... ASSISTANT:`,
    // -plus's full granite-4 envelope with a default system message). Both are
    // written against a string content the caller wrote `<|audio|>` into, so
    // the parts list gets flattened into that shape before rendering and the
    // checkpoint's own template stands. Serving one variant's envelope to the
    // other cost -plus its entire system block.
    let granite_audio = arch == "granite" && mmproj.is_some_and(mmproj_is_audio);
    let audio_serving = mmproj.is_some() && (arch == "qwen3vl" || granite_audio);
    // Which granite-speech sibling this is. IBM's own conversion stamps
    // `general.finetune = "plus"` on the -plus weights and leaves the field off
    // the base one, so the checkpoint says which it is and we never have to
    // guess from a filename the user may have renamed. `general.name` ("Granite
    // Speech 4.1 2b Plus") is checked too, as the one that survives a
    // re-conversion that drops the finetune field.
    let granite_plus = granite_audio
        && ["general.finetune", "general.name"].iter().any(|k| {
            matches!(map.gguf().metadata.get(*k),
                Some(paddock_models::gguf::Value::Str(s)) if s.to_lowercase().ends_with("plus"))
        });
    // The DeepSeek-OCR family ships no chat template anywhere in the
    // checkpoint, so a GGUF converter has nothing to copy: llama.cpp writes the
    // placeholder `{% for m in messages %}{{m['content']}}{% endfor %}`, which
    // concatenates content but never renders an image marker - every picture
    // would be dropped on the floor before `build_mm_chunks` ever saw a pad.
    // The family override carries the real shape (bare concat, markers at the
    // prefix, caller's own `<image>` wins); see the constant's docs and
    let deepseek_ocr = arch == "deepseek2-ocr";
    let chat_template = if granite_audio {
        tokenizer.chat_template.clone()
    } else if audio_serving {
        Some(crate::chat_template::QWEN3_ASR_AUDIO_TEMPLATE.to_owned())
    } else if deepseek_ocr {
        Some(crate::chat_template::DEEPSEEK_OCR_TEMPLATE.to_owned())
    } else if arch == "paddleocr" {
        // the GGUF's own template with its system-string arm aligned to the
        // checkpoint's list-walk bytes - see the constant's docs
        Some(crate::chat_template::PADDLEOCR_VL_TEMPLATE.to_owned())
    } else {
        tokenizer.chat_template.clone()
    };
    // Costs a handful of renders of a one-message conversation, once per
    // process, and only on a template that has candidates at all.
    let task_tags = chat_template
        .as_deref()
        .map(crate::chat_template::task_tags)
        .unwrap_or_default();
    // What reasoning control this checkpoint implements, read off its own
    // template once at load - see `crate::reasoning` for why it cannot be a
    // table keyed on `arch` or `dialect`.
    // Read off the template too, not just `arch`: one arch string can cover
    // two incompatible dialects (granite 4.1 JSON vs 4.2 XML+thinking).
    let dialect = crate::parsers::Dialect::for_arch_and_template(&arch, chat_template.as_deref());
    // Same split, other axis: what this file publishes for sampling, used only
    // where the arch-keyed table has nothing.
    let published_sampling = paddock_models::sampling::published_in_gguf(map.gguf());
    let reasoning = chat_template
        .as_deref()
        .map_or_else(crate::reasoning::ReasoningCaps::none, |t| {
            crate::reasoning::probe(t, dialect)
        });
    // per-image template placeholder: qwen renders <|image_pad|>, gemma4
    // renders <|image|> (the engine replaces it with begin+soft+end), granite
    // renders <image> (which mtmd treats as the whole marker - no end token),
    // muse-glimmer renders <|patch|> (and brackets with <|image_start|> /
    // <|image_end|>, which the engine adds). Only ever consulted once
    // `supports_vision` is true, so a text model that happens to carry one of
    // these strings in its vocab is unaffected.
    //
    // ORDER MATTERS where a vocab has several: muse-glimmer's carries both
    // <|image|> (200090, unused by its template) and <|patch|> (200092, what
    // the template actually emits), so <|image|> must not win. Keyed off the
    // arch rather than off vocab-probe order, because "which token does this
    // model's template render" is not something the vocab can answer.
    let image_pad_id = if arch == "muse-glimmer" {
        tokenizer.token_to_id("<|patch|>")
    } else if arch == "paddleocr" {
        // its vocab also holds an <|image_pad|> the template never emits -
        // the exact muse trap; the ERNIE template renders <|IMAGE_PLACEHOLDER|>
        tokenizer.token_to_id("<|IMAGE_PLACEHOLDER|>")
    } else {
        tokenizer
            .token_to_id("<|image_pad|>")
            .or_else(|| tokenizer.token_to_id("<|image|>"))
            .or_else(|| tokenizer.token_to_id("<image>"))
    };
    // per-audio template placeholder: the ASR template renders
    // <|audio_start|><|audio_pad|><|audio_end|> per clip and granite-speech
    // renders a bare <|audio|>; the pad is the splice marker, consulted only
    // once `supports_audio` is true.
    let audio_pad_id = tokenizer
        .token_to_id("<|audio_pad|>")
        .or_else(|| tokenizer.token_to_id("<|audio|>"));
    let tokenizer = Arc::new(tokenizer);
    let engine = build_engine(
        &arch,
        path.to_path_buf(),
        device,
        gpu,
        pack.map(Path::to_path_buf),
        max_ctx,
        max_batch,
        mmproj.map(Path::to_path_buf),
        mtp.map(Path::to_path_buf),
        fp8_native.map(Path::to_path_buf),
        vram_budget,
    )?;

    Ok(ServingModel {
        id,
        spec: SpecReport {
            heads: arch_has_infile_heads(&arch),
            drafter: mtp.and_then(|m| m.file_stem().map(|f| f.to_string_lossy().into_owned())),
        },
        arch: arch.clone(),
        published_sampling,
        engine,
        tokenizer,
        bos,
        stop_tokens,
        chat_template,
        task_tags,
        dialect,
        reasoning,
        // An attached mmproj is the whole test: every arm below fails the load
        // outright when the mmproj does not match the text model, so reaching
        // here with one means vision is really serving. `granite` is the one
        // arch that takes either kind of mmproj, so an audio one must not
        // advertise vision - the two towers are mutually exclusive there.
        supports_vision: mmproj.is_some()
            && !granite_audio
            && (arch == "qwen35"
                || arch == "qwen35moe"
                || arch == "gemma4"
                || arch == "muse-glimmer"
                || arch == "granite"
                || arch == "deepseek2-ocr"
                || arch == "paddleocr"),
        // same contract: the qwen3vl (ASR) and granite-speech arms hard-require
        // the audio tower and `attach_audio` refuses a mismatched file, so
        // reaching here means transcription really serves
        supports_audio: audio_serving,
        audio_frontend: if !audio_serving {
            AudioFrontend::None
        } else if granite_audio {
            AudioFrontend::GraniteSpeech
        } else {
            AudioFrontend::Qwen3Asr
        },
        image_pad_id,
        audio_pad_id,
        // Only granite-speech: Qwen3-ASR renders its clips through the audio
        // slots in its own (overridden) template, so its parts must stay parts.
        audio_inline_marker: granite_audio.then(|| "<|audio|>".to_owned()),
        audio_word_times: granite_plus,
        ocr: deepseek_ocr,
        paddleocr: arch == "paddleocr",
        document_parser: deepseek_ocr || arch == "paddleocr",
        vocab_cache: std::sync::OnceLock::new(),
    })
}

/// Serve an HF checkpoint directory (safetensors-primary lane).
/// Arch detection is `config.json`'s `model_type` - currently the one
/// supported family is `nemotron_h` (nemotron_h_moe / Nemotron 3.5
/// Lightning); anything else fails loudly rather than guessing a graph.
/// The tokenizer, decode contract (eos SET -> eos+eot) and chat template all
/// come from the checkpoint's own files via `GgufTokenizer::from_hf_dir`.
#[allow(clippy::too_many_arguments)]
fn load_hf_dir(
    id: String,
    dir: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    max_ctx: usize,
    max_batch: usize,
    vram_budget: Option<u64>,
) -> Result<ServingModel, ServeError> {
    // The family comes from the checkpoint's own `model_type`, because in this
    // lane there is no GGUF header to ask. Parsing the config here is also the
    // validation: reaching `build_engine` means the geometry is readable, so a
    // malformed checkpoint fails with its own name rather than inside a loader.
    let model_type = std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| {
            v.get("model_type")
                .and_then(|x| x.as_str())
                .map(str::to_owned)
        })
        .ok_or_else(|| {
            ServeError::Open(dir.to_path_buf(), "config.json has no model_type".into())
        })?;
    let arch = match model_type.as_str() {
        "nemotron_h" => {
            paddock_models::nemotron::NemotronConfig::read(dir)
                .map_err(|e| ServeError::Open(dir.to_path_buf(), e.to_string()))?;
            "nemotron".to_owned()
        }
        // granite 4.2's NVFP4 export ships no GGUF, so this is the only way to
        // serve it. Same arch string as the GGUF lane deliberately - it is the
        // same decoder, and everything keyed on family (dialect, sampling,
        // reasoning) must resolve identically for both.
        "granite" => {
            paddock_models::granite::GraniteConfig::read(dir)
                .map_err(|e| ServeError::Open(dir.to_path_buf(), e.to_string()))?;
            "granite".to_owned()
        }
        // Qwen3.8-Flash-Next: NVFP4 safetensors only (no GGUF exists for this
        // arch, and the FP8 checkpoint does not fit the disk). Serving it is
        // what turns this lane's bare-loop numbers into board cells - before
        // this arm, `qwen4_exp` appeared in gpu_model/ and nowhere else.
        "qwen4_exp" => {
            paddock_models::qwen4exp::Qwen4ExpConfig::read(dir)
                .map_err(|e| ServeError::Open(dir.to_path_buf(), e.to_string()))?;
            "qwen4exp".to_owned()
        }
        other => {
            return Err(ServeError::Open(
                dir.to_path_buf(),
                format!(
                    "no safetensors-primary lane for model_type {other:?} - this lane serves \
                     checkpoint directories, and only nemotron_h, granite and qwen4_exp \
                     have one"
                ),
            ));
        }
    };

    let tokenizer =
        GgufTokenizer::from_hf_dir(dir).map_err(|e| ServeError::Tokenizer(e.to_string()))?;
    let bos = tokenizer.add_bos.then_some(tokenizer.bos_id).flatten();
    // the generation_config eos SET landed as eos_id + eot_id ([2, 11] -
    // `</s>` and `<|im_end|>`); both stop a turn
    let mut stop_tokens = Vec::new();
    if let Some(eos) = tokenizer.eos_id {
        stop_tokens.push(eos);
    }
    if let Some(eot) = tokenizer.eot_id
        && !stop_tokens.contains(&eot)
    {
        stop_tokens.push(eot);
    }
    let chat_template = tokenizer.chat_template.clone();
    let task_tags = chat_template
        .as_deref()
        .map(crate::chat_template::task_tags)
        .unwrap_or_default();
    // What reasoning control this checkpoint implements, read off its own
    // template once at load - see `crate::reasoning` for why it cannot be a
    // table keyed on `arch` or `dialect`.
    let dialect = crate::parsers::Dialect::for_arch_and_template(&arch, chat_template.as_deref());
    let reasoning = chat_template
        .as_deref()
        .map_or_else(crate::reasoning::ReasoningCaps::none, |t| {
            crate::reasoning::probe(t, dialect)
        });
    let tokenizer = Arc::new(tokenizer);
    let engine = build_engine(
        &arch,
        dir.to_path_buf(),
        device,
        gpu,
        pack.map(Path::to_path_buf),
        max_ctx,
        max_batch,
        None,
        None,
        None,
        vram_budget,
    )?;

    Ok(ServingModel {
        id,
        spec: SpecReport::default(),
        arch: arch.clone(),
        // Read from `generation_config.json`, so this lane and the GGUF lane
        // answer at the same sampling for the same model - granite 4.2 serves
        // from either file and must not change behaviour with the format.
        published_sampling: paddock_models::sampling::published_in_hf_dir(dir),
        engine,
        tokenizer,
        bos,
        stop_tokens,
        chat_template,
        task_tags,
        dialect,
        reasoning,
        supports_vision: false,
        supports_audio: false,
        audio_frontend: AudioFrontend::None,
        image_pad_id: None,
        audio_pad_id: None,
        audio_inline_marker: None,
        audio_word_times: false,
        ocr: false,
        paddleocr: false,
        document_parser: false,
        vocab_cache: std::sync::OnceLock::new(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_engine(
    arch: &str,
    path: PathBuf,
    device: &str,
    gpu: usize,
    pack: Option<PathBuf>,
    max_ctx: usize,
    max_batch: usize,
    mmproj: Option<PathBuf>,
    mtp: Option<PathBuf>,
    fp8_native: Option<PathBuf>,
    vram_budget: Option<u64>,
) -> Result<Engine, ServeError> {
    let arch = arch.to_owned();
    let device = device.to_owned();

    // Continuous-batching width (config `max_batch` / PADDOCK_MAX_BATCH / --max-batch;
    // 1 forces the serial loop). GPU models batch concurrent requests through one
    // weight-amortized step; the CPU reference ignores this and runs serial.
    let max_batch = max_batch.max(1);

    // the factory runs on the engine thread (required for CUDA context binding)
    Engine::spawn(max_batch, move || {
        build_generator(
            &arch,
            &path,
            &device,
            gpu,
            pack.as_deref(),
            max_ctx,
            max_batch,
            mmproj.as_deref(),
            mtp.as_deref(),
            fp8_native.as_deref(),
            vram_budget,
        )
    })
    .map_err(ServeError::Engine)
}

#[allow(clippy::too_many_arguments)]
fn build_generator(
    arch: &str,
    path: &Path,
    device: &str,
    gpu: usize,
    pack: Option<&Path>,
    max_ctx: usize,
    // qwen4exp sizes its GDN recurrent state, both conv windows and every
    // scratch plane by the SLOT COUNT at load, so unlike the families that
    // allocate lazily in `enable_batch` it needs the serve width here.
    max_batch: usize,
    mmproj: Option<&Path>,
    mtp: Option<&Path>,
    fp8_native: Option<&Path>,
    vram_budget: Option<u64>,
) -> Result<Box<dyn Generator>, String> {
    // one constructor applies the config'd VRAM budget so a new family arm
    // can't forget it - the executor's headroom seam does the enforcing
    let make_exec = |pack: Option<&Path>| -> Result<Arc<paddock_engine::gpu::GpuExecutor>, String> {
        let exec =
            paddock_engine::gpu::GpuExecutor::with_pack(gpu, pack).map_err(|e| e.to_string())?;
        note_device_cc(&exec);
        if let Some(b) = vram_budget {
            tracing::info!(
                "VRAM budget {:.1} GiB (config vram_budget) - load gate, pools and caches keep inside it",
                b as f64 / (1u64 << 30) as f64
            );
            exec.set_vram_budget(b);
        }
        Ok(Arc::new(exec))
    };
    // The safetensors-primary arm reads the checkpoint DIRECTORY itself - no
    // GGUF exists in this lane, so it forks before the mmap below.
    //
    // The fork is on the PATH SHAPE, not on the arch string: granite serves
    // both lanes under the same `general.architecture`, so "is this a
    // directory" is the only thing that actually distinguishes them.
    if path.is_dir() {
        if device != "cuda" {
            return Err(format!("{arch} needs cuda (got {device:?})"));
        }
        let exec = make_exec(pack)?;
        return match arch {
            "nemotron" => {
                let mut model =
                    paddock_engine::gpu_model::nemotron::GpuNemotron::load_dir(exec, path, max_ctx)
                        .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                // the official DFlash drafter is a safetensors DIR (never a
                // GGUF); lib.rs leaves `mtp` None when spec is off
                if let Some(mp) = mtp {
                    model.attach_dflash(mp).map_err(|e| e.to_string())?;
                }
                Ok(Box::new(model) as Box<dyn Generator>)
            }
            "granite" => {
                let mut model =
                    paddock_engine::gpu_model::granite::GpuGranite::load_dir(exec, path, max_ctx)
                        .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                Ok(Box::new(model) as Box<dyn Generator>)
            }
            "qwen4exp" => {
                let model = paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu::load_with_slots(
                    &exec,
                    path,
                    max_ctx,
                    max_batch.max(1),
                )
                .map_err(|e| e.to_string())?;
                Ok(Box::new(model) as Box<dyn Generator>)
            }
            other => Err(format!("{other}: no safetensors-primary lane")),
        };
    }
    let map = MappedGguf::open(path).map_err(|e| e.to_string())?;
    match arch {
        "gpt-oss" => match device {
            "cuda" => {
                let exec = make_exec(pack)?;
                let mut model =
                    paddock_engine::gpu_model::gpt_oss::GpuGptOss::load(exec, &map, max_ctx)
                        .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                Ok(Box::new(model))
            }
            other => Err(format!("gpt-oss needs cuda (got {other:?})")),
        },
        // Qwen3.8-Flash-Next off a llama.cpp GGUF (the Unsloth UD exports):
        // the consumer-card lane - k-quant dense planes, k-quant / i-quant
        // expert seats, host-mapped under [moe_offload] with the VRAM slot
        // cache seated from whatever the load left free, the PLE table read
        // out of the mmap. The safetensors NVFP4 lane above is untouched.
        "qwen4exp" => match device {
            "cuda" => {
                let exec = make_exec(pack)?;
                drop(map);
                let mut model =
                    paddock_engine::gpu_model::qwen4exp::Qwen4ExpGpu::load_gguf_with_slots(
                        &exec,
                        path,
                        max_ctx,
                        max_batch.max(1),
                    )
                    .map_err(|e| e.to_string())?;
                let host = model.expert_host_bytes();
                if host > 0 {
                    let headroom = exec.vram_headroom().unwrap_or(0);
                    let budget = headroom.saturating_sub(512 << 20);
                    let seated = model.enable_moe_cache(budget).map_err(|e| e.to_string())?;
                    eprintln!(
                        "[q4x-moe] {:.1} GiB of experts host-mapped; slot cache on {seated} layers from {:.2} GiB headroom",
                        host as f64 / (1u64 << 30) as f64,
                        headroom as f64 / (1u64 << 30) as f64
                    );
                }
                Ok(Box::new(model) as Box<dyn Generator>)
            }
            other => Err(format!("qwen4exp needs cuda (got {other:?})")),
        },
        // Both the dense (qwen35) and MoE (qwen35moe, e.g. Qwen3.6-35B-A3B-MTP)
        // families load through GpuQwen35 - it detects MoE from the expert_* GGUF
        // metadata, so the arch string only needs to route here.
        "qwen35" | "qwen35moe" => match device {
            "cuda" => {
                // SPEC-SERVE ELECTION: the projections REPLACE
                // (w8_min 0 / f8_dec_min 1) was priced on nospec rows only.
                // Spec verify rows (r = 8..64) are launch-bound and lose
                // measurably through the f8row wmma class, so a spec serve
                // restores the pre-REPLACE floors - the Q8 projection planes
                // stay resident (+7.67 GB) and the r<=64 band keeps its
                // faster class. Nospec serves keep the full REPLACE and its
                // VRAM win. Explicit user env always wins; the deeper fix is
                // a small-M f8 election the REPLACE default can keep, which
                // belongs in the kernel pack, not here.
                if mtp.is_some() {
                    for (k, v) in [
                        ("PADDOCK_QWEN35_W8_MIN", "64"),
                        ("PADDOCK_QWEN35_F8_DEC_MIN", "8"),
                    ] {
                        if std::env::var_os(k).is_none() {
                            // SAFETY: before model load and serving threads
                            unsafe { std::env::set_var(k, v) };
                        }
                    }
                    eprintln!(
                        "[spec-elect] qwen35 spec serve: pre-REPLACE \
projection floors restored (W8_MIN=64, F8_DEC_MIN=8) - planes resident"
                    );
                }
                let exec = make_exec(pack)?;
                let mut model = paddock_engine::gpu_model::qwen35::GpuQwen35::load_with(
                    exec, &map, max_ctx, fp8_native,
                )
                .map_err(|e| e.to_string())?;
                if let Some(mp) = mmproj {
                    let mmap = MappedGguf::open(mp).map_err(|e| e.to_string())?;
                    model.attach_vision(&mmap).map_err(|e| e.to_string())?;
                }
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                // Drafter sideload dispatches on the companion's own arch,
                // exactly like the gemma4 arm: a "dflash" file attaches the
                // DFlash2 block-diffusion drafter (incoai/z-lab Qwen3.8),
                // anything else replays the in-file-MTP sideload path. The
                // in-file nextn head needs no companion and keeps working
                // when `mtp` is unset.
                if let Some(dp) = mtp {
                    let dmap = MappedGguf::open(dp).map_err(|e| e.to_string())?;
                    let darch = dmap
                        .gguf()
                        .metadata
                        .get("general.architecture")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned();
                    if darch == "dflash" {
                        drop(dmap);
                        model.attach_dflash(dp).map_err(|e| e.to_string())?;
                    } else {
                        return Err(format!(
                            "qwen35 drafter companion has arch {darch:?} - this family \
                             speculates from its in-file MTP head or a \"dflash\" drafter"
                        ));
                    }
                }
                Ok(Box::new(model))
            }
            other => Err(format!("qwen35 needs cuda (got {other:?})")),
        },
        // Gemma 4 (dense; 31B first). Correctness-milestone path: batch-1
        // decode + token-by-token prefill - serving lanes land with parity.
        // `muse-glimmer` (Meta Superintelligence Labs, Aug 2026) rides the same
        // family: it shares gemma4's sliding/full interleave, four-norm
        // sandwich, QK-norm and final logit softcap, and differs only in a
        // handful of file-derived constants plus an attention output gate.
        // GpuGemma4::load_with reads `general.architecture` and picks the
        // header key prefix and graph constants from it - see gemma4::Arch.
        "gemma4" | "muse-glimmer" => match device {
            "cuda" => {
                let exec = make_exec(pack)?;
                let mut model = paddock_engine::gpu_model::gemma4::GpuGemma4::load_with(
                    exec, &map, max_ctx, fp8_native,
                )
                .map_err(|e| e.to_string())?;
                // uniform with the other four families - gemma4
                // previously had no set_kv_dtype at all; the f16 direction
                // still also flows via the PADDOCK_G4_KV16 env (lib.rs), which
                // its alloc_kv keeps honoring
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                if let Some(mp) = mmproj {
                    let mmap = MappedGguf::open(mp).map_err(|e| e.to_string())?;
                    model.attach_vision(&mmap).map_err(|e| e.to_string())?;
                }
                // Two drafter classes share the `mtp` sideload key, and the
                // FILE says which: gemma-4 ships `gemma4-assistant` (an
                // EAGLE-class chained MTP head), muse-glimmer ships a
                // `dflash` block-diffusion drafter. Dispatch on the
                // checkpoint's own architecture rather than the target's, so
                // a mismatched pair fails at attach with a real message
                // instead of loading half a graph.
                if let Some(dp) = mtp {
                    let dmap = MappedGguf::open(dp).map_err(|e| e.to_string())?;
                    let darch = dmap
                        .gguf()
                        .metadata
                        .get("general.architecture")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default()
                        .to_owned();
                    if darch == "dflash" {
                        drop(dmap);
                        model.attach_dflash(dp).map_err(|e| e.to_string())?;
                    } else {
                        model.attach_mtp(&dmap).map_err(|e| e.to_string())?;
                    }
                }
                Ok(Box::new(model))
            }
            other => Err(format!("gemma4 needs cuda (got {other:?})")),
        },
        // Laguna (poolside XS-2.1 first). Correctness-milestone path: batch-1
        // decode + token-by-token prefill via the serial loop - the batch
        // engine / paged-KV lanes land with parity, same as gemma4 did.
        // KV defaults f16 (greedy-exact); --kv-cache-dtype fp8_e4m3 is a
        // lossy opt-in (the DFlash drafter's own aux-feature KV is
        // independent and stays pinned to f16 regardless).
        "laguna" => match device {
            "cuda" => {
                let exec = make_exec(pack)?;
                let mut model =
                    paddock_engine::gpu_model::laguna::GpuLaguna::load(exec, &map, max_ctx)
                        .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                // DFlash drafter (BF16 safetensors, not a GGUF) rides the same
                // `mtp` drafter-sideload config key the other families use
                if let Some(dp) = mtp {
                    model.attach_dflash(dp).map_err(|e| e.to_string())?;
                }
                Ok(Box::new(model))
            }
            other => Err(format!("laguna needs cuda (got {other:?})")),
        },
        // IBM Granite 4.1 (3b/8b/30b share the code - shapes come from the
        // file). Greedy-parity validated against llama.cpp on the identical
        // GGUF, and the full serving lane is in: paged KV, continuous
        // batching, chunked prefill, device sampling, captured decode graphs.
        // The GGUF second lane of the nemotron family - the
        // safetensors-primary NVFP4 arm forks pre-mmap above under arch
        // "nemotron"; this arm is what `general.architecture` in the unsloth
        // Q8_0 file actually says.
        "nemotron_h_moe" => match device {
            "cuda" => {
                let exec = make_exec(pack)?;
                let mut model =
                    paddock_engine::gpu_model::nemotron::GpuNemotron::load(exec, &map, max_ctx)
                        .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                if let Some(mp) = mtp {
                    model.attach_dflash(mp).map_err(|e| e.to_string())?;
                }
                Ok(Box::new(model))
            }
            other => Err(format!("nemotron needs cuda (got {other:?})")),
        },
        "granite" => match device {
            "cuda" => {
                let exec = make_exec(pack)?;
                let mut model =
                    paddock_engine::gpu_model::granite::GpuGranite::load(exec, &map, max_ctx)
                        .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                // The `granite` arch serves two multimodal families off one
                // decoder, told apart by the companion file: granite-vision's
                // mmproj carries the ViT tower AND the eight DeepStack
                // projectors, granite-speech's carries the conformer + Q-Former
                // audio tower (`clip.has_audio_encoder`). Both attach calls
                // refuse the other's file and a mismatched text model, so a
                // wrong pair fails here rather than mid-prefill.
                if let Some(mp) = mmproj {
                    let mmap = MappedGguf::open(mp).map_err(|e| e.to_string())?;
                    if mmproj_is_audio(mp) {
                        model.attach_audio(&mmap).map_err(|e| e.to_string())?;
                    } else {
                        model.attach_vision(&mmap).map_err(|e| e.to_string())?;
                    }
                }
                Ok(Box::new(model))
            }
            other => Err(format!("granite needs cuda (got {other:?})")),
        },
        // Qwen3-ASR: the llama.cpp conversion stamps arch `qwen3vl` (the
        // converter reuses that text class), with `rope.dimension_sections`
        // - but both references drive every mRoPE axis with the same
        // sequential position for text AND audio (vLLM qwen3_asr
        // get_mrope_input_positions: `arange(..).expand(3, -1)` on every
        // segment), and M-RoPE with equal axes is exactly plain 1D rope, so
        // the family decodes with plain NEOX rope. The audio tower is
        // required: bare qwen3vl (a vision checkpoint) has no family here.
        "qwen3vl" => match device {
            "cuda" => {
                let mp = mmproj.ok_or_else(|| {
                    "qwen3vl is served only as Qwen3-ASR - pass its audio mmproj \
                     (--mmproj mmproj-*.gguf)"
                        .to_string()
                })?;
                let exec = make_exec(pack)?;
                let mut model =
                    paddock_engine::gpu_model::qwen3_asr::GpuQwen3Asr::load(exec, &map, max_ctx)
                        .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                let mmap = MappedGguf::open(mp).map_err(|e| e.to_string())?;
                model.attach_audio(&mmap).map_err(|e| e.to_string())?;
                Ok(Box::new(model))
            }
            other => Err(format!("qwen3-asr needs cuda (got {other:?})")),
        },
        // DeepSeek-OCR family (`DeepSeek-OCR`, `DeepSeek-OCR-2`,
        // `baidu/Unlimited-OCR` - one decoder covers all three). The mmproj
        // is required: a document parser without its DeepEncoder tower can
        // only complete text, which is not a service anyone asked this model
        // for - and every prompt-mapping default assumes an image.
        // R-SWA means the fit estimate prices KV as prefill + 128 rows no
        // matter the output length; the batched lane is where that pinning,
        // the radix prefix and the tuned prefill ladder live.
        "deepseek2-ocr" => match device {
            "cuda" => {
                let mp = mmproj.ok_or_else(|| {
                    "deepseek2-ocr is a document parser - pass its vision mmproj \
                     (--mmproj mmproj-*.gguf)"
                        .to_string()
                })?;
                let exec = make_exec(pack)?;
                let mut model = paddock_engine::gpu_model::deepseek_ocr::GpuDeepseekOcr::load(
                    exec, &map, max_ctx,
                )
                .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                let mmap = MappedGguf::open(mp).map_err(|e| e.to_string())?;
                model.attach_vision(&mmap).map_err(|e| e.to_string())?;
                Ok(Box::new(model))
            }
            other => Err(format!("deepseek2-ocr needs cuda (got {other:?})")),
        },
        // PaddleOCR-VL 1.6 (`paddleocr`) - the Nordic/multilingual document
        // parser. The mmproj (NaViT tower + projector) is required for the
        // same reason deepseek2-ocr's is: text-only completion is not what
        // anyone deploys an OCR element recognizer for.
        "paddleocr" => match device {
            "cuda" => {
                let mp = mmproj.ok_or_else(|| {
                    "paddleocr is a document parser - pass its vision mmproj \
                     (--mmproj *mmproj*.gguf)"
                        .to_string()
                })?;
                let exec = make_exec(pack)?;
                let mut model = paddock_engine::gpu_model::paddleocr_vl::GpuPaddleOcrVl::load(
                    exec, &map, max_ctx,
                )
                .map_err(|e| e.to_string())?;
                apply_kv_dtype(|d| model.set_kv_dtype(d));
                let mmap = MappedGguf::open(mp).map_err(|e| e.to_string())?;
                model.attach_vision(&mmap).map_err(|e| e.to_string())?;
                Ok(Box::new(model))
            }
            other => Err(format!("paddleocr needs cuda (got {other:?})")),
        },
        other => Err(ServeError::UnsupportedArch(other.to_owned()).to_string()),
    }
}
