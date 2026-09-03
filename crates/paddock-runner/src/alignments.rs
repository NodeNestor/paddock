//! POST /v1/audio/alignments - forced alignment. A Paddock
//! surface: OpenAI has no alignment endpoint, so the shape borrows the one
//! thing worth borrowing - the `words` array is verbose_json's word shape
//! (`{word, start, end}` in seconds) so every consumer of transcription word
//! times reads this without a new parser.
//!
//! Multipart form: `file` (audio), `text` (the transcript to align),
//! optional `language`. The model addresses time in 80 ms classes over a
//! fixed bin budget (~6.7 min on the 0.6B); longer clips are refused loudly
//! with the cap in the message. Language rides two ways, both honest:
//! Japanese/Korean are REFUSED (the reference implementation aligns them
//! through morphological tokenizers this build does not carry, and the
//! default splitter would produce clause-sized "words"); everything else is
//! accepted, with `language_supported: false` in the response when it is
//! outside the model's trained eleven - the model may still align usefully
//! (that is a measurement), but the reader deserves the flag.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

use paddock_engine::audio::{decode::decode_audio, resample::resample};

use paddock_api::ErrorBody;

use crate::forced_align::{fix_timestamps, split_words};
use crate::routes::AppState;

/// The trained language set (transformers `FORCED_ALIGNER_LANGUAGES`), as
/// lowercase names + ISO codes. Outside-the-set requests still run - see the
/// module note - this list only feeds the honesty flag.
const SUPPORTED: &[(&str, &str)] = &[
    ("chinese", "zh"),
    ("english", "en"),
    ("cantonese", "yue"),
    ("french", "fr"),
    ("german", "de"),
    ("italian", "it"),
    ("japanese", "ja"),
    ("korean", "ko"),
    ("portuguese", "pt"),
    ("russian", "ru"),
    ("spanish", "es"),
];

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

pub async fn handle(State(state): State<Arc<AppState>>, mut mp: Multipart) -> Response {
    let Some(model) = state.aligner.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "no forced-alignment model is loaded (start paddock with the aligner checkpoint \
             directory as `model`)",
        );
    };

    let mut file: Option<Vec<u8>> = None;
    let mut text: Option<String> = None;
    let mut language: Option<String> = None;
    loop {
        let field = match mp.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    e.to_string(),
                );
            }
        };
        match field.name().unwrap_or_default().trim_end_matches("[]") {
            "file" => match field.bytes().await {
                Ok(b) => file = Some(b.to_vec()),
                Err(e) => {
                    return err(
                        StatusCode::BAD_REQUEST,
                        "invalid_request_error",
                        e.to_string(),
                    );
                }
            },
            "text" | "transcript" => {
                text = field.text().await.ok().filter(|s| !s.trim().is_empty())
            }
            "language" => language = field.text().await.ok().filter(|s| !s.is_empty()),
            // `model` is accepted-and-ignored like every single-model server
            _ => {}
        }
    }
    let Some(file) = file else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "missing `file` part",
        );
    };
    let Some(text) = text else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "missing `text` part - forced alignment aligns an EXISTING transcript; use \
             /v1/audio/transcriptions to produce one",
        );
    };

    let lang_norm = language.as_deref().map(str::to_lowercase);
    let supported = match lang_norm.as_deref() {
        None => true,
        Some(l) => SUPPORTED
            .iter()
            .any(|(name, code)| *name == l || *code == l),
    };
    if matches!(
        lang_norm.as_deref(),
        Some("ja" | "japanese" | "ko" | "korean")
    ) {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Japanese/Korean forced alignment needs a morphological word tokenizer this server \
             does not carry; the reference implementation uses nagisa/soynlp for these",
        );
    }

    let words = split_words(&text);
    if words.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`text` holds no alignable words after dropping punctuation",
        );
    }

    // decode + resample + mel off the async threads (transcriptions.rs's
    // pattern - a long WAV is real CPU work)
    let decoded = tokio::task::spawn_blocking(move || -> Result<_, String> {
        let wav = decode_audio(&file)?;
        if wav.samples.is_empty() {
            return Err("audio file holds no samples".into());
        }
        let samples = resample(&wav.samples, wav.sample_rate, 16000)?;
        let mel = paddock_engine::audio::qwen3_asr_features(&samples)?;
        let n_audio = paddock_engine::audio::audio_tokens_for_samples(samples.len());
        Ok((samples.len(), mel, n_audio))
    })
    .await;
    let (n_samples, mel, n_audio) = match decoded {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
        Err(e) => {
            return err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_error",
                e.to_string(),
            );
        }
    };
    let duration = n_samples as f64 / 16000.0;
    if duration > model.max_clip_s as f64 {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "clip is {duration:.1} s but this aligner addresses at most {:.0} s \
                 ({} bins of {} ms); align per segment instead",
                model.max_clip_s,
                (model.max_clip_s * 1000.0 / model.segment_ms) as u64,
                model.segment_ms
            ),
        );
    }

    // ── pack: <|audio_start|> pads... <|audio_end|> then per word: tokens + two
    // <timestamp> slots (the checkpoint's chat template, hard-coded because
    // it is fixed - words joined by the pair, one trailing pair) ──
    let mut ids: Vec<u32> = Vec::with_capacity(n_audio + 3 + words.len() * 4);
    ids.push(model.audio_start);
    let splice_at = ids.len();
    ids.extend(std::iter::repeat_n(model.audio_pad, n_audio));
    ids.push(model.audio_end);
    let mut ts_rows: Vec<usize> = Vec::with_capacity(words.len() * 2);
    for w in &words {
        match model.tokenizer.encode(w) {
            Ok(t) => ids.extend(t),
            Err(e) => {
                return err(
                    StatusCode::BAD_REQUEST,
                    "invalid_request_error",
                    e.to_string(),
                );
            }
        }
        ts_rows.push(ids.len());
        ids.push(model.timestamp);
        ts_rows.push(ids.len());
        ids.push(model.timestamp);
    }
    if ids.len() > model.max_ctx {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!(
                "packed sequence is {} rows ({} audio + transcript) but the server context is \
                 {}; align per segment instead",
                ids.len(),
                n_audio,
                model.max_ctx
            ),
        );
    }

    let bins = match model
        .aligner
        .align(paddock_engine::align::AlignReq {
            ids,
            mel,
            splice_at,
            n_audio,
            ts_rows,
        })
        .await
    {
        Ok(b) => b,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
    };

    // bins -> ms -> the reference's LIS monotonicity repair -> word pairs
    let raw_ms: Vec<f64> = bins
        .iter()
        .map(|&b| b as f64 * model.segment_ms as f64)
        .collect();
    let fixed = fix_timestamps(&raw_ms);
    let out: Vec<serde_json::Value> = words
        .iter()
        .enumerate()
        .map(|(k, w)| {
            // integer ms / 1000 is exactly the reference's round(·, 3)
            serde_json::json!({
                "word": w,
                "start": fixed[2 * k] as f64 / 1000.0,
                "end": fixed[2 * k + 1] as f64 / 1000.0,
            })
        })
        .collect();

    Json(serde_json::json!({
        "task": "alignment",
        "duration": duration,
        "language": language,
        "language_supported": supported,
        "words": out,
    }))
    .into_response()
}
