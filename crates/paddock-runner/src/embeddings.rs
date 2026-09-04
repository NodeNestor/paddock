//! `POST /v1/embeddings` (OpenAI) and `POST /v1/rerank` (Cohere/Jina-style),
//! served by an encoder-only model (Qwen3 dense) - the private-RAG surface.
//! Both run the whole request's batch through one weight-amortized encode
//! (the throughput lever), then pool: embeddings return L2-normalized vectors;
//! rerank returns sorted relevance scores.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use base64::Engine;
use paddock_api::ErrorBody;
use paddock_api::embeddings::{
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingValue,
    RerankRequest, RerankResponse, RerankResult,
};
use serde_json::Value;

use crate::extract::OaiJson;
use crate::routes::AppState;
use crate::serving::EmbedModel;

fn err(status: StatusCode, kind: &str, msg: impl Into<String>) -> Response {
    (status, Json(ErrorBody::new(kind, msg))).into_response()
}

/// Base64 of the little-endian f32 bytes (the OpenAI `encoding_format:
/// "base64"` form). The f32 slice is its own little-endian byte image on the
/// x86/CUDA target, so cast it directly - no per-float copy or text formatting.
fn encode_base64(v: &[f32]) -> EmbeddingValue {
    // SAFETY: reinterpret contiguous [f32] as [u8]; target is little-endian.
    let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), v.len() * 4) };
    EmbeddingValue::Base64(base64::engine::general_purpose::STANDARD.encode(bytes))
}

// the Err is the finished axum reply on a cold error path; boxing it buys nothing
#[allow(clippy::result_large_err)]
fn model_or_400(state: &AppState) -> Result<&EmbedModel, Response> {
    state.embedder.as_ref().ok_or_else(|| {
        err(
            StatusCode::SERVICE_UNAVAILABLE,
            "model_not_loaded",
            "this server has no embedding/rerank model loaded",
        )
    })
}

/// Parse the OpenAI `input` (string | [string] | [int] | [[int]]) into token
/// sequences. Text inputs are tokenized and get an appended EOS so last-token
/// pooling reads it; pre-tokenized inputs are used verbatim.
fn parse_input(input: &Value, m: &EmbedModel) -> Result<Vec<Vec<u32>>, String> {
    let tok_text = |s: &str| -> Result<Vec<u32>, String> {
        let mut e = m.tokenizer.encode(s).map_err(|e| e.to_string())?;
        if let Some(eos) = m.eos {
            e.push(eos);
        }
        Ok(e)
    };
    match input {
        Value::String(s) => Ok(vec![tok_text(s)?]),
        Value::Array(items) if items.is_empty() => Err("input is empty".into()),
        Value::Array(items) => {
            // [int...] = one pre-tokenized sequence
            if items.iter().all(Value::is_u64) {
                let seq: Vec<u32> = items
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect();
                return Ok(vec![seq]);
            }
            // batch requests tokenize in parallel: serial BPE over a few
            // hundred texts costs ~10 ms/request, a visible slice of a
            // batched-embed round trip
            if items.len() >= 32 && items.iter().all(Value::is_string) {
                let texts: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
                let chunk = texts.len().div_ceil(16);
                let results: Vec<Result<Vec<Vec<u32>>, String>> = std::thread::scope(|s| {
                    let handles: Vec<_> = texts
                        .chunks(chunk)
                        .map(|c| s.spawn(|| c.iter().map(|t| tok_text(t)).collect()))
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().expect("tokenize thread"))
                        .collect()
                });
                let mut out = Vec::with_capacity(texts.len());
                for r in results {
                    out.extend(r?);
                }
                return Ok(out);
            }
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                match it {
                    Value::String(s) => out.push(tok_text(s)?),
                    Value::Array(ids) => out.push(
                        ids.iter()
                            .filter_map(|v| v.as_u64().map(|n| n as u32))
                            .collect(),
                    ),
                    _ => return Err("input array must be strings or token-id arrays".into()),
                }
            }
            Ok(out)
        }
        _ => Err("input must be a string, an array of strings, or token ids".into()),
    }
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    OaiJson(req): OaiJson<EmbeddingRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    let m = match model_or_400(&state) {
        Ok(m) => m,
        Err(resp) => return resp,
    };
    scope.model(&m.id);
    scope.user(req.user.as_deref());
    let base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(_) => {
            return err(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "encoding_format must be \"float\" or \"base64\"",
            );
        }
    };
    if req.dimensions.is_some() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "the `dimensions` (Matryoshka) parameter is not supported yet",
        );
    }
    let seqs = match parse_input(&req.input, m) {
        Ok(s) => s,
        Err(e) => return err(StatusCode::BAD_REQUEST, "invalid_request_error", e),
    };
    let prompt_tokens: usize = seqs.iter().map(Vec::len).sum();
    scope.usage(prompt_tokens, 0);
    match m.encoder.embed(seqs).await {
        Ok(vectors) => {
            let data = vectors
                .into_iter()
                .enumerate()
                .map(|(index, v)| EmbeddingData {
                    object: "embedding",
                    embedding: if base64 {
                        encode_base64(&v)
                    } else {
                        EmbeddingValue::Floats(v)
                    },
                    index,
                })
                .collect();
            Json(EmbeddingResponse {
                object: "list",
                data,
                model: req.model,
                usage: EmbeddingUsage {
                    prompt_tokens,
                    total_tokens: prompt_tokens,
                },
            })
            .into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
    }
}

pub(crate) const DEFAULT_INSTRUCT: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// The Qwen3-Reranker prompt: a yes/no relevance judge over (instruct, query,
/// document). The score is P(yes) at the final position.
pub(crate) fn rerank_prompt(instruction: &str, query: &str, doc: &str) -> String {
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements \
         based on the Query and the Instruct provided. Note that the answer can \
         only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: \
         {instruction}\n<Query>: {query}\n<Document>: {doc}<|im_end|>\n\
         <|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

pub async fn rerank(
    State(state): State<Arc<AppState>>,
    scope: Option<axum::Extension<crate::events::EventScope>>,
    OaiJson(req): OaiJson<RerankRequest>,
) -> Response {
    let scope = scope.map(|e| e.0).unwrap_or_default();
    let m = match model_or_400(&state) {
        Ok(m) => m,
        Err(resp) => return resp,
    };
    scope.model(&m.id);
    let (Some(yes), Some(no)) = (m.yes_id, m.no_id) else {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "this model's vocabulary lacks the yes/no tokens a reranker needs",
        );
    };
    if req.documents.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "documents must not be empty",
        );
    }
    let instruction = req.instruction.as_deref().unwrap_or(DEFAULT_INSTRUCT);
    let t_tok = std::time::Instant::now();
    // rerank prompts are long (~120+ tokens); tokenize a batch in parallel
    // like the embeddings handler - serial BPE here cost ~35 ms of a
    // 128-doc round trip
    let chunk = req.documents.len().div_ceil(16).max(1);
    let results: Vec<Result<Vec<Vec<u32>>, String>> = std::thread::scope(|s| {
        let handles: Vec<_> = req
            .documents
            .chunks(chunk)
            .map(|docs| {
                s.spawn(|| {
                    docs.iter()
                        .map(|doc| {
                            m.tokenizer
                                .encode(&rerank_prompt(instruction, &req.query, doc))
                                .map_err(|e| e.to_string())
                        })
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("tokenize thread"))
            .collect()
    });
    let mut seqs = Vec::with_capacity(req.documents.len());
    for r in results {
        match r {
            Ok(batch) => seqs.extend(batch),
            Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
        }
    }
    let prompt_tokens: usize = seqs.iter().map(Vec::len).sum();
    scope.usage(prompt_tokens, 0);
    let tok_ms = t_tok.elapsed().as_secs_f64() * 1e3;
    let t_enc = std::time::Instant::now();
    let timing = paddock_models::dev_var_os!("PADDOCK_TIME_HANDLER").is_some();
    match m.encoder.rerank(seqs, yes, no).await {
        Ok(scores) => {
            let mut results: Vec<RerankResult> = scores
                .into_iter()
                .enumerate()
                .map(|(index, relevance_score)| RerankResult {
                    index,
                    relevance_score,
                    document: req.return_documents.then(|| req.documents[index].clone()),
                })
                .collect();
            if timing {
                tracing::info!(
                    "  [handler-timing] tokenize {tok_ms:.1} ms | encode await {:.1} ms",
                    t_enc.elapsed().as_secs_f64() * 1e3
                );
            }
            results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
            if let Some(n) = req.top_n {
                results.truncate(n);
            }
            Json(RerankResponse {
                model: req.model,
                results,
                usage: EmbeddingUsage {
                    prompt_tokens,
                    total_tokens: prompt_tokens,
                },
            })
            .into_response()
        }
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", e),
    }
}
