//! "Send feedback" - the Studio's dialog, forwarded to the truespar API.
//!
//! ## Why the manager forwards instead of the browser posting directly
//!
//! The Studio could `fetch()` `api.truespar.com` itself, and traverse's WASM
//! build does exactly that because it has no server to forward through. We do,
//! and going through it is strictly better: the manager is already the only
//! thing here that talks to that host (`updates.rs`), so there is one place
//! where outbound calls live and one place to read to find out what leaves the
//! box. It also sidesteps CORS entirely - the Studio is served from
//! `localhost:<port>`, which has no business being on anybody's allow-list.
//!
//! ## The context blob is assembled here, deliberately
//!
//! Traverse's panel assembles its "attach last query" payload in the browser.
//! We don't, for two reasons. The manager is the only process that actually
//! knows all of it (the GPU probe, the fleet, the build stamp), and - the real
//! reason - `GET /api/feedback/context` returns the same struct the POST
//! attaches. The preview the user approves is not a rendering of what we will
//! send; it is what we will send. A preview that can drift from the payload is
//! how a promise about telemetry quietly stops being true.
//!
//! ## What must never end up in here
//!
//! Runner config carries an inference `api_key` AND a `web_search_api_key`, and
//! a model can be spawned by absolute path - so a naive `serde_json::to_value`
//! of a `RunnerConfig` would ship two secrets and a user's directory layout to
//! a server. [`Context`] is therefore built field by field from an explicit
//! allow-list, never by serialising an existing struct. Keep it that way: the
//! failure mode of adding a field is silent.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Json, Response};
use serde::{Deserialize, Serialize};

use crate::routes::AppState;

/// Which product this is, for the API's `app_identifier` column.
///
/// The column exists but its handler does not yet bind it (it defaults to
/// `'traverse'`), so today this field is accepted and discarded - see
/// `truespar-core/`. Sending it now means
/// nothing changes here the day that lands.
const APP_ID: &str = "paddock";

/// The API's own cap (`feedback.rs`), mirrored rather than left to the server.
///
/// Not redundant validation: an anonymous IP gets five submissions an hour, and
/// spending one to be told the message was too long is a bad trade when the
/// answer is knowable locally and instantly.
const MAX_MESSAGE: usize = 10_000;

/// Validated exact and case-sensitive upstream, so send exactly these.
const CATEGORIES: [&str; 3] = ["bug", "feature", "feedback"];

/// What the Studio posts.
#[derive(Debug, Deserialize)]
pub struct Submission {
    pub category: String,
    pub message: String,
    #[serde(default)]
    pub email: Option<String>,
    /// Attach [`Context`]. Defaults false - the diagnostics go only when
    /// somebody has seen them and said yes, and a client that forgets the field
    /// must get the private behaviour, not the chatty one.
    #[serde(default)]
    pub include_context: bool,
}

/// The diagnostic blob: what this machine is, and what it is running.
///
/// Everything here is stuff a bug report would otherwise ask the user to go
/// and find. Nothing here identifies a person.
#[derive(Debug, Clone, Serialize)]
pub struct Context {
    pub manager: ManagerInfo,
    pub gpu: GpuInfo,
    pub runners: Vec<RunnerInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ManagerInfo {
    /// Bare SemVer, matching what `/api/server` reports.
    pub version: &'static str,
    /// The long stamp with the commit - what actually pins down a build.
    pub build: &'static str,
    pub os: &'static str,
    pub arch: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    /// The readiness verdict ("ready" | "untested" | "driver-too-old" |
    /// "needs-setup" | "no-card"). Half of every "it won't start" report is
    /// answered by this line alone.
    pub state: crate::readiness::State,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub card: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
    /// The CUDA version the driver speaks, and the one this build needs - a
    /// pair, because either alone is unactionable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cuda: Option<String>,
    pub cuda_needed: String,
    // `libraries_installed` lived here, for a box past setup but missing the
    // fetched CUDA maths after a half-finished download. There is no such
    // state: paddock ships and fetches no NVIDIA redistributable, so `state`
    // says everything there is to say.
}

/// One running model, described by what it is rather than where it came from.
#[derive(Debug, Clone, Serialize)]
pub struct RunnerInfo {
    /// The catalog's display name when we have one, else the model id with any
    /// path stripped. Never a filesystem path - see the module note.
    pub model: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Weights artifact as deployed ("q8", "q4", ...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_cache_dtype: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_ctx: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_batch: Option<usize>,
    /// Speculation policy ("off" | "auto" | "ladder" | "<k>").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spec: Option<String>,
}

/// A model id with any directory part removed.
///
/// A spawn takes "a catalog id, an installed model name, or a GGUF path"
/// (`SpawnSpec`), so the third case would otherwise put `C:\Users\<name>\...` in
/// an outbound payload. Handles both separators regardless of host OS: a config
/// written on one box can be read on another.
fn scrub_model(id: &str) -> String {
    id.rsplit(['/', '\\']).next().unwrap_or(id).to_string()
}

/// Assemble the blob. Also the preview - there is only this one implementation.
pub async fn context(state: &AppState) -> Context {
    let readiness = (*state.readiness).clone();

    let runners = state
        .supervisor
        .list()
        .await
        .into_iter()
        .map(|v| {
            let cfg = v.config.as_ref();
            RunnerInfo {
                // Prefer the catalog's human name: it is both friendlier and
                // guaranteed path-free. The scrub is the fallback's job.
                model: v.display.clone().unwrap_or_else(|| {
                    scrub_model(
                        v.model
                            .as_deref()
                            .or(v.embedder.as_deref())
                            .or(v.asr.as_deref())
                            .unwrap_or("unknown"),
                    )
                }),
                status: v.status.clone(),
                version: v.version.clone(),
                artifact: cfg.and_then(|c| c.artifact.clone()),
                kv_cache_dtype: cfg.and_then(|c| c.kv_cache_dtype.clone()),
                max_ctx: cfg.and_then(|c| c.max_ctx),
                max_batch: cfg.and_then(|c| c.max_batch),
                spec: cfg.and_then(|c| c.spec.clone()),
            }
        })
        .collect();

    Context {
        manager: ManagerInfo {
            version: paddock_admin::version::SEMVER,
            build: paddock_admin::version::LONG,
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
        },
        gpu: GpuInfo {
            state: readiness.state,
            card: readiness.card.clone(),
            generation: readiness.generation.clone(),
            driver: readiness.driver.clone(),
            cuda: readiness.cuda.clone(),
            cuda_needed: readiness.cuda_needed.clone(),
        },
        runners,
    }
}

/// GET /api/feedback/context - what the dialog previews, verbatim.
pub async fn preview(State(state): State<Arc<AppState>>) -> Response {
    Json(context(&state).await).into_response()
}

/// POST /api/feedback - validate, stamp, forward.
pub async fn submit(State(state): State<Arc<AppState>>, Json(body): Json<Submission>) -> Response {
    let category = body.category.trim();
    if !CATEGORIES.contains(&category) {
        return bad_request("category must be bug, feature or feedback");
    }
    let message = body.message.trim();
    if message.is_empty() {
        return bad_request("message is required");
    }
    // Count CHARS, not bytes: the upstream limit is a .NET string length, so a
    // message of accented text would otherwise be refused here at a length the
    // server would have accepted.
    if message.chars().count() > MAX_MESSAGE {
        return bad_request(&format!("message must be {MAX_MESSAGE} characters or less"));
    }

    let mut payload = serde_json::json!({
        "appIdentifier": APP_ID,
        "category": category,
        "message": message,
        // The LONG stamp, not the bare SemVer traverse sends. A bug report is
        // the one place the exact commit beats the marketing version.
        "appVersion": paddock_admin::version::LONG,
        // `{os}-{arch}`, shared with the update path. A bare
        // `std::env::consts::OS` is the shape traverse shipped and had to fix.
        "platform": crate::updates::release_platform(),
    });
    if let Some(email) = body
        .email
        .as_deref()
        .map(str::trim)
        .filter(|e| !e.is_empty())
    {
        payload["email"] = serde_json::Value::String(email.to_string());
    }
    if body.include_context {
        payload["context"] =
            serde_json::to_value(context(&state).await).unwrap_or(serde_json::Value::Null);
    }

    let url = format!(
        "{}/api/feedback",
        crate::updates::api_base().trim_end_matches('/')
    );
    let resp = crate::updates::http()
        .post(&url)
        .timeout(std::time::Duration::from_secs(20))
        .json(&payload)
        .send()
        .await;

    match resp {
        Ok(r) => {
            let status = r.status();
            // Relay the upstream body verbatim. The 429 in particular carries a
            // sentence about when to try again ("Feedback rate limit reached.
            // Please try again in an hour.") and a Retry-After - replacing that
            // with our own words would turn a rate limit into what reads like a
            // paddock failure.
            let text = r.text().await.unwrap_or_default();
            let parsed: serde_json::Value = serde_json::from_str(&text).unwrap_or_else(|_| {
                serde_json::json!({ "error": { "message": "feedback service returned an unreadable response" } })
            });
            if status.is_success() {
                Json(parsed).into_response()
            } else {
                tracing::warn!(%status, "feedback submission rejected upstream");
                (
                    StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
                    Json(parsed),
                )
                    .into_response()
            }
        }
        Err(e) => {
            // Offline, DNS, a timeout - all the same to the user, and none of
            // them mean their report was bad. Say plainly that it did not send,
            // so nobody believes a lost bug report was filed.
            tracing::warn!(error = %e, "feedback submission could not be sent");
            (
                StatusCode::BAD_GATEWAY,
                Json(serde_json::json!({
                    "error": {
                        "message": "could not reach the feedback service - check the connection and try again"
                    }
                })),
            )
                .into_response()
        }
    }
}

/// Shaped like the upstream 400 so the Studio has one error path, not two.
fn bad_request(message: &str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(serde_json::json!({ "error": { "message": message } })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_ids_lose_their_directory() {
        assert_eq!(
            scrub_model(r"C:\Users\someone\paddock\models\qwen3.5-9b-q8.gguf"),
            "qwen3.5-9b-q8.gguf"
        );
        assert_eq!(
            scrub_model("/home/someone/models/laguna.gguf"),
            "laguna.gguf"
        );
        // A catalog id has no separators and must survive untouched.
        assert_eq!(scrub_model("qwen3.5-9b"), "qwen3.5-9b");
    }

    /// The guard that matters. `RunnerConfig` carries an inference `api_key`, a
    /// `web_search_api_key` and a model string that may be an absolute path, so
    /// the one thing this module must never do is grow a field that copies any
    /// of them across.
    ///
    /// Pinned as an exact key set rather than "does not contain 'api_key'": a
    /// blocklist only catches the leaks somebody thought of, while this fails on
    /// any new field and makes whoever adds it look at this test and say why.
    #[test]
    fn runner_info_serialises_exactly_the_allow_list() {
        let row = RunnerInfo {
            model: "qwen3.5-9b".into(),
            status: "ok".into(),
            version: Some("0.1.0".into()),
            artifact: Some("q8".into()),
            kv_cache_dtype: Some("f16".into()),
            max_ctx: Some(32768),
            max_batch: Some(32),
            spec: Some("off".into()),
        };
        let json = serde_json::to_value(&row).expect("serialize");
        let mut keys: Vec<&str> = json
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(
            keys,
            [
                "artifact",
                "kv_cache_dtype",
                "max_batch",
                "max_ctx",
                "model",
                "spec",
                "status",
                "version",
            ],
            "RunnerInfo grew or lost a field - if it grew, confirm it cannot \
             carry a key or a filesystem path before updating this list"
        );
    }

    #[tokio::test]
    async fn context_assembles_on_a_box_with_no_runners() {
        // Not a leak test (there are no runners here to leak) - this covers the
        // empty fleet, which is what a first-run box looks like and the state a
        // "nothing starts" report is most likely to be filed from.
        let state = AppState::for_tests();
        let blob = context(&state).await;
        assert!(blob.runners.is_empty());
        assert_eq!(blob.manager.version, paddock_admin::version::SEMVER);
        assert!(!blob.gpu.cuda_needed.is_empty());
    }

    #[test]
    fn categories_match_what_the_api_validates() {
        // Exact and case-sensitive upstream; a capitalised value is a 400.
        assert!(CATEGORIES.contains(&"bug"));
        assert!(!CATEGORIES.contains(&"Bug"));
    }
}
