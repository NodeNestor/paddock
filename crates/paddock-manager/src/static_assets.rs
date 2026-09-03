//! Serve the built Studio SPA, embedded into the binary via `rust-embed`.
//!
//! Release builds embed `crates/paddock-manager/static/`; debug builds read it
//! from disk (keyed off `CARGO_MANIFEST_DIR`, so no CWD dependency and a
//! `vite build` needs no recompile). This is the router fallback: unmatched
//! API paths still return the OpenAI-shaped JSON 404 (SDKs parse those), and
//! every other unmatched path falls back to `index.html` so the SPA's own
//! client-side routing works on refresh/deep-link.

use axum::http::{HeaderMap, HeaderValue, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use rust_embed::RustEmbed;

#[derive(RustEmbed)]
#[folder = "static/"]
struct Studio;

pub(crate) async fn serve(uri: Uri, headers: HeaderMap) -> Response {
    let raw = uri.path().trim_start_matches('/');
    // Real API misses stay JSON - never the SPA shell.
    if raw.starts_with("v1/") || raw.starts_with("api/") || raw == "healthz" {
        return crate::routes::not_found(uri).await;
    }
    let path = if raw.is_empty() { "index.html" } else { raw };
    asset(path, &headers)
        .or_else(|| asset("index.html", &headers)) // unknown path -> SPA shell
        .unwrap_or_else(|| {
            // No Studio compiled into this binary (built without `studio/` output).
            (
                StatusCode::NOT_FOUND,
                "Studio UI is not built into this binary",
            )
                .into_response()
        })
}

fn asset(path: &str, headers: &HeaderMap) -> Option<Response> {
    let accepts_gzip = headers
        .get(header::ACCEPT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|s| s.contains("gzip"));

    // Serve the precompressed sibling when the client accepts gzip.
    if accepts_gzip && let Some(gz) = Studio::get(&format!("{path}.gz")) {
        let mut resp = axum::body::Bytes::from(gz.data.into_owned()).into_response();
        let h = resp.headers_mut();
        h.insert(header::CONTENT_TYPE, mime_of(path));
        h.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
        h.insert(header::CACHE_CONTROL, cache_control(path));
        return Some(resp);
    }

    let file = Studio::get(path)?;
    let mut resp = axum::body::Bytes::from(file.data.into_owned()).into_response();
    let h = resp.headers_mut();
    h.insert(header::CONTENT_TYPE, mime_of(path));
    h.insert(header::CACHE_CONTROL, cache_control(path));
    Some(resp)
}

/// MIME type from the base (uncompressed) path - the `.gz` sibling shares it.
fn mime_of(path: &str) -> HeaderValue {
    Studio::get(path)
        .and_then(|f| HeaderValue::from_str(f.metadata.mimetype()).ok())
        .unwrap_or_else(|| HeaderValue::from_static("application/octet-stream"))
}

fn cache_control(path: &str) -> HeaderValue {
    // Vite emits content-hashed filenames under assets/ - cache them forever;
    // the HTML shell (and anything else) must revalidate.
    if path.starts_with("assets/") {
        HeaderValue::from_static("public, max-age=31536000, immutable")
    } else {
        HeaderValue::from_static("no-cache")
    }
}
