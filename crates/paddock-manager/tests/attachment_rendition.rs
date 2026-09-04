//! `GET /api/attachments/{id}/rendition` - a viewable JPEG of a photo the
//! browser cannot decode itself.
//!
//! Like the metadata twin next door, the point is what the setup leaves out:
//! no runner, no model, no GPU. A photo has to preview on a cloud chat with
//! nothing loaded, so nothing here starts anything.
//!
//! These assert UNCONDITIONALLY. An earlier version accepted "no decoder
//! installed" as a pass on every case, which meant a machine without the
//! native pack showed green tests that had decoded nothing. There is no pack
//! now - rav1d is linked in - so there is no such state and no branch to hide
//! in. HEIC's refusal is asserted just as firmly, because it is permanent
//! rather than conditional.
//!
//! Fixtures are the ones paddock-heif tests with, reached across the crate
//! boundary rather than copied: duplicating them would let the two suites
//! drift onto different files while claiming to test the same decode.
// Test code: a failed assumption stops the test where it happened.
#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use paddock_manager::routes::{AppState, router};
use tower::ServiceExt;

const HEVC32: &[u8] = include_bytes!("../../paddock-heif/tests/data/hevc32.heif");
const AVIF32: &[u8] = include_bytes!("../../paddock-heif/tests/data/avif32.heif");

struct Reply {
    status: StatusCode,
    mime: String,
    body: Vec<u8>,
}

impl Reply {
    /// The response as a decoded image. Only valid on a 200.
    fn image(&self) -> image::DynamicImage {
        image::load_from_memory(&self.body).expect("the rendition must be a decodable image")
    }
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

async fn get(app: axum::Router, uri: &str) -> Reply {
    let resp = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .expect("router responds");
    let status = resp.status();
    let mime = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .to_owned();
    let body = axum::body::to_bytes(resp.into_body(), 32 << 20)
        .await
        .expect("body")
        .to_vec();
    Reply { status, mime, body }
}

fn store(id: &str, name: &str, mime: &str, bytes: &[u8]) -> Arc<AppState> {
    let state = Arc::new(AppState::for_tests());
    state
        .db
        .put_attachment(id, None, mime, name, None, None, bytes)
        .expect("store");
    state
}

#[tokio::test]
async fn an_avif_becomes_a_jpeg_anything_can_show() {
    let state = store("h1", "shot.avif", "image/avif", AVIF32);
    let r = get(router(state), "/api/attachments/h1/rendition").await;

    assert_eq!(r.status, StatusCode::OK, "got {}", r.text());
    assert_eq!(r.mime, "image/jpeg");
    let img = r.image();
    // Smaller than the default max, so it comes back at its own size - the
    // endpoint only ever scales DOWN.
    assert_eq!((img.width(), img.height()), (32, 32));
}

/// HEIC is HEVC, and no HEVC decoder can be embedded in a closed binary
/// without publishing relinkable object code. So this is
/// permanent, and the message must not read as "something is missing here" -
/// there is nothing the reader could install.
#[tokio::test]
async fn a_heic_is_refused_because_hevc_cannot_be_decoded() {
    let state = store("h2", "IMG_5195.HEIC", "image/heic", HEVC32);
    let r = get(router(state), "/api/attachments/h2/rendition").await;

    assert_eq!(r.status, StatusCode::NOT_IMPLEMENTED, "got {}", r.text());
    let t = r.text();
    assert!(
        t.contains("HEIC") && t.contains("HEVC"),
        "must name both: {t}"
    );
    assert!(
        !t.to_lowercase().contains("install"),
        "no fix to suggest: {t}"
    );
    // ...and it should say what is still true, so the panel can show something
    // useful rather than only an apology.
    assert!(t.contains("stored intact"), "{t}");
}

/// The browser leaves `File.type` empty for these often enough that the stored
/// mime is the one field this endpoint cannot trust. It sniffs the bytes.
#[tokio::test]
async fn the_stored_mime_is_not_believed() {
    let state = store("h3", "shot.avif", "application/octet-stream", AVIF32);
    let r = get(router(state), "/api/attachments/h3/rendition").await;
    assert_eq!(r.status, StatusCode::OK, "got {}", r.text());
}

#[tokio::test]
async fn max_bounds_the_longest_edge_and_never_upscales() {
    let state = store("h4", "shot.avif", "image/avif", AVIF32);
    let app = router(state);

    // Asking for more than the photo has must not invent pixels.
    let big = get(app.clone(), "/api/attachments/h4/rendition?max=2048").await;
    assert_eq!(big.status, StatusCode::OK, "got {}", big.text());
    assert_eq!(
        big.image().width(),
        32,
        "a 32px photo must not become 2048px"
    );

    let small = get(app, "/api/attachments/h4/rendition?max=16").await;
    assert_eq!(small.status, StatusCode::OK);
    let img = small.image();
    assert!(
        img.width().max(img.height()) <= 16,
        "{}x{}",
        img.width(),
        img.height()
    );
}

/// A JPEG has no business here - the client already has a copy it can show,
/// and answering would quietly become a general re-encoding service.
#[tokio::test]
async fn a_file_that_is_not_heif_is_refused_rather_than_re_encoded() {
    let state = store(
        "h5",
        "holiday.jpg",
        "image/jpeg",
        b"\xFF\xD8\xFF\xE0\0\x10JFIF\0",
    );
    let r = get(router(state), "/api/attachments/h5/rendition").await;
    assert_eq!(
        r.status,
        StatusCode::UNSUPPORTED_MEDIA_TYPE,
        "got {}",
        r.text()
    );
    assert!(r.text().contains("fetch it directly"), "{}", r.text());
}

/// Truncation is the common real failure. It must be an error about this photo,
/// not a 500 and not a zero-byte image. (avif-parse panics on a short file in
/// debug builds; paddock-heif catches that, and this is the case that proves
/// the catch is wired up from out here.)
#[tokio::test]
async fn a_truncated_photo_is_a_422_about_the_photo() {
    let state = store("h6", "half.avif", "image/avif", &AVIF32[..AVIF32.len() / 2]);
    let r = get(router(state), "/api/attachments/h6/rendition").await;
    assert_eq!(
        r.status,
        StatusCode::UNPROCESSABLE_ENTITY,
        "got {} {}",
        r.status,
        r.text()
    );
    assert!(
        r.text().contains("AVIF"),
        "the error must name the format: {}",
        r.text()
    );
}

#[tokio::test]
async fn an_unknown_attachment_is_a_404() {
    let r = get(
        router(Arc::new(AppState::for_tests())),
        "/api/attachments/nope/rendition",
    )
    .await;
    assert_eq!(r.status, StatusCode::NOT_FOUND);
}

/// The stored bytes are the record; a rendition is a copy for looking at. If
/// this ever fails, metadata has started being read from a re-encode - the
/// EXIF-loss regression, arriving by a different door.
#[tokio::test]
async fn rendering_never_touches_what_was_stored() {
    let state = store("h7", "shot.avif", "image/avif", AVIF32);
    let _ = get(
        router(state.clone()),
        "/api/attachments/h7/rendition?max=16",
    )
    .await;

    let (mime, stored) = state
        .db
        .get_attachment("h7")
        .expect("read back")
        .expect("present");
    assert_eq!(stored, AVIF32, "the original bytes must be byte-identical");
    assert_eq!(mime, "image/avif");
}
