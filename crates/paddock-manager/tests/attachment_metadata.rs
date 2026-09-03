//! `GET /api/attachments/{id}/metadata` - the manager answers this itself.
//!
//! The point is what this test's *setup* leaves out: there is no
//! runner here, no model, no GPU. Before it, a file's metadata came from a
//! runner's `/api/extract`, so a photo's capture time was unavailable whenever
//! nothing was loaded - and on a cloud-model chat, by design, always. Nothing
//! about EXIF depends on which model is up, so nothing here starts one.

use std::sync::Arc;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use paddock_manager::routes::{AppState, router};
use tower::ServiceExt;

/// Minimal JPEG carrying one EXIF tag (Make = TestCam) - a fixture file would
/// only hide what is being asserted.
fn exif_jpeg() -> Vec<u8> {
    let tiff: Vec<u8> = [
        &[0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00][..], // II*\0, IFD0 @8
        &[0x01, 0x00][..],                                     // 1 entry
        // tag 0x010F (Make), ASCII, count 8, value @ offset 26
        &[
            0x0F, 0x01, 0x02, 0x00, 0x08, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00,
        ][..],
        &[0x00, 0x00, 0x00, 0x00][..], // no next IFD
        b"TestCam\0",
    ]
    .concat();
    let mut jpeg = vec![0xFF, 0xD8]; // SOI
    jpeg.extend_from_slice(&[0xFF, 0xE1]); // APP1
    let len = (2 + 6 + tiff.len()) as u16;
    jpeg.extend_from_slice(&len.to_be_bytes());
    jpeg.extend_from_slice(b"Exif\0\0");
    jpeg.extend_from_slice(&tiff);
    jpeg.extend_from_slice(&[0xFF, 0xD9]); // EOI
    jpeg
}

/// A JPEG whose EXIF is nothing but a GPS IFD - 43°28'03" N, 11°53'06" E,
/// the Tuscany photo rounded to whole seconds.
fn gps_jpeg() -> Vec<u8> {
    let r3 = |a: u32, b: u32, c: u32| -> Vec<u8> {
        [a, 1, b, 1, c, 1]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    };
    let entry = |tag: u16, ty: u16, count: u32, val: [u8; 4]| -> Vec<u8> {
        [
            &tag.to_le_bytes()[..],
            &ty.to_le_bytes()[..],
            &count.to_le_bytes()[..],
            &val[..],
        ]
        .concat()
    };
    let tiff: Vec<u8> = [
        &[0x49, 0x49, 0x2A, 0x00, 0x08, 0x00, 0x00, 0x00][..], // II*\0, IFD0 @8
        &[0x01, 0x00][..],
        &entry(0x8825, 4, 1, 26u32.to_le_bytes())[..], // GPSInfo -> GPS IFD @26
        &[0x00, 0x00, 0x00, 0x00][..],
        &[0x04, 0x00][..],
        &entry(0x0001, 2, 2, *b"N\0\0\0")[..],
        &entry(0x0002, 5, 3, 80u32.to_le_bytes())[..],
        &entry(0x0003, 2, 2, *b"E\0\0\0")[..],
        &entry(0x0004, 5, 3, 104u32.to_le_bytes())[..],
        &[0x00, 0x00, 0x00, 0x00][..],
        &r3(43, 28, 3)[..],
        &r3(11, 53, 6)[..],
    ]
    .concat();
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
    jpeg.extend_from_slice(&((2 + 6 + tiff.len()) as u16).to_be_bytes());
    jpeg.extend_from_slice(b"Exif\0\0");
    jpeg.extend_from_slice(&tiff);
    jpeg.extend_from_slice(&[0xFF, 0xD9]);
    jpeg
}

async fn get(app: axum::Router, uri: &str) -> (StatusCode, serde_json::Value) {
    let resp = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .expect("router responds");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .expect("body");
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, json)
}

#[tokio::test]
async fn a_stored_photo_reports_its_metadata_with_nothing_running() {
    let state = Arc::new(AppState::for_tests());
    state
        .db
        .put_attachment(
            "a1",
            None,
            "image/jpeg",
            "holiday.jpg",
            None,
            None,
            &exif_jpeg(),
        )
        .expect("store");

    let (status, body) = get(router(state), "/api/attachments/a1/metadata").await;

    assert_eq!(status, StatusCode::OK, "got {body}");
    // The stored identity rides along so one call answers the whole question.
    assert_eq!(body["name"], "holiday.jpg");
    assert_eq!(body["mime"], "image/jpeg");
    assert!(body["size"].as_u64().unwrap_or(0) > 0);
    // `format` is what the BYTES say, never the declared mime.
    assert_eq!(body["format"], "JPEG");
    assert_eq!(body["reader"], "sift");
    let make = body["groups"]
        .as_array()
        .expect("groups")
        .iter()
        .find(|g| g["name"] == "EXIF")
        .and_then(|g| g["tags"].as_array())
        .and_then(|tags| tags.iter().find(|t| t["name"] == "Make"))
        .map(|t| t["value"].clone());
    assert_eq!(make, Some(serde_json::json!("TestCam")), "got {body}");
}

#[tokio::test]
async fn a_file_with_no_metadata_answers_empty_rather_than_failing() {
    // A screenshot, a code file, a stripped export: silence is an ordinary
    // answer, and a 4xx here would make the panel look broken.
    let state = Arc::new(AppState::for_tests());
    state
        .db
        .put_attachment("a2", None, "text/plain", "notes.txt", None, None, b"hello")
        .expect("store");

    let (status, body) = get(router(state), "/api/attachments/a2/metadata").await;

    assert_eq!(status, StatusCode::OK, "got {body}");
    assert_eq!(body["format"], serde_json::Value::Null);
    assert_eq!(body["reader"], "none");
    assert_eq!(body["groups"].as_array().map(Vec::len), Some(0));
}

#[tokio::test]
async fn a_geotagged_photo_carries_numbers_and_a_place_name() {
    // The pin the Studio draws needs two f64s, and the place name is resolved
    // here, offline, so that looking at your own photo's location never asks
    // anyone where you were. Both come off the manager with no
    // runner, which is the whole point of answering metadata here.
    let state = Arc::new(AppState::for_tests());
    state
        .db
        .put_attachment(
            "a3",
            None,
            "image/jpeg",
            "tuscany.jpg",
            None,
            None,
            &gps_jpeg(),
        )
        .expect("store");

    let (status, body) = get(router(state), "/api/attachments/a3/metadata").await;

    assert_eq!(status, StatusCode::OK, "got {body}");
    let lat = body["location"]["latitude"].as_f64().expect("latitude");
    let lon = body["location"]["longitude"].as_f64().expect("longitude");
    assert!((lat - 43.4675).abs() < 1e-4, "got {body}");
    assert!((lon - 11.885).abs() < 1e-4, "got {body}");
    assert_eq!(body["location"]["place"]["city"], "Arezzo", "got {body}");
    assert_eq!(
        body["location"]["place"]["description"], "in Arezzo (Tuscany, Italy)",
        "the phrase is written once and shared with the prompt line: {body}"
    );
}

#[tokio::test]
async fn a_photo_without_gps_sends_no_location_at_all() {
    // Absent, not null and not a zero coordinate off West Africa - the pane
    // decides whether to draw a map from this field's presence.
    let state = Arc::new(AppState::for_tests());
    state
        .db
        .put_attachment(
            "a4",
            None,
            "image/jpeg",
            "holiday.jpg",
            None,
            None,
            &exif_jpeg(),
        )
        .expect("store");

    let (_, body) = get(router(state), "/api/attachments/a4/metadata").await;
    assert!(body.get("location").is_none(), "got {body}");
}

#[tokio::test]
async fn an_unknown_attachment_is_a_404_not_an_empty_answer() {
    // The distinction matters to the UI: "this file says nothing" and "there
    // is no such file" are different problems with different fixes.
    let (status, body) = get(
        router(Arc::new(AppState::for_tests())),
        "/api/attachments/nope/metadata",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "got {body}");
}
