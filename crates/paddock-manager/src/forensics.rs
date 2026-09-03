//! Manager-side forensics: persist the reports the runner surfaced, and serve
//! them back to the Studio.
//!
//! The runner emits the structured forensic report as a `{type:"forensics"}`
//! output item on `/v1/responses` (see `paddock-runner`) - a STANDALONE API
//! capability that needs no manager. This module is the optional, fully
//! decoupled persistence half: the runner never calls the manager; the manager,
//! when it is in the loop, maps that report JSON to a [`NewForensicReport`] and
//! writes it to the one DB, then exposes it to the Studio.
//!
//! Deliberately dependency-free of `paddock-forensics`: the mapping reads plain
//! JSON `Value`s (the runner's `report` object), so the manager carries no
//! forensics-engine code - the same decoupling the store types already keep.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::routes::AppState;
use crate::store::{
    NewForensicExplanationCategory, NewForensicFinding, NewForensicKeyFinding, NewForensicReport,
};

/// The persist request the Studio sends when a `/v1/responses` reply carried a
/// forensics output item: the runner's `report` object plus the attachment
/// context. When `attachment_id` is given the manager resolves `sha256`, `mime`
/// and `name` from the stored bytes itself - the Studio should not have to hash
/// what it already handed us - so those fields are optional. For an ad-hoc
/// analysis with no stored attachment, `sha256` is required.
#[derive(serde::Deserialize, Default)]
pub struct PersistRequest {
    #[serde(default)]
    pub attachment_id: Option<String>,
    #[serde(default)]
    pub conversation_id: Option<String>,
    /// SHA-256 (hex) of the ORIGINAL analyzed bytes - the content key. Resolved
    /// from `attachment_id` when omitted.
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub mime: String,
    #[serde(default)]
    pub name: String,
    /// `"image"` or `"pdf"`; defaults to `"image"` (the item carries it).
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub gpu: bool,
    #[serde(default)]
    pub elapsed_ms: i64,
    /// The runner forensics item's `report` object.
    pub report: Value,
}

/// `POST /api/forensics` - persist one report the Studio received from a chat
/// turn. A write-through: the runner already computed it and returned it in the
/// response; this just lands it in the one DB.
///
/// Idempotent for a known attachment: its bytes are immutable, so the same
/// attachment always yields the same report - a re-post (re-render, re-chat of
/// identical bytes) returns the existing row rather than duplicating it.
pub async fn persist(
    State(state): State<Arc<AppState>>,
    Json(mut req): Json<PersistRequest>,
) -> Response {
    if let Some(att) = req.attachment_id.clone() {
        match state.db.get_attachment_named(&att) {
            Ok(Some((mime, name, bytes))) => {
                if req.sha256.is_empty() {
                    req.sha256 = sha256_hex(&bytes);
                }
                if req.mime.is_empty() {
                    req.mime = mime;
                }
                if req.name.is_empty() {
                    req.name = name;
                }
                // Already analyzed -> return the stored report, don't duplicate.
                if let Ok(Some(existing)) = state.db.latest_forensic_report_for_attachment(&att)
                    && let Some(id) = existing.get("id").and_then(Value::as_str)
                {
                    return (
                        StatusCode::OK,
                        Json(json!({"id": id, "deduplicated": true})),
                    )
                        .into_response();
                }
            }
            Ok(None) => return bad("no such attachment"),
            Err(e) => return internal(&e),
        }
    }
    if req.sha256.trim().is_empty() {
        return bad("sha256 is required when no attachment_id is given");
    }
    match state.db.save_forensic_report(&report_to_new(req)) {
        Ok(id) => (StatusCode::CREATED, Json(json!({"id": id}))).into_response(),
        Err(e) => internal(&e),
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    let d = h.finalize();
    let mut s = String::with_capacity(d.len() * 2);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// `GET /api/forensics/{id}` - the full report (columns + findings + key
/// findings + explanation categories).
pub async fn get_report(State(state): State<Arc<AppState>>, Path(id): Path<String>) -> Response {
    match state.db.get_forensic_report(&id) {
        Ok(Some(v)) => Json(v).into_response(),
        Ok(None) => not_found("no such forensic report"),
        Err(e) => internal(&e),
    }
}

/// `GET /api/conversations/{conversation_id}/forensics` - report summaries for
/// one conversation, newest first (the Studio's per-conversation list).
pub async fn list_for_conversation(
    State(state): State<Arc<AppState>>,
    Path(conversation_id): Path<String>,
) -> Response {
    match state.db.list_forensic_reports(&conversation_id) {
        Ok(v) => Json(v).into_response(),
        Err(e) => internal(&e),
    }
}

/// `GET /api/attachments/{attachment_id}/forensics` - the stored full report
/// for one attachment (the most recent if re-analyzed), or 404 when it has not
/// been analyzed yet. Mirrors `/metadata`: the manager answers from the DB,
/// runner-independent - a hit needs no live GPU server.
pub async fn latest_for_attachment(
    State(state): State<Arc<AppState>>,
    Path(attachment_id): Path<String>,
) -> Response {
    match state
        .db
        .latest_forensic_report_for_attachment(&attachment_id)
    {
        Ok(Some(v)) => Json(v).into_response(),
        Ok(None) => not_found("attachment has no forensic report"),
        Err(e) => internal(&e),
    }
}

/// Map a runner forensics `report` object (+ request context) to the owned
/// store type. Every field is best-effort: a malformed/absent value falls back
/// to a sane default rather than failing the persist - a partial report beats
/// dropping the analysis on the floor.
fn report_to_new(req: PersistRequest) -> NewForensicReport {
    let r = &req.report;
    let exp = &r["explanation"];
    NewForensicReport {
        attachment_id: req.attachment_id,
        conversation_id: req.conversation_id,
        sha256: req.sha256,
        kind: req.kind.unwrap_or_else(|| "image".to_string()),
        mime: req.mime,
        name: req.name,
        width: r["width"].as_u64().map(|n| n as i64),
        height: r["height"].as_u64().map(|n| n as i64),
        content_type: str_or(&r["content_type"], "unknown"),
        format: str_of(&r["format"]),
        risk_score: r["risk_score"].as_f64().unwrap_or(0.0),
        verdict: str_of(&r["verdict"]),
        risk_level: str_or(&r["risk_level"], "info"),
        corroborating_stages: r["corroborating_families"].as_i64().unwrap_or(0),
        explanation_summary: str_of(&exp["summary"]),
        explanation_visual_review: opt_str(&exp["visual_review"]),
        explanation_cross_corroboration: opt_str(&exp["cross_corroboration"]),
        explanation_anti_forensics: opt_str(&exp["anti_forensics_warning"]),
        gpu: req.gpu,
        elapsed_ms: req.elapsed_ms,
        key_findings: map_key_findings(&r["key_findings"]),
        explanation_categories: map_categories(&exp["categories"]),
        findings: map_findings(&r["findings"]),
    }
}

fn map_findings(v: &Value) -> Vec<NewForensicFinding> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|f| NewForensicFinding {
                    analyzer: str_of(&f["analyzer"]),
                    code: str_of(&f["code"]),
                    severity: str_of(&f["severity"]),
                    confidence: f["confidence"].as_f64().unwrap_or(0.0),
                    description: str_of(&f["description"]),
                    region: region_str(&f["region"]),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn map_key_findings(v: &Value) -> Vec<NewForensicKeyFinding> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|k| NewForensicKeyFinding {
                    title: str_of(&k["title"]),
                    description: str_of(&k["description"]),
                    severity: str_of(&k["severity"]),
                    confidence: k["confidence"].as_f64().unwrap_or(0.0),
                    sources: str_vec(&k["sources"]),
                    region: region_str(&k["region"]),
                    count: k["count"].as_i64().unwrap_or(0),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn map_categories(v: &Value) -> Vec<NewForensicExplanationCategory> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|c| NewForensicExplanationCategory {
                    name: str_of(&c["name"]),
                    finding_count: c["finding_count"].as_i64().unwrap_or(0),
                    // the runner names this field `severity` (the category's max)
                    max_severity: str_or(&c["severity"], "info"),
                    explanation: str_of(&c["explanation"]),
                    finding_codes: str_vec(&c["finding_codes"]),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn str_of(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_string()
}

fn str_or(v: &Value, default: &str) -> String {
    v.as_str()
        .filter(|s| !s.is_empty())
        .unwrap_or(default)
        .to_string()
}

fn opt_str(v: &Value) -> Option<String> {
    v.as_str().map(str::to_string)
}

fn str_vec(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// A finding/key-finding region is a nested JSON object in the runner report;
/// the store keeps it as a JSON string (or `""` when absent), matching how the
/// raw region already round-trips.
fn region_str(v: &Value) -> String {
    if v.is_null() {
        String::new()
    } else {
        v.to_string()
    }
}

fn bad(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, Json(json!({"error": msg}))).into_response()
}

fn not_found(msg: &str) -> Response {
    (StatusCode::NOT_FOUND, Json(json!({"error": msg}))).into_response()
}

fn internal(e: &crate::store::StoreError) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({"error": e.to_string()})),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A representative runner report object (the `report` field of a
    /// `{type:"forensics"}` output item), as `report_value` serializes it.
    fn sample_report() -> Value {
        json!({
            "count": 2,
            "content_type": "photo",
            "format": "jpeg",
            "width": 640,
            "height": 480,
            "risk_score": 0.66,
            "risk_level": "high",
            "verdict": "Likely manipulated",
            "corroborating_families": 2,
            "key_findings": [{
                "title": "Local noise inconsistency",
                "severity": "high",
                "confidence": 0.8,
                "sources": ["noise", "ela"],
                "description": "A region's noise floor differs from the frame.",
                "count": 3,
                "region": {"type": "bounding_box", "x": 1, "y": 2, "width": 3, "height": 4}
            }],
            "explanation": {
                "summary": "Independent signals agree on tampering.",
                "visual_review": "Inspect the sky.",
                "cross_corroboration": "noise + ela",
                "categories": [{
                    "name": "Sensor noise",
                    "severity": "high",
                    "finding_count": 2,
                    "explanation": "Noise varies across the frame.",
                    "finding_codes": ["noise_inconsistency"]
                }]
            },
            "findings": [
                {"analyzer": "ela", "code": "ela_block_outliers", "severity": "low",
                 "confidence": 0.5, "description": "ELA outliers"},
                {"analyzer": "noise", "code": "noise_inconsistency", "severity": "high",
                 "confidence": 0.8, "description": "noise varies",
                 "region": {"type": "bounding_box", "x": 1, "y": 2, "width": 3, "height": 4}}
            ]
        })
    }

    #[test]
    fn maps_runner_report_to_every_store_field() {
        let req = PersistRequest {
            attachment_id: Some("att-1".into()),
            conversation_id: Some("conv-1".into()),
            sha256: "deadbeef".into(),
            mime: "image/jpeg".into(),
            name: "photo.jpg".into(),
            kind: Some("image".into()),
            gpu: true,
            elapsed_ms: 42,
            report: sample_report(),
        };
        let n = report_to_new(req);
        assert_eq!(n.content_type, "photo");
        assert_eq!(n.format, "jpeg");
        assert_eq!(n.width, Some(640));
        assert_eq!(n.height, Some(480));
        assert_eq!(n.risk_level, "high");
        assert_eq!(n.corroborating_stages, 2);
        assert_eq!(n.verdict, "Likely manipulated");
        assert_eq!(
            n.explanation_summary,
            "Independent signals agree on tampering."
        );
        assert_eq!(
            n.explanation_visual_review.as_deref(),
            Some("Inspect the sky.")
        );
        assert_eq!(n.explanation_anti_forensics, None);
        assert_eq!(n.key_findings.len(), 1);
        assert_eq!(n.key_findings[0].count, 3);
        assert_eq!(n.key_findings[0].sources, vec!["noise", "ela"]);
        assert!(
            n.key_findings[0].region.contains("bounding_box"),
            "region kept as JSON"
        );
        assert_eq!(n.explanation_categories.len(), 1);
        assert_eq!(n.explanation_categories[0].max_severity, "high");
        assert_eq!(
            n.explanation_categories[0].finding_codes,
            vec!["noise_inconsistency"]
        );
        assert_eq!(n.findings.len(), 2);
        assert_eq!(n.findings[0].region, "", "absent region -> empty string");
        assert!(n.findings[1].region.contains("bounding_box"));
    }

    #[test]
    fn persists_and_reads_back_through_the_store() {
        let store = crate::store::Store::open(&std::path::PathBuf::from(":memory:")).unwrap();
        let req = PersistRequest {
            attachment_id: Some("att-9".into()),
            conversation_id: Some("conv-9".into()),
            sha256: "cafe".into(),
            mime: "image/jpeg".into(),
            name: "p.jpg".into(),
            kind: Some("image".into()),
            gpu: true,
            elapsed_ms: 7,
            report: sample_report(),
        };
        let id = store.save_forensic_report(&report_to_new(req)).unwrap();
        let got = store
            .get_forensic_report(&id)
            .unwrap()
            .expect("report exists");
        // The runner's report survived the whole path into columns + children.
        assert_eq!(got["risk_level"], "high");
        assert_eq!(got["corroborating_stages"], 2);
        assert_eq!(got["max_severity"], "high", "derived from the raw findings");
        assert_eq!(
            got["explanation"]["summary"],
            "Independent signals agree on tampering."
        );
        assert_eq!(got["explanation"]["categories"][0]["name"], "Sensor noise");
        assert_eq!(got["key_findings"][0]["count"], 3);
        assert_eq!(got["key_findings"][0]["region"]["type"], "bounding_box");
        assert_eq!(got["findings"].as_array().unwrap().len(), 2);

        // And the Studio's read surfaces resolve it.
        let list = store.list_forensic_reports("conv-9").unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0]["risk_level"], "high");
        let latest = store
            .latest_forensic_report_for_attachment("att-9")
            .unwrap()
            .unwrap();
        assert_eq!(latest["id"], id);
    }

    #[test]
    fn resolves_identity_from_the_stored_attachment() {
        let store = crate::store::Store::open(&std::path::PathBuf::from(":memory:")).unwrap();
        store
            .put_attachment(
                "att-x",
                Some("conv-x"),
                "image/jpeg",
                "r.jpg",
                Some(10),
                Some(20),
                b"the-bytes",
            )
            .unwrap();

        // What persist() does when handed only an attachment_id: pull identity
        // from the stored bytes rather than trusting the client to hash.
        let (mime, name, bytes) = store.get_attachment_named("att-x").unwrap().unwrap();
        let sha = sha256_hex(&bytes);
        assert_eq!(sha.len(), 64, "sha256 hex is 64 chars");
        assert_eq!(sha256_hex(b"the-bytes"), sha, "hashing is deterministic");

        let req = PersistRequest {
            attachment_id: Some("att-x".into()),
            conversation_id: Some("conv-x".into()),
            sha256: sha.clone(),
            mime,
            name,
            kind: Some("image".into()),
            report: sample_report(),
            ..Default::default()
        };
        let id = store.save_forensic_report(&report_to_new(req)).unwrap();

        // The Studio's GET /api/attachments/{id}/forensics resolves it.
        let got = store
            .latest_forensic_report_for_attachment("att-x")
            .unwrap()
            .unwrap();
        assert_eq!(got["id"], id);
        assert_eq!(got["mime"], "image/jpeg", "mime came from the attachment");
        assert_eq!(got["name"], "r.jpg", "name came from the attachment");
        assert_eq!(got["sha256"], sha, "content key resolved from the bytes");
        assert_eq!(got["risk_level"], "high");
    }

    #[test]
    fn tolerates_a_minimal_report() {
        // A clean/authentic image: empty findings, no explanation narrative.
        let req = PersistRequest {
            attachment_id: None,
            conversation_id: None,
            sha256: "00".into(),
            mime: String::new(),
            name: String::new(),
            kind: None,
            gpu: false,
            elapsed_ms: 0,
            report: json!({"count": 0, "risk_score": 0.0, "findings": []}),
        };
        let n = report_to_new(req);
        assert_eq!(n.kind, "image", "kind defaults to image");
        assert_eq!(n.content_type, "unknown", "missing content_type -> unknown");
        assert_eq!(n.risk_level, "info", "missing risk_level -> info");
        assert!(n.findings.is_empty());
        assert!(n.key_findings.is_empty());
        assert!(n.explanation_categories.is_empty());
    }
}
