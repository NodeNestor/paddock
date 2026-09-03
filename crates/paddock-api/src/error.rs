//! Error wire shapes.
//!
//! OpenAI clients expect `{"error": {"message", "type", "param", "code"}}` and
//! several SDKs hard-match on it. Getting error shapes right is part of the
//! conformance bar, so they live here next to the success shapes.

use serde::{Deserialize, Serialize};

/// The inner error object, OpenAI shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub message: String,
    /// e.g. "invalid_request_error", "not_found_error"
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

/// The `{"error": ...}` envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub error: ApiError,
}

impl ErrorBody {
    pub fn new(kind: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            error: ApiError {
                message: message.into(),
                kind: kind.into(),
                param: None,
                code: None,
            },
        }
    }

    /// 404 body for unknown routes/models - kept here so every crate 404s the same way.
    pub fn not_found(what: impl std::fmt::Display) -> Self {
        Self::new("not_found_error", format!("{what} not found"))
    }

    /// Attach the stable machine-readable `error.code` (e.g.
    /// "context_length_exceeded") SDKs branch on.
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.error.code = Some(code.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_envelope_matches_openai_shape() {
        let body = ErrorBody::new("invalid_request_error", "model is required");
        let json = serde_json::to_value(&body).expect("serializes");
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "model is required");
        // param/code must be absent when None, not null - SDKs care
        assert!(json["error"].get("param").is_none());
    }

    #[test]
    fn with_code_lands_in_error_code() {
        let body = ErrorBody::new("invalid_request_error", "too long")
            .with_code("context_length_exceeded");
        let json = serde_json::to_value(&body).expect("serializes");
        assert_eq!(json["error"]["code"], "context_length_exceeded");
    }
}
