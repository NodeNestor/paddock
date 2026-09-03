//! Request filters (doc §13): operator-declared rewrites of generation-request
//! bodies, adopted from llama-swap's field-proven trio - `stripParams`
//! (remove client-sent fields), `setParams` (force values regardless of what
//! clients send), `setParamsByID` (param presets bound to API-visible model
//! ids like `qwen3.5-9b:high`) - plus the plural `aliases` list. llama-swap's
//! config surface is the fossil record of what thousands of hot-swap users
//! actually demanded; we adopt the semantics, not the code.
//!
//! Application order is llama-swap's, verbatim: strip -> force -> variant, so a
//! variant preset overrides a forced value (the operator declared both; the
//! more specific binding wins). All sets are SHALLOW - a preset key replaces
//! the whole top-level field (`chat_template_kwargs` is set as one object),
//! which is the predictable contract their docs promise. `model` is protected
//! everywhere: filters must never change which model a request believes it
//! addressed (responses echo the requested id; event records already carry
//! the truthful served id independently).
//!
//! Variant selection: a request for `<base>:<name>` where `<base>` is the
//! served id or an alias and `<name>` is a declared variant applies the
//! preset. A matching base with an UNKNOWN suffix is a 404 - a typo'd variant
//! silently serving base behaviour would be a silent failure. Any other model
//! string passes through untouched (single-model endpoint semantics: clients
//! select a model by endpoint, doc §5).

use serde_json::{Map, Value};

/// One declared parameter variant: `name` becomes the API-visible id suffix
/// (`<served>:<name>`), `params` the request fields it sets.
#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub params: Map<String, Value>,
}

/// The resolved filter config, validated at startup. On `AppState`.
#[derive(Debug, Default)]
pub struct Filters {
    /// Alternative model ids this endpoint answers to (llama-swap `aliases`).
    pub aliases: Vec<String>,
    /// Param presets addressable as `<served-or-alias>:<name>`, name-sorted so
    /// the /v1/models listing is stable.
    pub variants: Vec<Variant>,
    /// Client-sent top-level fields to remove (llama-swap `stripParams`).
    pub strip: Vec<String>,
    /// Fields forced to a value regardless of the request (their `setParams`).
    pub force: Map<String, Value>,
}

/// Fields filters may never touch: rewriting `model` would make the endpoint
/// lie about what a request addressed (llama-swap protects the same key).
const PROTECTED: &[&str] = &["model"];

impl Filters {
    /// Build from config values, validating loudly: a malformed variant is a
    /// startup error, not a preset that silently never applies.
    pub fn build(
        aliases: Vec<String>,
        variants: std::collections::HashMap<String, Value>,
        strip: Vec<String>,
        force: std::collections::HashMap<String, Value>,
    ) -> Result<Self, String> {
        let aliases: Vec<String> = aliases
            .into_iter()
            .map(|a| a.trim().to_owned())
            .filter(|a| !a.is_empty())
            .collect();

        let mut vs = Vec::with_capacity(variants.len());
        for (name, params) in variants {
            let name = name.trim().to_owned();
            if name.is_empty() || name.contains(':') {
                return Err(format!(
                    "variant name {name:?} is invalid: names must be non-empty and contain no ':' \
                     (the id separator - the API id becomes <model>:<name>)"
                ));
            }
            let Value::Object(mut params) = params else {
                return Err(format!(
                    "variant {name:?} must be a table/object of request fields to set, got {params}"
                ));
            };
            for p in PROTECTED {
                if params.remove(*p).is_some() {
                    tracing::warn!(variant = %name, field = p, "variant may not set a protected field - dropped");
                }
            }
            vs.push(Variant { name, params });
        }
        vs.sort_by(|a, b| a.name.cmp(&b.name));

        let strip: Vec<String> = {
            let mut seen = std::collections::HashSet::new();
            strip
                .into_iter()
                .map(|s| s.trim().to_owned())
                .filter(|s| {
                    if s.is_empty() || !seen.insert(s.clone()) {
                        return false;
                    }
                    if PROTECTED.contains(&s.as_str()) {
                        tracing::warn!(field = %s, "strip_params may not remove a protected field - dropped");
                        return false;
                    }
                    true
                })
                .collect()
        };

        let mut fc = Map::new();
        for (k, v) in force {
            if PROTECTED.contains(&k.as_str()) {
                tracing::warn!(field = %k, "force_params may not set a protected field - dropped");
                continue;
            }
            fc.insert(k, v);
        }

        Ok(Self {
            aliases,
            variants: vs,
            strip,
            force: fc,
        })
    }

    /// Whether requests need the body transform at all. Aliases alone don't:
    /// the runner already accepts any model string (endpoint = selection), so
    /// they only matter for the /v1/models listing - zero request overhead.
    pub fn needs_body_transform(&self) -> bool {
        !(self.strip.is_empty() && self.force.is_empty() && self.variants.is_empty())
    }

    /// The declared variant ids for `served`: `served:name` per variant.
    pub fn variant_ids(&self, served: &str) -> Vec<String> {
        self.variants
            .iter()
            .map(|v| format!("{served}:{}", v.name))
            .collect()
    }

    /// Is `base` a name this endpoint answers to (served id or alias)?
    fn is_known_base(&self, base: &str, served: &str) -> bool {
        base == served || self.aliases.iter().any(|a| a == base)
    }

    /// Apply the filters to a parsed request body in place.
    /// `Err(UnknownVariant)` when the request selected `<known-base>:<name>`
    /// for a variant that doesn't exist - the honest 404, never a silent
    /// fall-through to base behaviour.
    pub fn apply(&self, body: &mut Value, served: &str) -> Result<(), UnknownVariant> {
        let Value::Object(obj) = body else {
            return Ok(()); // not an object - the handler owns the 400
        };
        for f in &self.strip {
            obj.remove(f);
        }
        for (k, v) in &self.force {
            obj.insert(k.clone(), v.clone());
        }
        // Variant after force (llama-swap: setParamsByID "runs after setParams
        // so it will override any settings").
        let requested = obj
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        if let Some((base, name)) = requested.rsplit_once(':')
            && self.is_known_base(base, served)
        {
            let Some(variant) = self.variants.iter().find(|v| v.name == name) else {
                return Err(UnknownVariant {
                    requested,
                    known: self.variant_ids(served),
                });
            };
            for (k, v) in &variant.params {
                obj.insert(k.clone(), v.clone());
            }
        }
        Ok(())
    }
}

/// A request selected `<known-base>:<name>` for an undeclared variant.
#[derive(Debug)]
pub struct UnknownVariant {
    pub requested: String,
    pub known: Vec<String>,
}

impl UnknownVariant {
    fn message(&self) -> String {
        if self.known.is_empty() {
            format!(
                "model {:?} not found: no parameter variants are declared on this endpoint",
                self.requested
            )
        } else {
            format!(
                "model {:?} not found: declared variants are {}",
                self.requested,
                self.known.join(", ")
            )
        }
    }
}

/// Mirrors axum's default `Json` body limit so the transform can't be used to
/// buffer more than an unfiltered request could carry; if a configurable body
/// limit ever lands, both move together.
const BODY_LIMIT: usize = 2 * 1024 * 1024;

/// Middleware: transform generation-request bodies per the declared filters.
/// Innermost layer (inside auth + drain), so refused requests are never
/// buffered and the 404 for an unknown variant still gets its event record.
pub async fn filters_mw(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::routes::AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let path = req.uri().path().to_owned();
    let applies = state.filters.needs_body_transform()
        && req.method() == axum::http::Method::POST
        && crate::routes::is_generation_path(&path)
        // the compact endpoint's summarization pass is server-controlled
        // (greedy, fixed cap): forced sampling params have nothing to force,
        // and its narrow request surface rejects unknown fields
        && path != "/v1/responses/compact";
    let Some(served) = (applies && state.serving.is_some())
        .then(|| state.serving.as_ref().map(|s| s.id.clone()))
        .flatten()
    else {
        return next.run(req).await;
    };

    let (parts, body) = req.into_parts();
    let bytes = match axum::body::to_bytes(body, BODY_LIMIT).await {
        Ok(b) => b,
        Err(_) => {
            // Same refusal the Json extractor would give an oversized body,
            // in the OpenAI error shape SDKs parse.
            return (
                axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                axum::Json(paddock_api::ErrorBody::new(
                    "invalid_request_error",
                    "request body exceeds the size limit",
                )),
            )
                .into_response();
        }
    };

    // Not JSON -> hand the original bytes through; the handler owns the 400.
    let Ok(mut parsed) = serde_json::from_slice::<Value>(&bytes) else {
        return next
            .run(axum::extract::Request::from_parts(
                parts,
                axum::body::Body::from(bytes),
            ))
            .await;
    };

    if let Err(unknown) = state.filters.apply(&mut parsed, &served) {
        // Honest 404 in the dialect's own error shape (Anthropic envelope on
        // /v1/messages, OpenAI everywhere else).
        let msg = unknown.message();
        return if path.starts_with("/v1/messages") {
            (
                axum::http::StatusCode::NOT_FOUND,
                axum::Json(serde_json::json!({
                    "type": "error",
                    "error": { "type": "not_found_error", "message": msg }
                })),
            )
                .into_response()
        } else {
            (
                axum::http::StatusCode::NOT_FOUND,
                axum::Json(paddock_api::ErrorBody::new("model_not_found", msg)),
            )
                .into_response()
        };
    }

    let rewritten = match serde_json::to_vec(&parsed) {
        Ok(v) => v,
        Err(_) => bytes.to_vec(), // can't re-serialize what we just parsed - pass the original
    };
    let mut req =
        axum::extract::Request::from_parts(parts, axum::body::Body::from(rewritten.clone()));
    // The buffered rewrite has an exact length; a stale streaming header would
    // make hyper/axum reject or truncate the body.
    req.headers_mut().insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from(rewritten.len()),
    );
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn filters(
        aliases: &[&str],
        variants: &[(&str, Value)],
        strip: &[&str],
        force: &[(&str, Value)],
    ) -> Filters {
        Filters::build(
            aliases.iter().map(|s| (*s).to_owned()).collect(),
            variants
                .iter()
                .map(|(n, p)| ((*n).to_owned(), p.clone()))
                .collect(),
            strip.iter().map(|s| (*s).to_owned()).collect(),
            force
                .iter()
                .map(|(k, v)| ((*k).to_owned(), v.clone()))
                .collect(),
        )
        .expect("valid filter config")
    }

    #[test]
    fn strip_then_force_then_variant_in_llama_swap_order() {
        let f = filters(
            &[],
            &[(
                "high",
                json!({"temperature": 0.2, "chat_template_kwargs": {"reasoning_effort": "high"}}),
            )],
            &["top_p"],
            &[("temperature", json!(0.7))],
        );
        // no variant selected: strip removes, force overrides
        let mut body = json!({"model": "m", "temperature": 1.5, "top_p": 0.5});
        f.apply(&mut body, "m").unwrap();
        assert_eq!(body["temperature"], json!(0.7));
        assert!(body.get("top_p").is_none());
        // variant selected: preset overrides the forced value (their order)
        let mut body = json!({"model": "m:high", "temperature": 1.5});
        f.apply(&mut body, "m").unwrap();
        assert_eq!(body["temperature"], json!(0.2));
        assert_eq!(
            body["chat_template_kwargs"],
            json!({"reasoning_effort": "high"})
        );
        // model is never rewritten - the response echoes what was requested
        assert_eq!(body["model"], json!("m:high"));
    }

    #[test]
    fn unknown_variant_on_a_known_base_is_an_error_not_a_fallthrough() {
        let f = filters(&["gpt-4o-mini"], &[("high", json!({}))], &[], &[]);
        let mut body = json!({"model": "gpt-4o-mini:hgih"});
        let err = f.apply(&mut body, "m").unwrap_err();
        assert_eq!(err.requested, "gpt-4o-mini:hgih");
        assert_eq!(err.known, vec!["m:high".to_owned()]);
        // an alias base selects variants exactly like the served id
        let mut body = json!({"model": "gpt-4o-mini:high"});
        f.apply(&mut body, "m").unwrap();
    }

    #[test]
    fn unrelated_model_strings_pass_through_untouched() {
        let f = filters(&[], &[("high", json!({"seed": 1}))], &[], &[]);
        // foreign id with a colon (ollama-style tag): base doesn't match -> no-op
        let mut body = json!({"model": "llama3:8b"});
        f.apply(&mut body, "m").unwrap();
        assert!(body.get("seed").is_none());
        // served id itself containing a colon still matches exactly
        let mut body = json!({"model": "org:m"});
        f.apply(&mut body, "org:m").unwrap();
        assert!(body.get("seed").is_none());
        let mut body = json!({"model": "org:m:high"});
        f.apply(&mut body, "org:m").unwrap();
        assert_eq!(body["seed"], json!(1));
    }

    #[test]
    fn protected_model_field_survives_all_three_filters() {
        let f = filters(
            &[],
            &[("v", json!({"model": "evil", "seed": 2}))],
            &["model"],
            &[("model", json!("evil"))],
        );
        assert!(f.strip.is_empty(), "strip may not carry protected fields");
        assert!(f.force.is_empty(), "force may not carry protected fields");
        let mut body = json!({"model": "m:v"});
        f.apply(&mut body, "m").unwrap();
        assert_eq!(
            body["model"],
            json!("m:v"),
            "variant must not rewrite model"
        );
        assert_eq!(
            body["seed"],
            json!(2),
            "non-protected preset fields still apply"
        );
    }

    #[test]
    fn malformed_variants_fail_startup_loudly() {
        let bad_name = Filters::build(
            vec![],
            [("hi:gh".to_owned(), json!({}))].into(),
            vec![],
            [].into(),
        );
        assert!(bad_name.is_err());
        let bad_params = Filters::build(
            vec![],
            [("high".to_owned(), json!("not an object"))].into(),
            vec![],
            [].into(),
        );
        assert!(bad_params.is_err());
    }

    #[test]
    fn aliases_alone_need_no_body_transform() {
        let f = filters(&["gpt-4o-mini"], &[], &[], &[]);
        assert!(!f.needs_body_transform());
        let f = filters(&[], &[], &["top_p"], &[]);
        assert!(f.needs_body_transform());
    }

    #[test]
    fn variant_ids_are_name_sorted_for_a_stable_listing() {
        let f = filters(&[], &[("low", json!({})), ("high", json!({}))], &[], &[]);
        assert_eq!(
            f.variant_ids("m"),
            vec!["m:high".to_owned(), "m:low".to_owned()]
        );
    }
}
