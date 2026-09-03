//! `/v1/models` response shapes, OpenAI wire format.

use serde::{Deserialize, Serialize};

/// One entry in the model list. Matches OpenAI's `model` object, extended with
/// the capability-metadata convention llama-swap seeded (and OpenRouter uses):
/// `architecture` / `capabilities` / `supported_parameters` / `context_length`
/// as top-level listing fields, so clients can filter model pickers without a
/// second round-trip. Every extension is `Option` + skip-on-None - a server
/// that sets none of them emits the exact OpenAI shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelObject {
    /// Model identifier as clients pass it in requests, e.g. "gpt-oss-20b".
    pub id: String,
    /// Always the literal "model" per spec.
    pub object: String,
    /// Unix seconds. For local models we use the GGUF file's mtime - it's the
    /// closest honest analogue to "created".
    pub created: u64,
    pub owned_by: String,
    /// paddock extension: does this model accept image input? Only known once a
    /// model is actually loaded (mmproj attached), so it's `None` (omitted from
    /// the wire) for the merely-listed GGUFs - keeping the OpenAI shape intact -
    /// and `Some(true/false)` for the served model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision: Option<bool>,
    /// paddock extension: what an image costs on this endpoint - the tower's
    /// pixel ceiling/floor, pixels-per-token, and what `detail` resolves to.
    /// Present only for a served vision model, so a client can price an image
    /// before sending it instead of discovering the cap by being truncated.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_budget: Option<serde_json::Value>,
    /// paddock extension: task tags the model's chat template expands, as
    /// `[{"tag": "<chart2csv>", "prompt": "<what the model actually gets>"}]`.
    /// granite-vision is the first: putting one of these in a message makes the
    /// template DISCARD that message and substitute the instruction IBM tuned
    /// on. Absent unless the served template has some, so no client sees a new
    /// field for a model that has nothing to say here.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_tags: Option<serde_json::Value>,
    /// Modalities, OpenRouter-shaped: `{"input_modalities": ["text","image"],
    /// "output_modalities": ["text"], "modality": "text+image->text"}`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<serde_json::Value>,
    /// Capability flags: `{"vision": true, "function_calling": true,
    /// "embeddings": true, "reranker": true, ...}` - only the true ones.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<serde_json::Value>,
    /// Request-body fields this endpoint actually honors for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supported_parameters: Option<Vec<String>>,
    /// The SERVED context window (the operational truth - what a request can
    /// use here), not the checkpoint's trained maximum.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
    /// Alternative ids this endpoint accepts for the model (llama-swap's
    /// `aliases` - e.g. impersonating a cloud id for tools that hardcode one).
    /// Listed on the record rather than as duplicate entries.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aliases: Option<Vec<String>>,
    /// For a parameter-variant entry (`<model>:high`): the base model id it
    /// resolves to. One resident model, several selectable behaviours.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_of: Option<String>,
    /// For a parameter-variant entry: the request-body params the variant
    /// applies - shown so the listing is self-documenting, never a mystery id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<serde_json::Value>,
}

impl ModelObject {
    pub fn new(id: impl Into<String>, created: u64, owned_by: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "model".to_owned(),
            created,
            owned_by: owned_by.into(),
            vision: None,
            vision_budget: None,
            task_tags: None,
            architecture: None,
            capabilities: None,
            supported_parameters: None,
            context_length: None,
            aliases: None,
            variant_of: None,
            params: None,
        }
    }

    /// Advertise the served model's image capability.
    pub fn with_vision(mut self, vision: bool) -> Self {
        self.vision = Some(vision);
        self
    }

    /// Advertise what an image costs here (None on a model without a tower).
    pub fn with_vision_budget(mut self, budget: Option<serde_json::Value>) -> Self {
        self.vision_budget = budget;
        self
    }

    /// Advertise the template's task tags. An empty list stays absent rather
    /// than serializing `[]`: "this model has no task tags" and "this server is
    /// too old to know about them" read identically to a client, and an empty
    /// array pretending to be the former would be the dishonest one.
    pub fn with_task_tags<T: Serialize>(mut self, tags: &[T]) -> Self {
        if !tags.is_empty() {
            self.task_tags = serde_json::to_value(tags).ok();
        }
        self
    }

    /// Attach the capability-metadata block (llama-swap convention).
    pub fn with_listing_meta(
        mut self,
        architecture: serde_json::Value,
        capabilities: serde_json::Value,
        supported_parameters: Vec<String>,
        context_length: usize,
    ) -> Self {
        self.architecture = Some(architecture);
        self.capabilities = Some(capabilities);
        self.supported_parameters = Some(supported_parameters);
        self.context_length = Some(context_length);
        self
    }

    /// Attach the endpoint's alias list (omitted when empty).
    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        if !aliases.is_empty() {
            self.aliases = Some(aliases);
        }
        self
    }

    /// Mark this entry as a parameter variant of `base`, carrying its preset.
    pub fn as_variant(mut self, base: impl Into<String>, params: serde_json::Value) -> Self {
        self.variant_of = Some(base.into());
        self.params = Some(params);
        self
    }
}

/// The `/v1/models` envelope: `{"object": "list", "data": [...]}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<ModelObject>,
}

impl ModelList {
    pub fn new(data: Vec<ModelObject>) -> Self {
        Self {
            object: "list".to_owned(),
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_list_matches_openai_shape() {
        let list = ModelList::new(vec![ModelObject::new(
            "gpt-oss-20b",
            1_720_000_000,
            "openai",
        )]);
        let json = serde_json::to_value(&list).expect("serializes");
        assert_eq!(json["object"], "list");
        assert_eq!(json["data"][0]["object"], "model");
        assert_eq!(json["data"][0]["id"], "gpt-oss-20b");
    }
}
