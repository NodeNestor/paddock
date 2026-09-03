//! Identity, namespace and content-chain keys.
//!
//! Everything persisted or shared hangs off one canonical identity digest, and
//! the digest must be **stable across restarts** - a per-boot salt would make
//! every restart a total miss and silently defeat persistence (audit finding).
//! Privacy is a separate axis: the same identity can be served under different
//! privacy scopes, and scopes never share keys (cache-hit timing + retained KV
//! are a prompt-probing risk between users - per-user isolation is the
//! default; enterprises can elect a shared scope per trust domain).
//!
//! All hashing is BLAKE3 in derive-key mode - each use site gets its own
//! context string, so a digest from one domain can never collide into another
//! (identity vs chain vs payload checksum), and the encoding is
//! length-prefixed so field boundaries can't be confused by concatenation.

use super::payload::PayloadSchema;

/// Domain-separation contexts. Changing any of these (or the encoding under
/// them) is a cache-format break: bump the trailing version and old entries
/// simply miss (never misread). They are part of the persistent format.
const CTX_IDENTITY: &str = "paddock kv_tier identity v1";
const CTX_NAMESPACE: &str = "paddock kv_tier namespace v1";
const CTX_CHAIN: &str = "paddock kv_tier block-chain v1";
const CTX_PAYLOAD: &str = "paddock kv_tier payload v1";

/// BLAKE3 payload checksum. Computed by the producer at pack time, validated
/// at every publish (store completion) and every restore (load completion) -
/// end-to-end, so a corruption anywhere between pack and unpack is caught at
/// the first read, not shipped into attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Checksum(pub [u8; 32]);

impl Checksum {
    /// Checksum of a packed payload extent.
    pub fn of_payload(bytes: &[u8]) -> Self {
        let mut h = blake3::Hasher::new_derive_key(CTX_PAYLOAD);
        h.update(bytes);
        Checksum(*h.finalize().as_bytes())
    }
}

/// The canonical serving-identity digest: two entries hit only if every input
/// that shapes KV bytes is identical. Deliberately coarse - a false miss costs
/// a prefill, a false hit costs correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IdentityDigest(pub [u8; 32]);

/// Every field that shapes the bytes a cache entry holds. A struct rather
/// than a builder so a new call site cannot forget a field - adding one here
/// breaks every constructor at compile time, which is exactly the review
/// point we want (it is also a persistent-format break, see `CTX_IDENTITY`).
#[derive(Debug, Clone, Copy)]
pub struct IdentityFields<'a> {
    /// Content identity of the weights actually loaded - for GGUF the file
    /// content hash(es) the loader already verifies; for safetensors the
    /// header + shard hashes. Not the file *path* (moves) and not the model
    /// *name* (honest-naming aside, names collide).
    pub model_tensors: &'a [u8],
    /// LoRA/adapter content identity, empty when none.
    pub adapter: &'a [u8],
    /// Architecture + everything positional: family name, rope
    /// base/scaling/config, sliding-window sizes, attention layout choices.
    /// Canonical text encoding chosen by the family loader.
    pub architecture: &'a [u8],
    /// The payload schema - cache-group shapes, dtypes, scale layout. Encoded
    /// via [`PayloadSchema::canonical_bytes`], so a KV-dtype flip (f16 vs
    /// fp8-e4m3) or a layout ABI change can never alias (SGLang shipped
    /// silent cross-run corruption by omitting kv_cache_dtype from cache
    /// keys - #33268; this field is why we can't).
    pub cache_schema: &'a [u8],
    /// Engine KV layout ABI revision - bumped whenever the on-device or
    /// packed representation changes shape without any config changing.
    pub layout_abi: u32,
    /// Tokenizer + chat-template identity (token ids are meaningless across
    /// tokenizer revisions; templates shape the prompt bytes).
    pub tokenizer: &'a [u8],
}

impl IdentityDigest {
    pub fn compute(f: &IdentityFields<'_>) -> Self {
        let mut h = blake3::Hasher::new_derive_key(CTX_IDENTITY);
        field(&mut h, "model_tensors", f.model_tensors);
        field(&mut h, "adapter", f.adapter);
        field(&mut h, "architecture", f.architecture);
        field(&mut h, "cache_schema", f.cache_schema);
        field(&mut h, "layout_abi", &f.layout_abi.to_le_bytes());
        field(&mut h, "tokenizer", f.tokenizer);
        IdentityDigest(*h.finalize().as_bytes())
    }

    /// Convenience: compute with the schema encoded canonically.
    pub fn with_schema(f: &IdentityFields<'_>, schema: &PayloadSchema) -> Self {
        let enc = schema.canonical_bytes();
        IdentityDigest::compute(&IdentityFields {
            cache_schema: &enc,
            ..*f
        })
    }
}

/// Length-prefixed labeled field: `len(label) ‖ label ‖ len(value) ‖ value`.
/// The prefixes make the encoding injective - no concatenation of fields can
/// produce another valid field sequence.
fn field(h: &mut blake3::Hasher, label: &str, value: &[u8]) {
    h.update(&(label.len() as u64).to_le_bytes());
    h.update(label.as_bytes());
    h.update(&(value.len() as u64).to_le_bytes());
    h.update(value);
}

/// Privacy scope - Who may hit these entries. Scopes partition the key space
/// completely (they are hashed into the namespace root), so isolation needs
/// no runtime filtering and a scope leak is unrepresentable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrivacyScope {
    /// Per-user isolation - the default. The tag is the serving layer's
    /// stable caller identity (API-key id, OS user, session principal).
    PerUser(Vec<u8>),
    /// One shared pool for every caller of this runner - the single-user
    /// local box and explicitly-elected enterprise trust domains.
    Shared,
}

/// Identity × privacy: the root every logical key chain grows from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheNamespace {
    pub identity: IdentityDigest,
    pub scope: PrivacyScope,
}

impl CacheNamespace {
    /// The chain root for this namespace.
    pub fn root(&self) -> LogicalKey {
        let mut h = blake3::Hasher::new_derive_key(CTX_NAMESPACE);
        field(&mut h, "identity", &self.identity.0);
        match &self.scope {
            PrivacyScope::Shared => field(&mut h, "scope", b"shared"),
            PrivacyScope::PerUser(tag) => {
                field(&mut h, "scope", b"per-user");
                field(&mut h, "user", tag);
            }
        }
        LogicalKey(*h.finalize().as_bytes())
    }
}

/// Content-chain key for one native block (16 tokens - `kv_pool::BLOCK_TOKENS`)
/// of one prefix. `key(block_i) = H(key(block_{i-1}) ‖ tokens_i)`, rooted at
/// the namespace, so a key commits to the entire token prefix - equal keys =>
/// equal prefixes (up to hash collision), which is what makes cross-sequence
/// dedup and partial-prefix hits sound without storing token strings.
///
/// Multimodal spans chain their content digest (image/audio bytes hash from
/// the multimodal pipeline) instead of raw token ids via [`LogicalKey::child_bytes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalKey(pub [u8; 32]);

impl LogicalKey {
    /// Key of the child block holding `tokens` (one block's worth).
    pub fn child(&self, tokens: &[u32]) -> LogicalKey {
        let mut enc = Vec::with_capacity(tokens.len() * 4);
        for t in tokens {
            enc.extend_from_slice(&t.to_le_bytes());
        }
        self.child_bytes("tokens", &enc)
    }

    /// Chain arbitrary content (multimodal spans, boundary-state markers).
    pub fn child_bytes(&self, kind: &str, content: &[u8]) -> LogicalKey {
        let mut h = blake3::Hasher::new_derive_key(CTX_CHAIN);
        field(&mut h, "parent", &self.0);
        field(&mut h, kind, content);
        LogicalKey(*h.finalize().as_bytes())
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    fn fields<'a>() -> IdentityFields<'a> {
        IdentityFields {
            model_tensors: b"tensor-hash",
            adapter: b"",
            architecture: b"gemma4 rope=10000 swa=1024",
            cache_schema: b"schema-bytes",
            layout_abi: 3,
            tokenizer: b"tok-hash",
        }
    }

    #[test]
    fn identity_is_stable_and_field_sensitive() {
        let a = IdentityDigest::compute(&fields());
        let b = IdentityDigest::compute(&fields());
        assert_eq!(
            a, b,
            "same inputs must digest identically across calls (and boots)"
        );
        let c = IdentityDigest::compute(&IdentityFields {
            layout_abi: 4,
            ..fields()
        });
        assert_ne!(a, c, "layout ABI must partition the key space");
        let d = IdentityDigest::compute(&IdentityFields {
            cache_schema: b"other",
            ..fields()
        });
        assert_ne!(
            a, d,
            "cache schema (kv dtype!) must partition the key space"
        );
    }

    #[test]
    fn scopes_partition_the_chain() {
        let id = IdentityDigest::compute(&fields());
        let shared = CacheNamespace {
            identity: id,
            scope: PrivacyScope::Shared,
        }
        .root();
        let user_a = CacheNamespace {
            identity: id,
            scope: PrivacyScope::PerUser(b"a".to_vec()),
        }
        .root();
        let user_b = CacheNamespace {
            identity: id,
            scope: PrivacyScope::PerUser(b"b".to_vec()),
        }
        .root();
        assert_ne!(shared, user_a);
        assert_ne!(user_a, user_b);
        // and the divergence propagates down the whole chain
        assert_ne!(user_a.child(&[1, 2, 3]), user_b.child(&[1, 2, 3]));
    }

    #[test]
    fn chain_commits_to_the_whole_prefix() {
        let id = IdentityDigest::compute(&fields());
        let root = CacheNamespace {
            identity: id,
            scope: PrivacyScope::Shared,
        }
        .root();
        let a = root.child(&[1, 2]).child(&[3, 4]);
        let b = root.child(&[1, 2]).child(&[3, 5]);
        let c = root.child(&[9, 9]).child(&[3, 4]);
        assert_ne!(a, b, "differing block content must differ");
        assert_ne!(
            a, c,
            "same block content under a different parent must differ"
        );
        // length-prefix injectivity: [1,2]+[3] vs [1]+[2,3] must not alias
        assert_ne!(
            root.child(&[1, 2]).child(&[3]),
            root.child(&[1]).child(&[2, 3])
        );
    }
}
