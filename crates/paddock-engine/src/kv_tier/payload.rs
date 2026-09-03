//! Payload schema + per-family codec contract.
//!
//! The unit a tier stores is not "the KV cache" - it is *everything required
//! to resume at a boundary*: attention cache groups, SWA ring bytes + window
//! metadata, recurrent/conv state sealed at the same epoch, positional and
//! multimodal metadata, drafter state (or a priced re-warm plan). A hit is
//! usable only when every `required` component is ready; rivals that stored
//! attention KV alone either broke hybrids outright (vLLM #38230/#40696,
//! SGLang #33713) or collapsed spec acceptance on cached hits (SGLang #31600).
//!
//! **V1 is representation-exact**: the live serving bytes are stored
//! unchanged for every component. "fp8 at rest" is free exactly where the
//! serving KV dtype is already fp8 (KV8); any codec beyond that is a
//! separately named lossy mode with its own contract - none in v1.
//!
//! Each model family implements [`KvPayloadCodec`] and keeps its own layouts;
//! the tier layer stays generic. This fixes the contract; family
//! implementations land in 1a (dense) and 1b (SWA ring, DeltaNet/Mamba
//! checkpoints, drafter state).

/// Version of one family's payload schema. Bumped whenever the packed
/// representation changes; part of the identity digest, so old entries miss
/// instead of misreading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SchemaVersion(pub u16);

/// What a component is - the resume semantics, not the byte layout (layout is
/// the codec's business).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ComponentKind {
    /// One attention cache group: contiguous K/V planes for a set of layers
    /// sharing shape + dtype (per-layer cache kinds - a family may carry
    /// several groups: full-attn f16, fp8 KV8, ...).
    AttnGroup,
    /// Sliding-window ring bytes + the window/wrap metadata needed to resume
    /// the ring exactly (gemma4-class SWA layers).
    SwaRing,
    /// Recurrent state checkpoint at a block boundary (DeltaNet/Mamba/GDN) -
    /// the two-boundary checkpoint discipline `paged_radix` already keeps.
    RecurrentState,
    /// Convolutional shift state sealed at the same epoch as the recurrent
    /// state (Mamba conv1d tail).
    ConvState,
    /// Positional/rope bookkeeping that is not derivable from token count
    /// alone (scaling state, mrope sections for multimodal).
    PositionalMeta,
    /// Multimodal span metadata (image/audio placement, content digests).
    MultimodalMeta,
    /// Drafter (MTP/DFlash) state, or its absence with a priced re-warm plan -
    /// the restore election charges re-warm time either way.
    DrafterState,
}

/// How a component's size scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentSize {
    /// Fixed bytes per 16-token block (attention KV, SWA ring pages).
    PerBlock(u64),
    /// Fixed bytes per sealed boundary regardless of block count (recurrent
    /// checkpoints, drafter state, metadata).
    PerBoundary(u64),
    /// Variable with a hard cap the reservation charges (byte accounting is
    /// exact at seal time; the cap is for admission before sealing).
    Variable { max: u64 },
}

/// Element representation tag - coarse deliberately: the tier layer only needs
/// it for honest observability and schema identity, never to interpret bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtypeTag {
    F32,
    F16,
    Bf16,
    F8E4M3,
    /// Quantized or otherwise opaque byte blobs (scales inline).
    Bytes,
}

/// One component of a family's payload.
#[derive(Debug, Clone)]
pub struct ComponentDesc {
    /// Stable id within the family schema (never reused across versions).
    pub id: u16,
    pub kind: ComponentKind,
    /// Human label for observability ("gdn state", "swa ring", ...).
    pub label: &'static str,
    pub dtype: DtypeTag,
    pub size: ComponentSize,
    /// A hit missing any required component is not a hit (it may still be a
    /// partial-compute plan). Optional components (e.g. drafter state) degrade
    /// to their priced fallback instead.
    pub required: bool,
}

/// A family's complete payload schema.
#[derive(Debug, Clone)]
pub struct PayloadSchema {
    /// Family name as the loaders spell it ("gemma4", "qwen35", ...).
    pub family: &'static str,
    pub version: SchemaVersion,
    pub components: Vec<ComponentDesc>,
}

impl PayloadSchema {
    /// Canonical encoding for the identity digest - length-prefixed, ordered,
    /// covering every field that shapes stored bytes. Changing this encoding
    /// is itself a schema break (bump [`SchemaVersion`]).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.components.len() * 32);
        let mut put = |bytes: &[u8]| {
            out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
            out.extend_from_slice(bytes);
        };
        put(self.family.as_bytes());
        put(&self.version.0.to_le_bytes());
        for c in &self.components {
            put(&c.id.to_le_bytes());
            put(format!("{:?}", c.kind).as_bytes());
            put(format!("{:?}", c.dtype).as_bytes());
            let sz = match c.size {
                ComponentSize::PerBlock(b) => ("per-block", b),
                ComponentSize::PerBoundary(b) => ("per-boundary", b),
                ComponentSize::Variable { max } => ("variable", max),
            };
            put(sz.0.as_bytes());
            put(&sz.1.to_le_bytes());
            put(&[c.required as u8]);
        }
        out
    }

    /// Reservation-time byte bound for a payload covering `blocks` blocks -
    /// exact when no component is `Variable`, an upper bound otherwise.
    /// Admission charges this (reservation-first); the seal reports
    /// the exact size and the difference is returned to the ledger.
    pub fn reserve_bytes(&self, blocks: u32) -> u64 {
        self.components
            .iter()
            .map(|c| match c.size {
                ComponentSize::PerBlock(b) => b * blocks as u64,
                ComponentSize::PerBoundary(b) => b,
                ComponentSize::Variable { max } => max,
            })
            .sum()
    }
}

/// A device span the pack path gathers from (or the unpack path scatters
/// into): plane is an opaque per-family device-allocation id, offsets are
/// bytes within it. The transport's gather/scatter capability consumes these;
/// the layout-transform pack kernel is what turns strided spans into one
/// contiguous staging run (the lesson every rival paid for - fragmented
/// per-page transfers reach ~20% of the bus).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevSpan {
    pub plane: u32,
    pub offset: u64,
    pub len: u64,
}

/// One realized component inside a packed payload: where it sits in the
/// extent and how big it actually is. The offset table is what lets a partial
/// hit read only useful spans (or knowingly account read amplification).
#[derive(Debug, Clone)]
pub struct PackedComponent {
    pub id: u16,
    pub offset: u64,
    pub bytes: u64,
}

/// The sealed, immutable description of one stored payload. Produced by
/// [`KvPayloadCodec::seal`]; travels with the replica record; consumed by
/// restore. Never mutated after seal - mutable tails and cyclic SWA rings get
/// a boundary epoch before snapshotting.
#[derive(Debug, Clone)]
pub struct PayloadManifest {
    pub schema_version: SchemaVersion,
    /// Number of 16-token blocks this payload resumes.
    pub blocks: u32,
    /// Epoch of the sealed boundary (token position of the seal point) - the
    /// recurrent/conv/drafter components are exact at this epoch.
    pub boundary_epoch: u64,
    pub components: Vec<PackedComponent>,
    /// Exact packed size (≤ the reservation bound).
    pub total_bytes: u64,
}

/// Why a seal was refused. Not exhaustive yet - 1a/1b implementations extend
/// this as real failure modes surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SealError {
    /// The requested boundary is not a sealable epoch for this family (e.g.
    /// no recurrent checkpoint exists at that block boundary).
    NoBoundary,
    /// The blocks are still mutable (tail in flight) - caller must seal at a
    /// published boundary.
    Mutable,
}

/// Why a restore was refused by the codec (transport-level failures live in
/// the catalog/transport layer).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreError {
    /// Manifest schema version this codec no longer speaks.
    SchemaMismatch,
    /// Destination shape doesn't match the manifest (wrong block count).
    ShapeMismatch,
}

/// Per-family payload codec - the seam that keeps family layouts inside the
/// family. The tier layer calls this; it never touches planes.
///
/// Contract (refined by the 1a implementation, but these hold):
/// - `seal` is called only at an immutable boundary; the manifest it returns
///   is final for that generation.
/// - `device_spans` for the same manifest is deterministic - the pack kernel
///   and a debug CPU reader must see the same gather list.
/// - `restore` runs after the packed bytes are back in staging and the
///   destination blocks are reserved; it re-establishes AUXILIARY state
///   (recurrent epoch, ring wrap metadata, drafter warmth) - the bulk byte
///   movement into planes is the transport's scatter, not the codec's.
pub trait KvPayloadCodec {
    fn schema(&self) -> &PayloadSchema;

    /// Seal the payload for `blocks` blocks ending at `boundary_epoch`.
    fn seal(&mut self, blocks: u32, boundary_epoch: u64) -> Result<PayloadManifest, SealError>;

    /// Gather list for a sealed payload (pack) - offsets pair with the
    /// manifest's component offset table (unpack scatters the same spans).
    fn device_spans(&self, manifest: &PayloadManifest) -> Vec<DevSpan>;

    /// Re-establish auxiliary state after the bytes are back on-device.
    fn restore(&mut self, manifest: &PayloadManifest) -> Result<(), RestoreError>;
}

#[cfg(test)]
mod unit {
    use super::*;

    fn schema() -> PayloadSchema {
        PayloadSchema {
            family: "test",
            version: SchemaVersion(1),
            components: vec![
                ComponentDesc {
                    id: 0,
                    kind: ComponentKind::AttnGroup,
                    label: "kv f16",
                    dtype: DtypeTag::F16,
                    size: ComponentSize::PerBlock(1024),
                    required: true,
                },
                ComponentDesc {
                    id: 1,
                    kind: ComponentKind::RecurrentState,
                    label: "gdn state",
                    dtype: DtypeTag::F32,
                    size: ComponentSize::PerBoundary(4096),
                    required: true,
                },
                ComponentDesc {
                    id: 2,
                    kind: ComponentKind::DrafterState,
                    label: "mtp",
                    dtype: DtypeTag::Bytes,
                    size: ComponentSize::Variable { max: 512 },
                    required: false,
                },
            ],
        }
    }

    #[test]
    fn reserve_bytes_is_an_upper_bound() {
        let s = schema();
        assert_eq!(s.reserve_bytes(4), 4 * 1024 + 4096 + 512);
        assert_eq!(s.reserve_bytes(0), 4096 + 512);
    }

    #[test]
    fn canonical_bytes_distinguishes_versions_and_dtypes() {
        let a = schema().canonical_bytes();
        let mut v2 = schema();
        v2.version = SchemaVersion(2);
        assert_ne!(a, v2.canonical_bytes());
        let mut fp8 = schema();
        fp8.components[0].dtype = DtypeTag::F8E4M3;
        assert_ne!(
            a,
            fp8.canonical_bytes(),
            "kv dtype must change the schema identity"
        );
    }
}
