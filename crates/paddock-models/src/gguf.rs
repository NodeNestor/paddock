//! Memory-safe GGUF parser.
//!
//! This replaces the C parsing path where llama.cpp shipped real CVEs
//! (unauthenticated heap OOB via crafted GGUF - CVE-2026-7482 class). Rules:
//! every read is bounds-checked, every count is capped, malformed input returns
//! a specific error and never panics or over-allocates. A fuzz target comes
//! with the hardening pass.
//!
//! Format: GGUF v3 (v2 accepted - same layout, older writers). Little-endian
//! only, per spec.

use std::collections::HashMap;

use crate::ggml_type::GgmlType;

// Caps chosen from real-world headroom: biggest tokenizers are ~200k entries,
// biggest models ~2k tensors (MoE with fused experts). 100x margin, not 1e9.
const MAX_TENSORS: u64 = 262_144;
const MAX_KV: u64 = 65_536;
const MAX_STRING: u64 = 64 << 20; // chat templates get big; 64 MiB is generous
const MAX_ARRAY: u64 = 16_777_216;
const MAX_ARRAY_DEPTH: u32 = 4;
const MAX_DIMS: u32 = 8;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("not a GGUF file (bad magic)")]
    BadMagic,
    #[error("GGUF version {0} not supported (this parser reads v2/v3)")]
    UnsupportedVersion(u32),
    #[error("file truncated: needed {need} bytes at offset {at}")]
    Truncated { at: usize, need: u64 },
    #[error("{what} count {count} exceeds cap {max} - refusing (corrupt or hostile file)")]
    CountTooLarge {
        what: &'static str,
        count: u64,
        max: u64,
    },
    #[error("string at offset {at} is not valid UTF-8")]
    BadUtf8 { at: usize },
    #[error("unknown metadata value type {0}")]
    UnknownValueType(u32),
    #[error("array nesting deeper than {MAX_ARRAY_DEPTH}")]
    ArrayTooDeep,
    #[error("alignment {0} is not a power of two")]
    BadAlignment(u32),
    #[error("tensor {name}: {problem}")]
    BadTensor { name: String, problem: String },
    #[error("duplicate tensor name {0}")]
    DuplicateTensor(String),
}

/// A metadata value. Mirrors the GGUF value-type table exactly.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Widening integer view - writers disagree about u32 vs u64 for the same
    /// keys, so lookups shouldn't care.
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => Some(v),
            Value::I8(v) if v >= 0 => Some(v as u64),
            Value::I16(v) if v >= 0 => Some(v as u64),
            Value::I32(v) if v >= 0 => Some(v as u64),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// Signed counterpart to `as_u64` - same widening view, but keeps negative
    /// values instead of dropping them. Needed for keys that use a negative
    /// sentinel: granite-vision's `clip.vision.projector.spatial_offsets` is
    /// `[-1,-1,-1,-1,0,1,2,3]`, where -1 means "no 2×2 offset pick, use the
    /// area-interpolate downsampler". Reading that through `as_u64` silently
    /// yields None for exactly the four entries that select a different
    /// algorithm.
    pub fn as_i64(&self) -> Option<i64> {
        match *self {
            Value::U8(v) => Some(v.into()),
            Value::U16(v) => Some(v.into()),
            Value::U32(v) => Some(v.into()),
            Value::U64(v) => i64::try_from(v).ok(),
            Value::I8(v) => Some(v.into()),
            Value::I16(v) => Some(v.into()),
            Value::I32(v) => Some(v.into()),
            Value::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match *self {
            Value::F32(v) => Some(v),
            Value::F64(v) => Some(v as f32),
            _ => None,
        }
    }
}

/// One tensor's descriptor from the header. `offset` is relative to the start
/// of the data section, per spec.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ne[0] is the fastest-moving dimension, as stored.
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub raw_type: u32,
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    /// None when the type layout is unverified - callers must surface that,
    /// not guess.
    pub fn byte_size(&self) -> Option<u64> {
        self.ggml_type.byte_size(self.element_count())
    }
}

/// Parsed GGUF header + metadata + tensor directory. Tensor *data* stays in the
/// caller's mmap; `data_offset` says where it begins.
#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub alignment: u64,
    pub metadata: HashMap<String, Value>,
    /// In file order - order matters for loaders that stream sequentially.
    pub tensors: Vec<TensorInfo>,
    /// Absolute file offset where the aligned data section starts.
    pub data_offset: u64,
}

impl GgufFile {
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture")?.as_str()
    }

    /// Per-arch keys like "{arch}.context_length" without the caller doing
    /// string assembly everywhere.
    pub fn arch_field(&self, suffix: &str) -> Option<&Value> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{arch}.{suffix}"))
    }

    /// Parse a complete in-memory file (tensor bounds validated against
    /// `bytes.len()`).
    pub fn parse(bytes: &[u8]) -> Result<Self, GgufError> {
        Self::parse_with_len(bytes, bytes.len() as u64)
    }

    /// Parse from a prefix of the file (header region only) while validating
    /// tensor bounds against the real on-disk size. This is the probe path: the
    /// store reads a few MB, not 12 GB, to answer /v1/models.
    pub fn parse_prefix(bytes: &[u8], file_len: u64) -> Result<Self, GgufError> {
        Self::parse_with_len(bytes, file_len)
    }

    fn parse_with_len(bytes: &[u8], file_len: u64) -> Result<Self, GgufError> {
        let mut c = Cursor { bytes, pos: 0 };

        if c.read_u32()? != 0x4655_4747 {
            return Err(GgufError::BadMagic);
        }
        let version = c.read_u32()?;
        if !(2..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }
        let tensor_count = c.read_u64()?;
        if tensor_count > MAX_TENSORS {
            return Err(GgufError::CountTooLarge {
                what: "tensor",
                count: tensor_count,
                max: MAX_TENSORS,
            });
        }
        let kv_count = c.read_u64()?;
        if kv_count > MAX_KV {
            return Err(GgufError::CountTooLarge {
                what: "metadata kv",
                count: kv_count,
                max: MAX_KV,
            });
        }

        let mut metadata = HashMap::with_capacity(kv_count.min(1024) as usize);
        for _ in 0..kv_count {
            let key = c.read_string()?;
            let type_id = c.read_u32()?;
            let value = c.read_value(type_id, 0)?;
            metadata.insert(key, value);
        }

        let alignment = match metadata.get("general.alignment").and_then(Value::as_u64) {
            Some(a) => {
                let a32 = u32::try_from(a).map_err(|_| GgufError::BadAlignment(u32::MAX))?;
                if !a32.is_power_of_two() {
                    return Err(GgufError::BadAlignment(a32));
                }
                a
            }
            None => 32, // spec default
        };

        let mut tensors = Vec::with_capacity(tensor_count.min(4096) as usize);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..tensor_count {
            let name = c.read_string()?;
            if !seen.insert(name.clone()) {
                return Err(GgufError::DuplicateTensor(name));
            }
            let n_dims = c.read_u32()?;
            if n_dims > MAX_DIMS {
                return Err(GgufError::BadTensor {
                    name,
                    problem: format!("{n_dims} dimensions (max {MAX_DIMS})"),
                });
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(c.read_u64()?);
            }
            let raw_type = c.read_u32()?;
            let offset = c.read_u64()?;
            if !offset.is_multiple_of(alignment) {
                return Err(GgufError::BadTensor {
                    name,
                    problem: format!("data offset {offset} not aligned to {alignment}"),
                });
            }
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type: GgmlType::from_raw(raw_type),
                raw_type,
                offset,
            });
        }

        // data section starts at the next alignment boundary after the header
        let data_offset = (c.pos as u64).div_ceil(alignment) * alignment;

        let parsed = Self {
            version,
            alignment,
            metadata,
            tensors,
            data_offset,
        };
        parsed.validate_tensor_bounds(file_len)?;
        Ok(parsed)
    }

    /// Every sizable tensor must land inside the file. Unsizable types (unknown
    /// layouts) only get the weaker offset-in-range check.
    fn validate_tensor_bounds(&self, file_len: u64) -> Result<(), GgufError> {
        let data_len = file_len.saturating_sub(self.data_offset);
        for t in &self.tensors {
            match t.byte_size() {
                Some(size) => {
                    let end = t.offset.checked_add(size);
                    if end.is_none() || end.unwrap_or(u64::MAX) > data_len {
                        return Err(GgufError::BadTensor {
                            name: t.name.clone(),
                            problem: format!(
                                "extends past end of file (offset {} + {} bytes > data section {})",
                                t.offset, size, data_len
                            ),
                        });
                    }
                }
                None => {
                    if t.offset >= data_len && !(t.offset == 0 && data_len == 0) {
                        return Err(GgufError::BadTensor {
                            name: t.name.clone(),
                            problem: format!(
                                "offset {} outside data section {}",
                                t.offset, data_len
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// Bounds-checked little-endian reader. Private deliberately - all format logic
/// goes through it, nothing reads `bytes` directly.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: u64) -> Result<&'a [u8], GgufError> {
        let n_usize = usize::try_from(n).map_err(|_| GgufError::Truncated {
            at: self.pos,
            need: n,
        })?;
        let end = self.pos.checked_add(n_usize).ok_or(GgufError::Truncated {
            at: self.pos,
            need: n,
        })?;
        if end > self.bytes.len() {
            return Err(GgufError::Truncated {
                at: self.pos,
                need: n,
            });
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn read_u32(&mut self) -> Result<u32, GgufError> {
        // take() guarantees 4 bytes, so the conversion cannot fail
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn read_u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }

    fn read_string(&mut self) -> Result<String, GgufError> {
        let len = self.read_u64()?;
        if len > MAX_STRING {
            return Err(GgufError::CountTooLarge {
                what: "string length",
                count: len,
                max: MAX_STRING,
            });
        }
        let at = self.pos;
        let raw = self.take(len)?;
        String::from_utf8(raw.to_vec()).map_err(|_| GgufError::BadUtf8 { at })
    }

    fn read_value(&mut self, type_id: u32, depth: u32) -> Result<Value, GgufError> {
        Ok(match type_id {
            0 => Value::U8(self.take(1)?[0]),
            1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(
                self.take(2)?.try_into().expect("2 bytes"),
            )),
            3 => Value::I16(i16::from_le_bytes(
                self.take(2)?.try_into().expect("2 bytes"),
            )),
            4 => Value::U32(self.read_u32()?),
            5 => Value::I32(self.read_u32()? as i32),
            6 => Value::F32(f32::from_le_bytes(
                self.take(4)?.try_into().expect("4 bytes"),
            )),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.read_string()?),
            9 => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(GgufError::ArrayTooDeep);
                }
                let elem_type = self.read_u32()?;
                let count = self.read_u64()?;
                if count > MAX_ARRAY {
                    return Err(GgufError::CountTooLarge {
                        what: "array",
                        count,
                        max: MAX_ARRAY,
                    });
                }
                // pre-allocation is capped independently of the declared count:
                // a hostile header can claim 16M entries but bytes run out fast
                let mut items = Vec::with_capacity(count.min(65_536) as usize);
                for _ in 0..count {
                    items.push(self.read_value(elem_type, depth + 1)?);
                }
                Value::Array(items)
            }
            10 => Value::U64(self.read_u64()?),
            11 => Value::I64(self.read_u64()? as i64),
            12 => Value::F64(f64::from_le_bytes(
                self.take(8)?.try_into().expect("8 bytes"),
            )),
            other => return Err(GgufError::UnknownValueType(other)),
        })
    }
}

#[cfg(test)]
mod tests;
