//! Test-only GGUF byte writer, shared by the parser, mapped and probe tests.
//! Intentionally dumb: it writes exactly what it's told, so tests can build
//! both well-formed and hostile inputs.

/// Minimal GGUF v3 writer.
pub(crate) struct Writer {
    pub(crate) buf: Vec<u8>,
}

impl Writer {
    pub(crate) fn new(tensor_count: u64, kv_count: u64) -> Self {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // "GGUF"
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&tensor_count.to_le_bytes());
        buf.extend_from_slice(&kv_count.to_le_bytes());
        Self { buf }
    }

    pub(crate) fn string(&mut self, s: &str) {
        self.buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub(crate) fn kv_str(&mut self, key: &str, val: &str) {
        self.string(key);
        self.buf.extend_from_slice(&8u32.to_le_bytes());
        self.string(val);
    }

    pub(crate) fn kv_u16(&mut self, key: &str, val: u16) {
        self.string(key);
        self.buf.extend_from_slice(&2u32.to_le_bytes());
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    pub(crate) fn kv_u32(&mut self, key: &str, val: u32) {
        self.string(key);
        self.buf.extend_from_slice(&4u32.to_le_bytes());
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    /// gguf-split writes split.tensors.count as i32 - tests use this to keep
    /// the width-agnostic metadata lookups honest.
    pub(crate) fn kv_i32(&mut self, key: &str, val: i32) {
        self.string(key);
        self.buf.extend_from_slice(&5u32.to_le_bytes());
        self.buf.extend_from_slice(&val.to_le_bytes());
    }

    /// Per-layer flag arrays (gemma4's sliding-window pattern is one).
    pub(crate) fn kv_bool_array(&mut self, key: &str, vals: &[bool]) {
        self.string(key);
        self.buf.extend_from_slice(&9u32.to_le_bytes()); // array
        self.buf.extend_from_slice(&7u32.to_le_bytes()); // of bools
        self.buf
            .extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.buf.push(u8::from(*v));
        }
    }

    pub(crate) fn kv_str_array(&mut self, key: &str, vals: &[&str]) {
        self.string(key);
        self.buf.extend_from_slice(&9u32.to_le_bytes()); // array
        self.buf.extend_from_slice(&8u32.to_le_bytes()); // of strings
        self.buf
            .extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            self.string(v);
        }
    }

    /// F32 tensor descriptor; offset must respect the default 32-byte alignment.
    pub(crate) fn tensor_f32(&mut self, name: &str, dims: &[u64], offset: u64) {
        self.string(name);
        self.buf
            .extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            self.buf.extend_from_slice(&d.to_le_bytes());
        }
        self.buf.extend_from_slice(&0u32.to_le_bytes()); // F32
        self.buf.extend_from_slice(&offset.to_le_bytes());
    }

    /// Pad to the data-section alignment boundary, then append `data_len`
    /// bytes of tensor data (zeros unless `fill` says otherwise).
    pub(crate) fn finish_with_data(self, data_len: usize) -> Vec<u8> {
        self.finish_with_filled_data(data_len, 0)
    }

    pub(crate) fn finish_with_filled_data(mut self, data_len: usize, fill: u8) -> Vec<u8> {
        let aligned = self.buf.len().div_ceil(32) * 32;
        self.buf.resize(aligned, 0);
        self.buf.resize(aligned + data_len, fill);
        self.buf
    }
}
