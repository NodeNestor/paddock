//! Minimal host tensor for the reference path. GGUF dim convention: dims[0]
//! is the fastest-moving (row length). A 2-D weight [in_dim, out_dim] is
//! out_dim rows of in_dim contiguous floats.

#[derive(Debug, Clone)]
pub struct HostTensor {
    pub data: Vec<f32>,
    pub dims: Vec<usize>,
}

impl HostTensor {
    pub fn new(data: Vec<f32>, dims: Vec<usize>) -> Self {
        debug_assert_eq!(data.len(), dims.iter().product::<usize>());
        Self { data, dims }
    }

    /// Row `r` of a 2-D tensor (contiguous dims[0] floats).
    pub fn row(&self, r: usize) -> &[f32] {
        let w = self.dims[0];
        &self.data[r * w..(r + 1) * w]
    }

    /// out += W·x for a [in, out] weight - `out` typically pre-loaded with the
    /// bias vector.
    pub fn matvec_add(&self, x: &[f32], out: &mut [f32]) {
        let in_dim = self.dims[0];
        debug_assert_eq!(x.len(), in_dim);
        debug_assert_eq!(out.len(), self.dims[1]);
        for (o, row) in out.iter_mut().zip(self.data.chunks_exact(in_dim)) {
            let mut acc = 0f32;
            for (a, b) in row.iter().zip(x) {
                acc += a * b;
            }
            *o += acc;
        }
    }

    /// y = W·x for a [in, out] weight: out[o] = dot(row o, x). Reference-grade
    /// loop; the GPU path owns performance.
    pub fn matvec(&self, x: &[f32], out: &mut [f32]) {
        let in_dim = self.dims[0];
        let out_dim = self.dims[1];
        debug_assert_eq!(x.len(), in_dim);
        debug_assert_eq!(out.len(), out_dim);
        for (o, out_slot) in out.iter_mut().enumerate() {
            let row = &self.data[o * in_dim..(o + 1) * in_dim];
            let mut acc = 0f32;
            for (a, b) in row.iter().zip(x) {
                acc += a * b;
            }
            *out_slot = acc;
        }
    }
}
