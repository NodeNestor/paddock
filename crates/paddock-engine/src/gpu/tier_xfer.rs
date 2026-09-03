//! KV tier extent gather/scatter launches - the
//! device half of the demote/restore data plane. The kernels rearrange
//! scattered paged-pool blocks into page-first contiguous extents in device
//! staging (and back), so the PCIe leg is always one >=2 MiB DMA: per-page
//! fragments measured 5% of the bus, contiguous extents 97%.
//!
//! The RAM transport launches these on its own forked lane (`fork_stream`),
//! event-fenced against the compute stream; nothing here touches the
//! serving path's stream.

use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::error::*;
use super::*;

impl GpuExecutor {
    /// Whether the loaded pack carries the tier gather/scatter pair - the
    /// capability probe the tier layer keys on. A stale pack reads absent
    /// and the tier declines loudly instead of failing at first demote.
    pub fn has_kv_tier_xfer(&self) -> bool {
        self.kernels.kv_gather_blocks.is_some() && self.kernels.kv_scatter_blocks.is_some()
    }

    /// Gather `n_blocks` pool blocks x `n_planes` planes into a contiguous
    /// extent. `planes` holds device 4-tuples {base, stride, bytes, dst_off}
    /// (u64), `block_ids` the pool block ids; the TRANSPORT validated the
    /// 16-multiple contract when it built the descriptors. Capacity is
    /// checked here - a too-small extent is a caller bug said in full.
    pub fn kv_gather_blocks(
        &self,
        planes: &CudaSlice<u64>,
        block_ids: &CudaSlice<u32>,
        extent: &mut CudaSlice<u8>,
        record_stride: u64,
        max_plane_bytes: u64,
        n_planes: usize,
        n_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kv_gather_blocks
            .ok_or(GpuError::MissingOp("kv_gather_blocks"))?;
        self.kv_xfer_check(
            planes,
            block_ids,
            extent.len(),
            record_stride,
            n_planes,
            n_blocks,
        )?;
        let (pp, _g1) = planes.device_ptr(&self.stream);
        let (bp, _g2) = block_ids.device_ptr(&self.stream);
        let (ep, _g3) = extent.device_ptr_mut(&self.stream);
        // SAFETY: ABI contract; descriptor/id/extent capacities checked above
        check(unsafe {
            f(
                pp as *const _,
                bp as *const _,
                ep as *mut _,
                record_stride,
                max_plane_bytes,
                n_planes as u32,
                n_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    /// The restore-direction twin: extent back into pool blocks.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_scatter_blocks(
        &self,
        planes: &CudaSlice<u64>,
        block_ids: &CudaSlice<u32>,
        extent: &CudaSlice<u8>,
        record_stride: u64,
        max_plane_bytes: u64,
        n_planes: usize,
        n_blocks: usize,
    ) -> Result<(), GpuError> {
        let f = self
            .kernels
            .kv_scatter_blocks
            .ok_or(GpuError::MissingOp("kv_scatter_blocks"))?;
        self.kv_xfer_check(
            planes,
            block_ids,
            extent.len(),
            record_stride,
            n_planes,
            n_blocks,
        )?;
        let (pp, _g1) = planes.device_ptr(&self.stream);
        let (bp, _g2) = block_ids.device_ptr(&self.stream);
        let (ep, _g3) = extent.device_ptr(&self.stream);
        // SAFETY: ABI contract; descriptor/id/extent capacities checked above.
        // The shared fn type says *mut for the extent; the scatter kernel
        // only reads it (its C signature is const) - the cast is FFI shape,
        // not a mutation.
        check(unsafe {
            f(
                pp as *const _,
                bp as *const _,
                ep as *mut _,
                record_stride,
                max_plane_bytes,
                n_planes as u32,
                n_blocks as u32,
                self.stream_ptr(),
            )
        })
    }

    fn kv_xfer_check(
        &self,
        planes: &CudaSlice<u64>,
        block_ids: &CudaSlice<u32>,
        extent_len: usize,
        record_stride: u64,
        n_planes: usize,
        n_blocks: usize,
    ) -> Result<(), GpuError> {
        if planes.len() < n_planes * 4 {
            return Err(oob("kv_xfer: plane descriptor buffer under n_planes*4"));
        }
        if block_ids.len() < n_blocks {
            return Err(oob("kv_xfer: block id buffer under n_blocks"));
        }
        if (extent_len as u64) < n_blocks as u64 * record_stride {
            return Err(oob("kv_xfer: extent under n_blocks * record_stride"));
        }
        Ok(())
    }
}
