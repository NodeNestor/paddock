//! The kv-offload tier's DEVICE-side geometry, in the one crate both the
//! engine and the estimator can see.
//!
//! The engine reserves these bytes out of the KV pool the moment a tier is
//! armed (`kv_plan`'s "kv-tier staging" reserve), and the manager's fit
//! estimate has to subtract exactly the same number or its answer stops
//! matching what the runner seats. Two copies of a constant is how that goes
//! wrong quietly a release later, so there is one.
//!
//! Host-side capacity is deliberately not here: that is a user budget
//! (`[kv_offload] ram_gb`), not engine geometry.

/// One staging extent. Sized from R1's transfer ladder: extents of 2-16 MiB
/// ride the bus at ~97% of ceiling, and 32 MiB gives the packer room for the
/// largest run a family can produce without a second round trip.
pub const STAGING_EXTENT_BYTES: u64 = 32 << 20;

/// How many extents the ring holds. Two is the minimum that lets one flight
/// bounce through the ring while the next is being filled - the pipelining
/// the transport's kick loop depends on.
pub const STAGING_EXTENTS: u64 = 2;

/// Device VRAM an armed tier holds for staging, whatever else it is doing.
/// Conditional on the tier actually being armed: an untiered model pays none
/// of it, which is why the engine's reserve and the estimate's subtraction
/// are both gated on the same condition.
pub const fn device_staging_bytes() -> u64 {
    STAGING_EXTENTS * STAGING_EXTENT_BYTES
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reserve the estimate cannot express in whole MiB would make the two
    /// sides disagree by rounding alone.
    #[test]
    fn the_reserve_is_a_whole_number_of_mib() {
        assert_eq!(device_staging_bytes() % (1 << 20), 0);
        assert_eq!(device_staging_bytes(), 64 << 20);
    }
}
