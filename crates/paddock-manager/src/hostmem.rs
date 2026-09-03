//! Total physical RAM on this machine, and what the fleet has committed of it.
//!
//! The manager has always priced VRAM and never host memory, because nothing
//! it ran spent host memory at a scale worth a fit check. Prefix-cache offload
//! does: `[kv_offload] ram_gb` is a page-locked commitment that grows to its
//! ceiling under the workload the feature exists for. A form that lets an
//! operator type 128 on a 32 GB box without a word is the VRAM over-promise
//! this project already refuses, one resource over.
//!
//! `Option` everywhere, matching the NVML stance next door: a platform whose
//! total we cannot read reports None and the surface simply omits the
//! denominator, rather than guessing one.

/// Total physical RAM in bytes, or None where we cannot ask.
pub fn total_bytes() -> Option<u64> {
    #[cfg(windows)]
    {
        // GlobalMemoryStatusEx: the documented total-physical read, and one
        // the manager can make without a new dependency (windows-sys is
        // already here for job objects).
        use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut st: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        st.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        // SAFETY: dwLength is set as the API requires; the struct is ours.
        if unsafe { GlobalMemoryStatusEx(&mut st) } != 0 {
            return Some(st.ullTotalPhys);
        }
        None
    }
    #[cfg(target_os = "linux")]
    {
        // /proc/meminfo is stable, text, and needs no crate. MemTotal is in
        // kibibytes by definition of the file's format.
        let txt = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = txt.lines().find(|l| l.starts_with("MemTotal:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        // macOS would take a sysctl; until the fit surface is wanted there,
        // reporting nothing beats reporting a guess.
        None
    }
}

/// Host RAM the running fleet has already promised to prefix caches, summed
/// from each endpoint's own config. A ceiling, not a measurement - which is
/// the right thing to subtract when asking "can I also give this one 24 GB",
/// because every one of those ceilings is reachable at once.
pub fn committed_bytes(specs: impl Iterator<Item = Option<f64>>) -> u64 {
    specs
        .flatten()
        .filter(|g| *g > 0.0)
        .map(|g| (g * (1u64 << 30) as f64) as u64)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn this_box_reports_a_plausible_total() {
        // The two platforms we serve on must answer; a wrong answer here is
        // worse than none, so sanity-bound it rather than just unwrapping.
        #[cfg(any(windows, target_os = "linux"))]
        {
            let t = total_bytes().expect("a supported platform must report its RAM");
            assert!(t >= 1 << 30, "implausibly small: {t}");
            assert!(t < 1 << 50, "implausibly large: {t}");
        }
    }

    #[test]
    fn commitments_sum_and_ignore_the_absent() {
        let got = committed_bytes([Some(8.0), None, Some(0.0), Some(16.0)].into_iter());
        assert_eq!(got, 24 << 30);
        assert_eq!(committed_bytes(std::iter::empty()), 0);
    }
}
