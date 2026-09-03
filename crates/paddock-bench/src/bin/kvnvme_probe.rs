//! kvnvme-probe - the KV-offload NVMe ceiling probe
//! (Phase R): the local-NVMe facts
//! the T2 tier design elects from, measured with the OS cache out of the way.
//!
//!   0. alignment discovery - the unbuffered-IO granularity this device
//!      actually demands (never assumed; plan D7).
//!   1. sequential write ladder 256 KiB -> 16 MiB (also builds the test file -
//!      NTFS returns zeros without device IO for never-written ranges, so
//!      reads below only touch written spans).
//!   2. sequential read ladder, same chunks.
//!   3. random read: chunk × queue depth (thread-pool positioned reads) -
//!      the restore path's shape (partial hits at extent granularity).
//!   4. read/write contention - Tutti (arXiv 2605.03375) reports concurrent
//!      NVMe read/write collapsing bandwidth ~60%; measure our device, since
//!      "defer writes to read-slack" is a scheduling election, not a law.
//!   5. flush latency - the WAL group-commit budget (persistent-format spec).
//!
//! Windows: FILE_FLAG_NO_BUFFERING positioned IO (seek_read/seek_write).
//! Unix: O_DIRECT + read_at/write_at. Queue depth is emulated with threads -
//! it measures what the device can do; the production backend (overlapped /
//! IORing / io_uring) is a separate Phase-3 election measured against these
//! same ceilings.
//!
//! Usage: kvnvme-probe <dir-on-target-drive> [--file-gb N] [--keep]

use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[cfg(unix)]
use std::os::unix::fs::{FileExt, OpenOptionsExt};
#[cfg(windows)]
use std::os::windows::fs::{FileExt, OpenOptionsExt};

const KIB: u64 = 1024;
const MIB: u64 = 1024 * 1024;

/// Buffer aligned to `align` bytes (unbuffered IO wants the MEMORY aligned
/// too, not just the offsets).
struct AlignedBuf {
    raw: Vec<u8>,
    off: usize,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Self {
        let mut raw = vec![0u8; len + align];
        let addr = raw.as_ptr() as usize;
        let off = (align - (addr % align)) % align;
        // deterministic non-zero content so device-side compression (some
        // consumer controllers) can't fake sequential-write numbers with
        // all-zero blocks
        for (i, b) in raw[off..off + len].iter_mut().enumerate() {
            *b = (i as u8) ^ ((i >> 9) as u8) ^ 0x5a;
        }
        AlignedBuf { raw, off, len }
    }
    fn slice(&self) -> &[u8] {
        &self.raw[self.off..self.off + self.len]
    }
    fn slice_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }
}

fn open_direct(path: &Path, create: bool) -> std::io::Result<File> {
    let mut o = OpenOptions::new();
    o.read(true).write(true).create(create);
    #[cfg(windows)]
    {
        // FILE_FLAG_NO_BUFFERING: bypass the cache manager entirely.
        o.custom_flags(0x2000_0000);
    }
    #[cfg(unix)]
    {
        o.custom_flags(libc::O_DIRECT);
    }
    o.open(path)
}

fn pread(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
    #[cfg(windows)]
    return f.seek_read(buf, off);
    #[cfg(unix)]
    return f.read_at(buf, off);
}

fn pwrite(f: &File, buf: &[u8], off: u64) -> std::io::Result<usize> {
    #[cfg(windows)]
    return f.seek_write(buf, off);
    #[cfg(unix)]
    return f.write_at(buf, off);
}

fn gbs(bytes: u64, secs: f64) -> f64 {
    bytes as f64 / secs / 1e9
}

/// xorshift64* - deterministic offsets without a rand dep.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut dir: Option<PathBuf> = None;
    let mut file_gb: u64 = 8;
    let mut keep = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--file-gb" => {
                i += 1;
                file_gb = args[i].parse().expect("--file-gb N");
            }
            "--keep" => keep = true,
            a => dir = Some(PathBuf::from(a)),
        }
        i += 1;
    }
    let dir = dir.expect("usage: kvnvme-probe <dir-on-target-drive> [--file-gb N] [--keep]");
    fs::create_dir_all(&dir).expect("create target dir");
    let path = dir.join("kvnvme-probe.dat");
    let file_bytes = file_gb * 1024 * MIB;
    println!(
        "kvnvme-probe: {} ({} GiB test file)",
        path.display(),
        file_gb
    );

    // -- 0. alignment discovery ---------------------------------------------
    let align = {
        let f = open_direct(&path, true).expect("open direct");
        f.set_len(1024 * KIB).unwrap();
        let mut found = None;
        for cand in [512usize, 4096, 65536] {
            let mut b = AlignedBuf::new(cand, cand);
            // a write at this granularity must land; a read must come back
            if pwrite(&f, b.slice(), 0).is_ok() && pread(&f, b.slice_mut(), 0).is_ok() {
                found = Some(cand);
                break;
            }
        }
        let a = found.expect("no unbuffered alignment in {512,4K,64K} worked - record and abort");
        println!("\n== 0. unbuffered alignment: {} bytes", a);
        a
    };

    // -- 1+2. sequential ladders --------------------------------------------
    // the file is split into one region per chunk size; the write pass fills
    // every region so later reads touch real data (NTFS valid-data-length)
    let chunks: [u64; 4] = [256 * KIB, MIB, 4 * MIB, 16 * MIB];
    let region = file_bytes / chunks.len() as u64;
    {
        let f = open_direct(&path, true).expect("open direct");
        f.set_len(file_bytes).unwrap();
        println!("\n== 1. sequential WRITE (QD1, GB/s)");
        println!("{:>10} {:>10}", "chunk KiB", "GB/s");
        for (r, &chunk) in chunks.iter().enumerate() {
            let buf = AlignedBuf::new(chunk as usize, align);
            let base = r as u64 * region;
            let t0 = Instant::now();
            let mut off = 0;
            while off + chunk <= region {
                let n = pwrite(&f, buf.slice(), base + off).expect("write");
                assert_eq!(n as u64, chunk, "short write mid-ladder");
                off += chunk;
            }
            f.sync_data().unwrap();
            println!(
                "{:>10} {:>10.2}",
                chunk / KIB,
                gbs(off, t0.elapsed().as_secs_f64())
            );
        }
        println!("\n== 2. sequential READ (QD1, GB/s)");
        println!("{:>10} {:>10}", "chunk KiB", "GB/s");
        for (r, &chunk) in chunks.iter().enumerate() {
            let mut buf = AlignedBuf::new(chunk as usize, align);
            let base = r as u64 * region;
            let t0 = Instant::now();
            let mut off = 0;
            while off + chunk <= region {
                let n = pread(&f, buf.slice_mut(), base + off).expect("read");
                assert_eq!(n as u64, chunk);
                off += chunk;
            }
            println!(
                "{:>10} {:>10.2}",
                chunk / KIB,
                gbs(off, t0.elapsed().as_secs_f64())
            );
        }
    }

    // -- 3. random read: chunk x queue depth --------------------------------
    println!("\n== 3. random READ (GB/s; threads emulate queue depth)");
    println!(
        "{:>10} {:>8} {:>8} {:>8}",
        "chunk KiB", "qd1", "qd4", "qd16"
    );
    for &chunk in &[256 * KIB, MIB, 4 * MIB] {
        print!("{:>10}", chunk / KIB);
        for qd in [1usize, 4, 16] {
            let per_thread = (512 * MIB / qd as u64 / chunk).max(8);
            let total = per_thread * qd as u64 * chunk;
            let t0 = Instant::now();
            std::thread::scope(|s| {
                for t in 0..qd {
                    let path = &path;
                    s.spawn(move || {
                        let f = open_direct(path, false).expect("open");
                        let mut buf = AlignedBuf::new(chunk as usize, align);
                        let mut rng = Rng(0x9E3779B97F4A7C15 ^ (t as u64 + 1));
                        let slots = (file_bytes - chunk) / chunk;
                        for _ in 0..per_thread {
                            let off = (rng.next() % slots) * chunk;
                            let n = pread(&f, buf.slice_mut(), off).expect("read");
                            assert_eq!(n as u64, chunk);
                        }
                    });
                }
            });
            print!(" {:>8.2}", gbs(total, t0.elapsed().as_secs_f64()));
        }
        println!();
    }

    // -- 4. read/write contention -------------------------------------------
    // 4 readers (1 MiB random) alone, then with 2 writers (4 MiB random-slot).
    println!("\n== 4. read/write contention (GB/s)");
    let read_chunk = MIB;
    let write_chunk = 4 * MIB;
    let run_readers = |with_writers: bool| -> (f64, f64) {
        let stop = std::sync::atomic::AtomicBool::new(false);
        let read_bytes = std::sync::atomic::AtomicU64::new(0);
        let write_bytes = std::sync::atomic::AtomicU64::new(0);
        let t0 = Instant::now();
        std::thread::scope(|s| {
            for t in 0..4usize {
                let (path, stop, read_bytes) = (&path, &stop, &read_bytes);
                s.spawn(move || {
                    let f = open_direct(path, false).expect("open");
                    let mut buf = AlignedBuf::new(read_chunk as usize, align);
                    let mut rng = Rng(0xDEADBEEF ^ (t as u64 + 1));
                    let slots = (file_bytes - read_chunk) / read_chunk;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        let off = (rng.next() % slots) * read_chunk;
                        pread(&f, buf.slice_mut(), off).expect("read");
                        read_bytes.fetch_add(read_chunk, std::sync::atomic::Ordering::Relaxed);
                    }
                });
            }
            if with_writers {
                for t in 0..2usize {
                    let (path, stop, write_bytes) = (&path, &stop, &write_bytes);
                    s.spawn(move || {
                        let f = open_direct(path, false).expect("open");
                        let buf = AlignedBuf::new(write_chunk as usize, align);
                        let mut rng = Rng(0xC0FFEE ^ (t as u64 + 1));
                        let slots = (file_bytes - write_chunk) / write_chunk;
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            let off = (rng.next() % slots) * write_chunk;
                            pwrite(&f, buf.slice(), off).expect("write");
                            write_bytes
                                .fetch_add(write_chunk, std::sync::atomic::Ordering::Relaxed);
                        }
                    });
                }
            }
            std::thread::sleep(std::time::Duration::from_secs(8));
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        let secs = t0.elapsed().as_secs_f64();
        (
            gbs(read_bytes.load(std::sync::atomic::Ordering::Relaxed), secs),
            gbs(write_bytes.load(std::sync::atomic::Ordering::Relaxed), secs),
        )
    };
    let (r_alone, _) = run_readers(false);
    let (r_mixed, w_mixed) = run_readers(true);
    println!(
        "readers alone: {:.2}  |  readers+writers: {:.2} read + {:.2} write  ({:.0}% read retained)",
        r_alone,
        r_mixed,
        w_mixed,
        100.0 * r_mixed / r_alone
    );

    // -- 5. flush latency ----------------------------------------------------
    println!("\n== 5. flush latency (write + sync_data, ms, median of 21)");
    println!("{:>10} {:>10}", "chunk KiB", "ms");
    {
        let f = open_direct(&path, false).expect("open");
        for &chunk in &[4 * KIB, 64 * KIB, MIB] {
            let buf = AlignedBuf::new(chunk.max(align as u64) as usize, align);
            let mut times: Vec<f64> = (0..21)
                .map(|i| {
                    let t0 = Instant::now();
                    pwrite(&f, buf.slice(), (i as u64) * 16 * MIB).expect("write");
                    f.sync_data().unwrap();
                    t0.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("{:>10} {:>10.2}", chunk / KIB, times[10]);
        }
    }

    if !keep {
        drop(fs::remove_file(&path));
    }
    println!("\ndone.");
    let _ = std::io::stdout().flush();
}
