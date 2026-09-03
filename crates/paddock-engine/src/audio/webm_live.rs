//! Fill in the element sizes a live WebM muxer never wrote.
//!
//! A browser recording a microphone cannot go back and write down how big
//! anything is - it is streaming the file out as it is made - so `MediaRecorder`
//! emits the shape EBML has for exactly this case: `Segment` and every `Cluster`
//! carry the reserved "unknown size" vint, and a parent ends when an element
//! turns up that could only belong to an ancestor. It is legal Matroska, it is
//! what every browser on earth produces, and ffmpeg/whisper.cpp/the providers
//! all read it.
//!
//! symphonia 0.6.0 does not. Its unknown-size escape hatch (`ebml.rs`, the
//! ancestor walk in `next_header`) asks whether the surprising element would be
//! valid at the ANCESTOR'S own DEPTH:
//!
//! ```ignore
//! if header.is_valid_at(ancestor.depth, ancestor.id).unwrap_or(false)
//! ```
//!
//! but a child of that ancestor sits one level DEEPER, so the test needs
//! `ancestor.depth + 1`. `Cluster` has `min_depth = 1` and `Segment` sits at
//! depth 0, so `1 <= 0` is false, the hatch can never fire, and the reader
//! calls the perfectly ordinary next cluster an `UnexpectedElement`. Every
//! browser recording past its first cluster - about two seconds - dies there.
//! (Measured on a Chrome clip: `unexpected element Cluster`.)
//!
//! So we seal the sizes before symphonia sees the bytes. This is deliberately
//! the SMALLEST fix that is still correct: we walk element headers only, and we
//! do not touch a single byte of audio, timestamp, lacing, pre-skip or codec
//! delay - all of that stays symphonia's and libopus's, which is the whole
//! reason `decode.rs` reuses them (our PCM matches the arbiter's bit for bit).
//! A file that already has its sizes is not copied and not modified.
//!
//! Delete this when symphonia's ancestor walk is fixed; the tests below then
//! become a check that it still works without us.

/// The two ids we have to know by name. Everything else we only need to
/// classify, and unknown ids are simply skipped by their declared size.
const SEGMENT: u32 = 0x1853_8067;
const CLUSTER: u32 = 0x1F43_B675;

/// Can this element sit directly inside a `Cluster`? That question is the end
/// of an unknown-size cluster: the first element that cannot is the next
/// sibling (another `Cluster`, or `Cues`/`Tags` at the tail of the segment).
///
/// `Void` and `CRC-32` are EBML globals - legal anywhere, so they never end a
/// cluster.
fn cluster_child(id: u32) -> bool {
    matches!(
        id,
        0xE7        // Timestamp
        | 0x5854    // SilentTracks
        | 0xA7      // Position
        | 0xAB      // PrevSize
        | 0xA3      // SimpleBlock
        | 0xA0      // BlockGroup
        | 0xAF      // EncryptedBlock
        | 0xEC      // Void (global)
        | 0xBF // CRC-32 (global)
    )
}

/// Read an element id, marker bit and all - ids are compared as written, so
/// unlike a size the marker is part of the value. Ids are 1-4 bytes.
fn read_id(b: &[u8], p: usize) -> Option<(u32, usize)> {
    let first = *b.get(p)?;
    let w = first.leading_zeros() as usize + 1;
    if w > 4 || p + w > b.len() {
        return None;
    }
    let mut v = 0u32;
    for i in 0..w {
        v = (v << 8) | b[p + i] as u32;
    }
    Some((v, p + w))
}

/// Read a size vint: `(size, width)`, where `None` is EBML's "unknown" - every
/// value bit set. Width matters as much as the value, because sealing has to
/// write the real size back into the same bytes.
fn read_size(b: &[u8], p: usize) -> Option<(Option<u64>, usize)> {
    let first = *b.get(p)?;
    let w = first.leading_zeros() as usize + 1;
    if w > 8 || p + w > b.len() {
        return None;
    }
    // The marker bit is bit (8 - w) of the first byte; the bits above it are
    // value. For w == 8 the first byte is pure marker and contributes nothing.
    let mut v = (first as u64) & (0xFFu64 >> w);
    for i in 1..w {
        v = (v << 8) | b[p + i] as u64;
    }
    let unknown = v == (1u64 << (7 * w)) - 1;
    Some((if unknown { None } else { Some(v) }, w))
}

/// Write a real size into the `w` bytes an unknown size occupied. Fails (and
/// then we leave the whole file alone) if the value cannot be expressed in that
/// width - the all-ones pattern is spoken for, so the last representable size
/// is one below it.
fn write_size(out: &mut [u8], p: usize, w: usize, v: u64) -> bool {
    if w == 0 || w > 8 || p + w > out.len() || v >= (1u64 << (7 * w)) - 1 {
        return false;
    }
    let mut x = v;
    for i in (0..w).rev() {
        out[p + i] = (x & 0xFF) as u8;
        x >>= 8;
    }
    out[p] |= 0x80 >> (w - 1);
    true
}

/// Where an unknown-size cluster actually ends: walk its children until one
/// turns up that cannot be a cluster child.
///
/// Anything we cannot read hands the rest of the segment to the cluster, which
/// is the right answer for the one way this really happens - a recording cut
/// off mid-block. symphonia then reports the torn tail itself, honestly, having
/// decoded everything before it.
fn cluster_end(b: &[u8], mut p: usize, end: usize) -> usize {
    while p < end {
        let Some((id, after_id)) = read_id(b, p) else {
            return end;
        };
        if !cluster_child(id) {
            return p;
        }
        let Some((size, w)) = read_size(b, after_id) else {
            return end;
        };
        // A cluster CHILD with an unknown size is not a thing any muxer emits
        // and not something we could measure - stop the cluster before it
        // rather than guess.
        let Some(n) = size else { return p };
        let next = (after_id + w).saturating_add(n as usize);
        if next > end {
            return end;
        }
        p = next;
    }
    end
}

/// Walk a segment's children, noting every unknown-size cluster.
fn scan_segment(b: &[u8], mut p: usize, end: usize, patches: &mut Vec<(usize, usize, u64)>) {
    while p < end {
        let Some((id, after_id)) = read_id(b, p) else {
            return;
        };
        let Some((size, w)) = read_size(b, after_id) else {
            return;
        };
        let data = after_id + w;
        if data > end {
            return;
        }
        match size {
            Some(n) => p = data.saturating_add(n as usize).min(end),
            None => {
                // Only a cluster is allowed to be unknown-sized down here. If
                // something else is, this is not a shape we understand and
                // guessing at it would be worse than handing symphonia the
                // original bytes and letting it say so.
                if id != CLUSTER {
                    return;
                }
                let stop = cluster_end(b, data, end);
                patches.push((after_id, w, (stop - data) as u64));
                p = stop;
            }
        }
    }
}

/// Seal a live-muxed WebM/Matroska stream. `None` means nothing needed sealing
/// (the overwhelmingly common case - every file from ffmpeg and friends), and
/// the caller should use the bytes it already has.
pub fn seal_live_sizes(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut patches: Vec<(usize, usize, u64)> = Vec::new();
    let mut p = 0usize;

    while p < bytes.len() {
        let Some((id, after_id)) = read_id(bytes, p) else {
            break;
        };
        let Some((size, w)) = read_size(bytes, after_id) else {
            break;
        };
        let data = after_id + w;
        if data > bytes.len() {
            break;
        }
        match size {
            Some(n) => {
                let end = data.saturating_add(n as usize).min(bytes.len());
                if id == SEGMENT {
                    scan_segment(bytes, data, end, &mut patches);
                }
                p = end;
            }
            None => {
                // At the top level only the segment may be unknown-sized, and
                // it then owns the rest of the file. Sealing it too is what
                // lets iteration end on "reached the parent's end" instead of
                // running off the back of the buffer.
                if id != SEGMENT {
                    break;
                }
                patches.push((after_id, w, (bytes.len() - data) as u64));
                scan_segment(bytes, data, bytes.len(), &mut patches);
                p = bytes.len();
            }
        }
    }

    if patches.is_empty() {
        return None;
    }
    let mut out = bytes.to_vec();
    for (at, w, v) in patches {
        if !write_size(&mut out, at, w, v) {
            // Cannot express one of the sizes in the space we have. Partly
            // sealing would be worse than not sealing at all.
            return None;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An unknown size of `w` bytes: the marker bit, then all ones.
    fn unknown(w: usize) -> Vec<u8> {
        let mut v = vec![0xFFu8; w];
        // Widened deliberately: at w == 8 the first byte keeps no value bits at
        // all, and `0xFFu8 >> 8` is an overflow rather than the 0 we want.
        v[0] = ((0x80u16 >> (w - 1)) | (0xFFu16 >> w)) as u8;
        v
    }

    fn elem(id: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut v = id.to_vec();
        // 1-byte size is enough for these hand-built payloads
        assert!(payload.len() < 0x7F);
        v.push(0x80 | payload.len() as u8);
        v.extend_from_slice(payload);
        v
    }

    /// Segment and clusters unknown-sized, exactly as a browser writes them.
    fn live_file() -> Vec<u8> {
        let mut cluster = Vec::new();
        cluster.extend_from_slice(&[0x1F, 0x43, 0xB6, 0x75]);
        cluster.extend_from_slice(&unknown(8));
        cluster.extend_from_slice(&elem(&[0xE7], &[0x00])); // Timestamp
        cluster.extend_from_slice(&elem(&[0xA3], &[1, 2, 3, 4])); // SimpleBlock
        cluster.extend_from_slice(&elem(&[0xA3], &[5, 6, 7, 8]));

        let mut seg = Vec::new();
        seg.extend_from_slice(&[0x18, 0x53, 0x80, 0x67]);
        seg.extend_from_slice(&unknown(8));
        seg.extend_from_slice(&elem(&[0x16, 0x54, 0xAE, 0x6B], &[0x80])); // Tracks
        seg.extend_from_slice(&cluster);
        seg.extend_from_slice(&cluster);

        let mut f = elem(&[0x1A, 0x45, 0xDF, 0xA3], &[0x42, 0x86, 0x81, 0x01]);
        f.extend_from_slice(&seg);
        f
    }

    /// Walk the sealed bytes and collect (id, size) for every top-level and
    /// segment-level element, so a test can assert on the shape rather than on
    /// byte offsets it would have to be rewritten for.
    fn sizes(b: &[u8]) -> Vec<(u32, Option<u64>)> {
        let mut out = Vec::new();
        let mut p = 0;
        while p < b.len() {
            let Some((id, ai)) = read_id(b, p) else { break };
            let Some((size, w)) = read_size(b, ai) else {
                break;
            };
            out.push((id, size));
            let data = ai + w;
            let n = match size {
                Some(n) => n as usize,
                None => break,
            };
            if id == SEGMENT {
                // descend
                let end = (data + n).min(b.len());
                let mut q = data;
                while q < end {
                    let Some((cid, cai)) = read_id(b, q) else {
                        break;
                    };
                    let Some((csize, cw)) = read_size(b, cai) else {
                        break;
                    };
                    out.push((cid, csize));
                    let Some(cn) = csize else { break };
                    q = cai + cw + cn as usize;
                }
            }
            p = data + n;
        }
        out
    }

    #[test]
    fn a_live_muxed_segment_and_every_cluster_get_their_true_sizes() {
        let f = live_file();
        let sealed = seal_live_sizes(&f).expect("a live file must need sealing");
        // Sealing rewrites sizes in place - it must never change the length,
        // because every offset already written into the file (SeekHead, Cues)
        // would otherwise be a lie.
        assert_eq!(sealed.len(), f.len());

        let got = sizes(&sealed);
        assert!(
            got.iter().all(|(_, s)| s.is_some()),
            "nothing may still be unknown: {got:?}"
        );

        let seg = got
            .iter()
            .find(|(id, _)| *id == SEGMENT)
            .expect("segment")
            .1
            .unwrap();
        // The segment owns everything after the EBML header element (4-byte id
        // + 1-byte size + 4-byte payload) and its own header (4 + 8).
        assert_eq!(seg as usize, f.len() - (9 + 12));

        let clusters: Vec<u64> = got
            .iter()
            .filter(|(id, _)| *id == CLUSTER)
            .map(|(_, s)| s.unwrap())
            .collect();
        assert_eq!(clusters.len(), 2, "both clusters must be found: {got:?}");
        // Timestamp (1 + 1 + 1) + two SimpleBlocks (1 + 1 + 4 each) = 15.
        assert_eq!(clusters, vec![15, 15]);
    }

    /// The end of an unknown-size cluster is the first element that could not
    /// be its child - here the trailing Cues, which is what ffmpeg-shaped files
    /// put after the last cluster.
    #[test]
    fn a_trailing_cues_ends_the_last_cluster_rather_than_joining_it() {
        let mut f = live_file();
        let before = f.len();
        f.extend_from_slice(&elem(&[0x1C, 0x53, 0xBB, 0x6B], &[0x80]));
        let sealed = seal_live_sizes(&f).expect("still needs sealing");
        let got = sizes(&sealed);
        let clusters: Vec<u64> = got
            .iter()
            .filter(|(id, _)| *id == CLUSTER)
            .map(|(_, s)| s.unwrap())
            .collect();
        assert_eq!(
            clusters,
            vec![15, 15],
            "Cues must not be swallowed: {got:?}"
        );
        assert_eq!(f.len(), before + 6);
    }

    #[test]
    fn a_file_that_already_has_its_sizes_is_left_completely_alone() {
        let sealed = seal_live_sizes(&live_file()).expect("live file seals");
        assert!(
            seal_live_sizes(&sealed).is_none(),
            "sealing must be idempotent, and the second pass must not copy"
        );
    }

    #[test]
    fn nothing_that_is_not_a_live_matroska_stream_is_touched() {
        assert!(seal_live_sizes(b"").is_none());
        assert!(seal_live_sizes(b"RIFF\0\0\0\0WAVEfmt ").is_none());
        assert!(
            seal_live_sizes(b"\x1a\x45\xdf\xa3").is_none(),
            "a truncated header seals nothing"
        );
        // An unknown-size element that is not a segment is a shape we do not
        // claim to understand - hand it back untouched rather than guess.
        let mut odd = vec![0x18, 0x53, 0x80, 0x68];
        odd.extend_from_slice(&unknown(8));
        assert!(seal_live_sizes(&odd).is_none());
    }

    #[test]
    fn sizes_round_trip_through_every_vint_width() {
        for w in 1..=8usize {
            let cap = (1u64 << (7 * w)) - 1;
            for v in [0u64, 1, cap / 2, cap - 1] {
                let mut buf = vec![0u8; w];
                assert!(write_size(&mut buf, 0, w, v), "w={w} v={v}");
                assert_eq!(read_size(&buf, 0), Some((Some(v), w)), "w={w} v={v}");
            }
            // The all-ones pattern is "unknown" and must never be written as a
            // real size, nor read back as one.
            let mut buf = vec![0u8; w];
            assert!(
                !write_size(&mut buf, 0, w, cap),
                "w={w}: all-ones is reserved"
            );
            assert_eq!(read_size(&unknown(w), 0), Some((None, w)));
        }
    }
}
