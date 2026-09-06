//! The original reference and its gates were contributed by NodeNestor
//! (github.com/Nodenester) in truespar/paddock PR #17; this rewrite keeps
//! that API and its bit-identity contract.
//! CPU reference dequant for the ggml i-quant family (IQ1_S, IQ1_M, IQ2_XXS,
//! IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), IQ4_NL and the two low-bit k-quants
//! (Q2_K, Q3_K). Written from the block layouts (ggml-common.h - the byte
//! order and the codebooks in `iq_grids.rs` are the format itself), not from
//! anyone's decoder, and structured around what the codebook formats share:
//!
//!   a 256-weight super-block is 8 blocks of 32, each 4 groups of 8;
//!   a group is ONE codebook entry (8 bytes, one weight each), ONE 8-bit sign
//!   field and ONE scale (shared by 8, 16 or 32 weights).
//!
//! Every codebook format is therefore a small extractor that yields that
//! triple per group ([`Group`]), and a single loop widens it. IQ1 differs only
//! in that its grid bytes are signed and carry a +-1/8 offset instead of a
//! sign field. Q2_K / Q3_K (2- and 3-bit fields, no codebook) and IQ4_NL
//! (nibble codebook, 32-weight blocks) get a per-weight walk of their own.
//!
//! Numerics: every factor is a few bits wide (an f16 `d`, a <= 6-bit scale, a
//! <= 8-bit grid value), so each product is exact in f32 and the pack's
//! `pd_iq_dequant_super` - repack, window unpack, widen - agrees with this
//! module bit for bit. The gate in tests/gpu_kquant_parity.rs relies on that.

use super::DequantError;
use super::iq_grids::*;

/// GGUF raw type ids of the family this module serves.
pub const IQ2_XXS: u32 = 16;
pub const IQ2_XS: u32 = 17;
pub const IQ3_XXS: u32 = 18;
pub const IQ1_S: u32 = 19;
pub const IQ4_NL: u32 = 20;
pub const Q2_K: u32 = 10;
pub const Q3_K: u32 = 11;
pub const IQ3_S: u32 = 21;
pub const IQ2_S: u32 = 22;
pub const IQ1_M: u32 = 29;

/// IQ1's per-group offset: every weight is `scale * (grid +- 1/8)`.
const IQ1_DELTA: f32 = 0.125;

/// Raw bytes per 256-weight super-block, or None when `raw_type` is not
/// served here. IQ4_NL is 8 x 18-byte blocks.
pub fn iq_block_bytes(raw_type: u32) -> Option<usize> {
    Some(match raw_type {
        IQ2_XXS => 66,
        IQ2_XS => 74,
        IQ2_S => 82,
        IQ3_XXS => 98,
        IQ3_S => 110,
        IQ1_S => 50,
        IQ1_M => 56,
        IQ4_NL => 144,
        Q2_K => 84,
        Q3_K => 110,
        _ => return None,
    })
}

fn f16(b: &[u8]) -> f32 {
    half::f16::from_le_bytes([b[0], b[1]]).to_f32()
}

fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Nibble `hi` (upper) or `!hi` (lower) of a byte, as f32.
fn nibble(b: u8, hi: bool) -> f32 {
    (if hi { b >> 4 } else { b & 0xF }) as f32
}

/// The 7-bit sign index of IQ2_XXS / IQ2_XS / IQ3_XXS widened to its 8-sign
/// byte: the format stores 7 bits and fixes the eighth so the byte has even
/// parity (what ggml tabulates as `ksigns_iq2xs`).
fn signs7(idx: u32) -> u8 {
    (idx | ((idx.count_ones() & 1) << 7)) as u8
}

/// One 8-weight group of a codebook format.
struct Group {
    /// The codebook entry: byte j is weight j. Unsigned magnitudes for the
    /// IQ2 / IQ3 grids, signed `{-1, 0, 1}` for the IQ1 grid.
    grid: u64,
    /// Bit j set = weight j is negated (IQ2 / IQ3 only; IQ1 carries none).
    signs: u8,
    scale: f32,
    /// IQ1's per-group offset, `+-IQ1_DELTA`; 0 for the sign-field formats.
    delta: f32,
    signed_grid: bool,
}

impl Group {
    fn widen(&self, y: &mut [f32]) {
        for (j, out) in y.iter_mut().enumerate() {
            let byte = ((self.grid >> (8 * j)) & 0xFF) as u8;
            *out = if self.signed_grid {
                self.scale * (byte as i8 as f32 + self.delta)
            } else {
                let mag = byte as f32;
                self.scale
                    * if (self.signs >> j) & 1 == 1 {
                        -mag
                    } else {
                        mag
                    }
            };
        }
    }
}

/// IQ2_XXS block layout: d f16 | per 32-weight block: 4 grid-index bytes,
/// then a u32 of 4 x 7-bit sign indices with the block scale in its top 4 bits.
fn iq2_xxs(s: &[u8], ib: usize, l: usize) -> Group {
    let d = f16(s);
    let idx = u32le(&s[2 + 8 * ib..]);
    let sig = u32le(&s[2 + 8 * ib + 4..]);
    Group {
        grid: IQ2XXS_GRID[((idx >> (8 * l)) & 0xFF) as usize],
        signs: signs7((sig >> (7 * l)) & 127),
        scale: d * (0.5 + (sig >> 28) as f32) * 0.25,
        delta: 0.0,
        signed_grid: false,
    }
}

/// IQ2_XS: d f16 | 32 x u16 (9-bit grid index, 7-bit sign index) | 8 scale
/// bytes, a nibble per 16 weights.
fn iq2_xs(s: &[u8], ib: usize, l: usize) -> Group {
    let d = f16(s);
    let q = u16le(&s[2 + 8 * ib + 2 * l..]);
    Group {
        grid: IQ2XS_GRID[(q & 511) as usize],
        signs: signs7((q >> 9) as u32),
        scale: d * (0.5 + nibble(s[66 + ib], l >= 2)) * 0.25,
        delta: 0.0,
        signed_grid: false,
    }
}

/// IQ2_S: d f16 | 32 grid-index bytes | 32 sign bytes | 8 bytes of index
/// high bits (2 per group) | 8 scale bytes, a nibble per 16 weights.
fn iq2_s(s: &[u8], ib: usize, l: usize) -> Group {
    let d = f16(s);
    let hi = ((s[66 + ib] as usize) >> (2 * l)) & 3;
    Group {
        grid: IQ2S_GRID[s[2 + 4 * ib + l] as usize | (hi << 8)],
        signs: s[34 + 4 * ib + l],
        scale: d * (0.5 + nibble(s[74 + ib], l >= 2)) * 0.25,
        delta: 0.0,
        signed_grid: false,
    }
}

/// IQ3_XXS: d f16 | 64 grid-index bytes (two 4-weight entries per group) |
/// per block a u32 of 4 x 7-bit sign indices with the scale in its top 4 bits.
fn iq3_xxs(s: &[u8], ib: usize, l: usize) -> Group {
    let d = f16(s);
    let sig = u32le(&s[66 + 4 * ib..]);
    let lo = IQ3XXS_GRID[s[2 + 8 * ib + 2 * l] as usize] as u64;
    let hi = IQ3XXS_GRID[s[2 + 8 * ib + 2 * l + 1] as usize] as u64;
    Group {
        grid: lo | (hi << 32),
        signs: signs7((sig >> (7 * l)) & 127),
        scale: d * (0.5 + (sig >> 28) as f32) * 0.5,
        delta: 0.0,
        signed_grid: false,
    }
}

/// IQ3_S: d f16 | 64 grid-index bytes | 8 bytes of index high bits (1 per
/// 4-weight entry) | 32 sign bytes | 4 scale bytes, a nibble per block.
fn iq3_s(s: &[u8], ib: usize, l: usize) -> Group {
    let d = f16(s);
    let qh = s[66 + ib] as usize;
    let lo = IQ3S_GRID[s[2 + 8 * ib + 2 * l] as usize | (((qh >> (2 * l)) & 1) << 8)] as u64;
    let hi =
        IQ3S_GRID[s[2 + 8 * ib + 2 * l + 1] as usize | (((qh >> (2 * l + 1)) & 1) << 8)] as u64;
    Group {
        grid: lo | (hi << 32),
        signs: s[74 + 4 * ib + l],
        scale: d * (1.0 + 2.0 * nibble(s[106 + (ib >> 1)], ib & 1 == 1)),
        delta: 0.0,
        signed_grid: false,
    }
}

/// IQ1_S: d f16 | 32 grid-index bytes | per block a u16: 4 x 3 index high
/// bits, a 3-bit scale, and the block's offset sign in bit 15.
fn iq1_s(s: &[u8], ib: usize, l: usize) -> Group {
    let d = f16(s);
    let h = u16le(&s[34 + 2 * ib..]) as usize;
    Group {
        grid: IQ1S_GRID[s[2 + 4 * ib + l] as usize | (((h >> (3 * l)) & 7) << 8)],
        signs: 0,
        scale: d * (2 * ((h >> 12) & 7) + 1) as f32,
        delta: if h & 0x8000 != 0 {
            -IQ1_DELTA
        } else {
            IQ1_DELTA
        },
        signed_grid: true,
    }
}

/// IQ1_M: 32 grid-index bytes | 16 bytes of high bits, one per two groups
/// (3 index bits + an offset sign per nibble) | 4 x u16 scales, each holding
/// two 3-bit scales per block pair plus a nibble of the shared `d`.
fn iq1_m(s: &[u8], ib: usize, l: usize) -> Group {
    let sc = &s[48..56];
    let d = {
        let s16 = |i: usize| u16le(&sc[2 * i..]);
        // d's 16 bits are the top nibble of each scale word, low word first
        let bits = (0..4).fold(0u16, |acc, i| acc | ((s16(i) >> 12) << (4 * i)));
        half::f16::from_bits(bits).to_f32()
    };
    let hq = s[32 + 2 * ib + (l >> 1)] as usize;
    let nib = if l & 1 == 1 { hq >> 4 } else { hq & 0xF };
    let sw = u16le(&sc[2 * (ib >> 1)..]) as usize;
    let shift = 6 * (ib & 1) + 3 * (l >> 1);
    Group {
        grid: IQ1S_GRID[s[4 * ib + l] as usize | ((nib & 7) << 8)],
        signs: 0,
        scale: d * (2 * ((sw >> shift) & 7) + 1) as f32,
        delta: if nib & 8 != 0 { -IQ1_DELTA } else { IQ1_DELTA },
        signed_grid: true,
    }
}

/// One 256-weight super-block of `raw_type` -> 256 f32.
pub fn dequant_iq_super(raw_type: u32, s: &[u8], y: &mut [f32]) {
    let group: Option<fn(&[u8], usize, usize) -> Group> = match raw_type {
        IQ2_XXS => Some(iq2_xxs),
        IQ2_XS => Some(iq2_xs),
        IQ2_S => Some(iq2_s),
        IQ3_XXS => Some(iq3_xxs),
        IQ3_S => Some(iq3_s),
        IQ1_S => Some(iq1_s),
        IQ1_M => Some(iq1_m),
        _ => None,
    };
    if let Some(group) = group {
        for ib in 0..8 {
            for l in 0..4 {
                let at = 32 * ib + 8 * l;
                group(s, ib, l).widen(&mut y[at..at + 8]);
            }
        }
        return;
    }
    match raw_type {
        Q2_K => {
            // scales[16] (a nibble each of scale | min per 16 weights) | qs[64]
            // (2-bit fields: byte 32*(i/128) + i%32, bits 2*((i/32)%4)) | d | dmin
            let (d, dmin) = (f16(&s[80..]), f16(&s[82..]));
            for (i, out) in y.iter_mut().enumerate() {
                let sc = s[i / 16];
                let q = (s[16 + 32 * (i / 128) + i % 32] >> (2 * ((i / 32) % 4))) & 3;
                *out = d * (sc & 0xF) as f32 * q as f32 - dmin * (sc >> 4) as f32;
            }
        }
        Q3_K => {
            // hmask[32] (bit i/32 of byte i%32 is the weight's third bit) |
            // qs[64] (low two bits, laid out as Q2_K's) | 12 packed 6-bit
            // scales | d. Values are the 3-bit field minus 4.
            let d = f16(&s[108..]);
            let sc6 = |i: usize| -> i32 {
                let lo = (s[96 + (i & 7)] >> (4 * (i >> 3))) & 0xF;
                let hi = (s[104 + (i & 3)] >> (2 * (i >> 2))) & 3;
                (lo | (hi << 4)) as i32 - 32
            };
            for (i, out) in y.iter_mut().enumerate() {
                let lo2 = ((s[32 + 32 * (i / 128) + i % 32] >> (2 * ((i / 32) % 4))) & 3) as i32;
                let hb = ((s[i % 32] >> (i / 32)) & 1) as i32;
                // `+ 0.0` mirrors the pack's widen (`f * s8 + g`, g = 0 here): a
                // zero field under a negative scale is -0.0 as a bare product
                // and +0.0 after the add, and the gate compares bits. Q3_K is
                // the one format in the family whose fields can be 0.
                *out = d * sc6(i / 16) as f32 * (lo2 + 4 * hb - 4) as f32 + 0.0;
            }
        }
        _ => {
            // IQ4_NL: 8 blocks of {d f16, 16 nibble bytes}; low nibbles are
            // the block's first 16 weights, high nibbles the last 16, each
            // through the 16-entry codebook
            for (j, blk) in s.as_chunks::<18>().0.iter().enumerate() {
                let d = f16(blk);
                for (l, &b) in blk[2..18].iter().enumerate() {
                    y[32 * j + l] = d * KVALUES_IQ4NL[(b & 0xF) as usize] as f32;
                    y[32 * j + 16 + l] = d * KVALUES_IQ4NL[(b >> 4) as usize] as f32;
                }
            }
        }
    }
}

/// A whole tensor of `raw_type` -> f32.
pub fn dequant_iq(raw_type: u32, data: &[u8], out: &mut [f32]) -> Result<(), DequantError> {
    let Some(block) = iq_block_bytes(raw_type) else {
        return Err(DequantError::BadInputLen {
            kind: "iq",
            input: data.len(),
            block: 0,
        });
    };
    if !data.len().is_multiple_of(block) {
        return Err(DequantError::BadInputLen {
            kind: "iq",
            input: data.len(),
            block,
        });
    }
    let expected = data.len() / block * 256;
    if out.len() != expected {
        return Err(DequantError::BadOutputLen {
            out: out.len(),
            expected,
        });
    }
    for (i, s) in data.chunks_exact(block).enumerate() {
        dequant_iq_super(raw_type, s, &mut out[i * 256..(i + 1) * 256]);
    }
    Ok(())
}
