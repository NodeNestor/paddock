//! CPU reference dequant for the ggml i-quant family (IQ1_S, IQ1_M, IQ2_XXS,
//! IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) and IQ4_NL - a line-for-line port of
//! ggml-quants.c's `dequantize_row_*`, in the same f32 operation order the
//! pack's `pd_iq_dequant_super` (quant/iquant.cuh) uses, so the GPU gate is a
//! bit-identity check. Codebooks in `iq_grids.rs`.

use super::DequantError;
use super::iq_grids::*;

/// GGUF raw type ids of the family this module serves.
pub const IQ2_XXS: u32 = 16;
pub const IQ2_XS: u32 = 17;
pub const IQ3_XXS: u32 = 18;
pub const IQ1_S: u32 = 19;
pub const IQ4_NL: u32 = 20;
pub const IQ3_S: u32 = 21;
pub const IQ2_S: u32 = 22;
pub const IQ1_M: u32 = 29;

const IQ1S_DELTA: f32 = 0.125;

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

fn sgn(signs: u32, j: u32) -> f32 {
    if (signs >> j) & 1 == 1 { -1.0 } else { 1.0 }
}

fn gb(grid: u64, j: u32) -> f32 {
    ((grid >> (8 * j)) & 0xFF) as f32
}

fn gb32(grid: u32, j: u32) -> f32 {
    ((grid >> (8 * j)) & 0xFF) as f32
}

fn iq1m_d(scales: &[u8]) -> f32 {
    let s = |i: usize| u16le(&scales[2 * i..]);
    let u = (s(0) >> 12) | ((s(1) >> 8) & 0x00f0) | ((s(2) >> 4) & 0x0f00) | (s(3) & 0xf000);
    half::f16::from_bits(u).to_f32()
}

/// One 256-weight super-block of `raw_type` -> 256 f32.
pub fn dequant_iq_super(raw_type: u32, s: &[u8], y: &mut [f32]) {
    match raw_type {
        IQ2_XXS => {
            let d = f16(s);
            let qs = &s[2..];
            for ib32 in 0..8usize {
                let a0 = u32le(&qs[8 * ib32..]);
                let a1 = u32le(&qs[8 * ib32 + 4..]);
                let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                for l in 0..4u32 {
                    let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 0xFF) as usize];
                    let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize] as u32;
                    for j in 0..8u32 {
                        y[ib32 * 32 + l as usize * 8 + j as usize] =
                            db * gb(grid, j) * sgn(signs, j);
                    }
                }
            }
        }
        IQ2_XS => {
            let d = f16(s);
            let qs = &s[2..];
            let scales = &s[66..];
            for ib32 in 0..8usize {
                let db0 = d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25;
                let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
                for l in 0..4usize {
                    let q = u16le(&qs[8 * ib32 + 2 * l..]);
                    let grid = IQ2XS_GRID[(q & 511) as usize];
                    let signs = KSIGNS_IQ2XS[(q >> 9) as usize] as u32;
                    let dl = if l < 2 { db0 } else { db1 };
                    for j in 0..8u32 {
                        y[ib32 * 32 + l * 8 + j as usize] = dl * gb(grid, j) * sgn(signs, j);
                    }
                }
            }
        }
        IQ2_S => {
            let d = f16(s);
            // qs[64] = 32 grid-index bytes then 32 sign bytes; qh follows qs
            let qs = &s[2..];
            let signs = &s[34..];
            let qh = &s[66..];
            let scales = &s[74..];
            for ib32 in 0..8usize {
                let db0 = d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25;
                let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
                for l in 0..4usize {
                    let idx =
                        qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                    let grid = IQ2S_GRID[idx];
                    let sg = signs[4 * ib32 + l] as u32;
                    let dl = if l < 2 { db0 } else { db1 };
                    for j in 0..8u32 {
                        y[ib32 * 32 + l * 8 + j as usize] = dl * gb(grid, j) * sgn(sg, j);
                    }
                }
            }
        }
        IQ3_XXS => {
            let d = f16(s);
            let qs = &s[2..];
            let sas = &qs[64..];
            for ib32 in 0..8usize {
                let aux = u32le(&sas[4 * ib32..]);
                let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                for l in 0..4usize {
                    let signs = KSIGNS_IQ2XS[((aux >> (7 * l as u32)) & 127) as usize] as u32;
                    let g1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize];
                    let g2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize];
                    for j in 0..4u32 {
                        y[ib32 * 32 + l * 8 + j as usize] = db * gb32(g1, j) * sgn(signs, j);
                        y[ib32 * 32 + l * 8 + 4 + j as usize] =
                            db * gb32(g2, j) * sgn(signs, j + 4);
                    }
                }
            }
        }
        IQ3_S => {
            let d = f16(s);
            let qs = &s[2..];
            let qh = &s[66..];
            let signs = &s[74..];
            let scales = &s[106..];
            for ib32 in 0..8usize {
                let sc = scales[ib32 >> 1];
                let nib = if ib32 & 1 == 1 { sc >> 4 } else { sc & 0xF };
                let db = d * (1.0 + 2.0 * nib as f32);
                for l in 0..4usize {
                    let i1 = qs[8 * ib32 + 2 * l] as usize
                        | (((qh[ib32] as usize) << (8 - 2 * l)) & 256);
                    let i2 = qs[8 * ib32 + 2 * l + 1] as usize
                        | (((qh[ib32] as usize) << (7 - 2 * l)) & 256);
                    let (g1, g2) = (IQ3S_GRID[i1], IQ3S_GRID[i2]);
                    let sg = signs[4 * ib32 + l] as u32;
                    for j in 0..4u32 {
                        y[ib32 * 32 + l * 8 + j as usize] = db * gb32(g1, j) * sgn(sg, j);
                        y[ib32 * 32 + l * 8 + 4 + j as usize] = db * gb32(g2, j) * sgn(sg, j + 4);
                    }
                }
            }
        }
        IQ1_S => {
            let d = f16(s);
            let qs = &s[2..];
            let qh = &s[34..];
            for ib in 0..8usize {
                let h = u16le(&qh[2 * ib..]);
                let dl = d * (2 * ((h >> 12) & 7) + 1) as f32;
                let delta = if h & 0x8000 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                };
                for l in 0..4usize {
                    let idx = qs[4 * ib + l] as usize | ((((h >> (3 * l)) & 7) as usize) << 8);
                    let grid = IQ1S_GRID[idx];
                    for j in 0..8u32 {
                        let g = ((grid >> (8 * j)) & 0xFF) as u8 as i8 as f32;
                        y[ib * 32 + l * 8 + j as usize] = dl * (g + delta);
                    }
                }
            }
        }
        IQ1_M => {
            let qs = &s[0..];
            let qh = &s[32..];
            let scb = &s[48..];
            let d = iq1m_d(scb);
            for ib in 0..8usize {
                let sc = u16le(&scb[2 * (ib >> 1)..]);
                let sh = 6 * (ib & 1) as u16;
                let dl1 = d * (2 * ((sc >> sh) & 7) + 1) as f32;
                let dl2 = d * (2 * ((sc >> (sh + 3)) & 7) + 1) as f32;
                let (h0, h1) = (qh[2 * ib] as usize, qh[2 * ib + 1] as usize);
                let idx = [
                    qs[4 * ib] as usize | ((h0 << 8) & 0x700),
                    qs[4 * ib + 1] as usize | ((h0 << 4) & 0x700),
                    qs[4 * ib + 2] as usize | ((h1 << 8) & 0x700),
                    qs[4 * ib + 3] as usize | ((h1 << 4) & 0x700),
                ];
                let dd = |bit: bool| if bit { -IQ1S_DELTA } else { IQ1S_DELTA };
                let delta = [
                    dd(h0 & 0x08 != 0),
                    dd(h0 & 0x80 != 0),
                    dd(h1 & 0x08 != 0),
                    dd(h1 & 0x80 != 0),
                ];
                for l in 0..4usize {
                    let grid = IQ1S_GRID[idx[l]];
                    let dl = if l < 2 { dl1 } else { dl2 };
                    for j in 0..8u32 {
                        let g = ((grid >> (8 * j)) & 0xFF) as u8 as i8 as f32;
                        y[ib * 32 + l * 8 + j as usize] = dl * (g + delta[l]);
                    }
                }
            }
        }
        _ => {
            // IQ4_NL: 8 blocks of {f16 d, 16 nibble bytes}
            for j in 0..8usize {
                let blk = &s[j * 18..];
                let d = f16(blk);
                for l in 0..16usize {
                    y[j * 32 + l] = d * KVALUES_IQ4NL[(blk[2 + l] & 0xF) as usize] as f32;
                    y[j * 32 + 16 + l] = d * KVALUES_IQ4NL[(blk[2 + l] >> 4) as usize] as f32;
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
