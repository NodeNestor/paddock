//! CPU reference implementations of quantized-type decodes.
//!
//! These are permanent parity baselines, not temporary shims: every GPU kernel
//! is CI-diffed against them forever. Ported bit-exactly from the format's
//! de-facto reference (ggml) - the "HALF" trick and the
//! doubled FP4 value table are the spec as written into millions of GGUF files,
//! so we reproduce them exactly rather than the textbook OCP formulas.

pub mod delta_net;
pub mod dflash;
pub mod ops;
pub mod qwen35_attn;
pub mod qwen4exp;

/// FP4 (E2M1) values as stored in MXFP4 nibbles - DOUBLED relative to the OCP
/// E2M1 table (0, .5, 1, 1.5, 2, 3, 4, 6), compensated by the halved E8M0
/// scale below. Index = nibble; upper half is the sign bit.
const FP4_VALUES: [f32; 16] = [
    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
];

/// E8M0 exponent byte -> 2^(e-127) * 0.5, bit-exact with ggml_e8m0_to_fp32_half:
/// e < 2 hits f32 denormal territory and is built from raw bit patterns.
pub fn e8m0_half_to_f32(e: u8) -> f32 {
    let bits: u32 = if e < 2 {
        // 0x0020_0000 = 2^-128, doubled once per step below e=2
        0x0020_0000 << e
    } else {
        // 0.5 * 2^(e-127) = 2^(e-128) -> normalized f32 with biased exponent e-1
        u32::from(e - 1) << 23
    };
    f32::from_bits(bits)
}

pub const MXFP4_BLOCK_ELEMS: usize = 32;
pub const MXFP4_BLOCK_BYTES: usize = 17;

#[derive(Debug, thiserror::Error)]
pub enum DequantError {
    #[error("input length {input} is not a whole number of {kind} blocks ({block} bytes)")]
    BadInputLen {
        kind: &'static str,
        input: usize,
        block: usize,
    },
    #[error("output length {out} != expected {expected} elements")]
    BadOutputLen { out: usize, expected: usize },
}

/// MXFP4 -> f32. Layout per block: 1 byte E8M0 scale + 16 bytes packed nibbles;
/// low nibble of qs[j] is element j, high nibble is element j+16.
pub fn dequant_mxfp4(data: &[u8], out: &mut [f32]) -> Result<(), DequantError> {
    if !data.len().is_multiple_of(MXFP4_BLOCK_BYTES) {
        return Err(DequantError::BadInputLen {
            kind: "mxfp4",
            input: data.len(),
            block: MXFP4_BLOCK_BYTES,
        });
    }
    let blocks = data.len() / MXFP4_BLOCK_BYTES;
    let expected = blocks * MXFP4_BLOCK_ELEMS;
    if out.len() != expected {
        return Err(DequantError::BadOutputLen {
            out: out.len(),
            expected,
        });
    }

    for (i, block) in data.as_chunks::<MXFP4_BLOCK_BYTES>().0.iter().enumerate() {
        let d = e8m0_half_to_f32(block[0]);
        let qs = &block[1..];
        let y = &mut out[i * MXFP4_BLOCK_ELEMS..(i + 1) * MXFP4_BLOCK_ELEMS];
        for (j, &q) in qs.iter().enumerate() {
            y[j] = FP4_VALUES[(q & 0x0F) as usize] * d;
            y[j + 16] = FP4_VALUES[(q >> 4) as usize] * d;
        }
    }
    Ok(())
}

pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Q8_0 -> f32. Layout per block: f16 scale + 32 signed bytes.
pub fn dequant_q8_0(data: &[u8], out: &mut [f32]) -> Result<(), DequantError> {
    if !data.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
        return Err(DequantError::BadInputLen {
            kind: "q8_0",
            input: data.len(),
            block: Q8_0_BLOCK_BYTES,
        });
    }
    let blocks = data.len() / Q8_0_BLOCK_BYTES;
    let expected = blocks * Q8_0_BLOCK_ELEMS;
    if out.len() != expected {
        return Err(DequantError::BadOutputLen {
            out: out.len(),
            expected,
        });
    }

    for (i, block) in data.as_chunks::<Q8_0_BLOCK_BYTES>().0.iter().enumerate() {
        let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let y = &mut out[i * Q8_0_BLOCK_ELEMS..(i + 1) * Q8_0_BLOCK_ELEMS];
        for (j, &q) in block[2..].iter().enumerate() {
            y[j] = (q as i8) as f32 * d;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_half_matches_known_points() {
        assert_eq!(e8m0_half_to_f32(128), 1.0); // 2^1 * 0.5
        assert_eq!(e8m0_half_to_f32(127), 0.5); // 2^0 * 0.5
        assert_eq!(e8m0_half_to_f32(129), 2.0);
        assert_eq!(e8m0_half_to_f32(0), f32::from_bits(0x0020_0000)); // 2^-128
        assert_eq!(e8m0_half_to_f32(1), f32::from_bits(0x0040_0000)); // 2^-127
    }

    #[test]
    fn mxfp4_dequant_hand_computed_block() {
        // scale byte 128 -> d = 1.0; qs[0] = 0x51: low nibble 1 -> elem0 = 1.0,
        // high nibble 5 -> elem16 = 6.0; qs[1] = 0x9F: low F -> elem1 = -12.0,
        // high 9 -> elem17 = -1.0
        let mut block = [0u8; 17];
        block[0] = 128;
        block[1] = 0x51;
        block[2] = 0x9F;
        let mut out = [0f32; 32];
        dequant_mxfp4(&block, &mut out).expect("dequants");
        assert_eq!(out[0], 1.0);
        assert_eq!(out[16], 6.0);
        assert_eq!(out[1], -12.0);
        assert_eq!(out[17], -1.0);
        assert_eq!(out[2], 0.0);
    }

    #[test]
    fn mxfp4_scale_applies() {
        let mut block = [0u8; 17];
        block[0] = 129; // d = 2.0
        block[1] = 0x07; // low nibble 7 -> 12.0 * 2.0
        let mut out = [0f32; 32];
        dequant_mxfp4(&block, &mut out).expect("dequants");
        assert_eq!(out[0], 24.0);
    }

    #[test]
    fn q8_0_dequant_hand_computed_block() {
        let mut block = [0u8; 34];
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        block[2] = 100; // elem0 = 100 * 0.5
        block[3] = (-100i8) as u8; // elem1 = -50
        let mut out = [0f32; 32];
        dequant_q8_0(&block, &mut out).expect("dequants");
        assert_eq!(out[0], 50.0);
        assert_eq!(out[1], -50.0);
    }

    #[test]
    fn length_mismatches_are_errors() {
        let mut out = [0f32; 32];
        assert!(dequant_mxfp4(&[0u8; 16], &mut out).is_err()); // not a block multiple
        assert!(dequant_mxfp4(&[0u8; 17], &mut out[..31]).is_err()); // wrong out len
    }
}
