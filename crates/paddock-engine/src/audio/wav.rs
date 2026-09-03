//! Minimal RIFF/WAVE decoder for the transcription endpoints: PCM 16/24/32,
//! IEEE float32, any channel count (downmixed to mono by averaging), any
//! sample rate (the caller resamples to the model rate). In-house on
//! purpose - the parity requirement makes every sample-level transform part
//! of the numeric contract, so no external codec dependency.

pub struct WavAudio {
    /// Mono samples in [-1, 1].
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

fn rd_u16(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(o..o + 2)?.try_into().ok()?))
}

fn rd_u32(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?))
}

/// Decode a WAV file image. Errors are user-facing (they surface as HTTP
/// 400s on the transcription endpoint).
pub fn decode_wav(bytes: &[u8]) -> Result<WavAudio, String> {
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut off = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // (format, channels, rate, bits)
    let mut data: Option<&[u8]> = None;
    while off + 8 <= bytes.len() {
        let id = &bytes[off..off + 4];
        let sz = rd_u32(bytes, off + 4).ok_or("truncated chunk header")? as usize;
        let body = bytes
            .get(off + 8..off + 8 + sz)
            .ok_or("truncated chunk body")?;
        match id {
            b"fmt " => {
                if sz < 16 {
                    return Err("fmt chunk too short".into());
                }
                let mut format = rd_u16(body, 0).unwrap();
                let channels = rd_u16(body, 2).unwrap();
                let rate = rd_u32(body, 4).unwrap();
                let bits = rd_u16(body, 14).unwrap();
                // WAVE_FORMAT_EXTENSIBLE: real format lives in the GUID
                if format == 0xFFFE && sz >= 40 {
                    format = rd_u16(body, 24).unwrap();
                }
                fmt = Some((format, channels, rate, bits));
            }
            b"data" => data = Some(body),
            _ => {}
        }
        // chunks are word-aligned
        off += 8 + sz + (sz & 1);
    }
    let (format, channels, rate, bits) = fmt.ok_or("missing fmt chunk")?;
    let data = data.ok_or("missing data chunk")?;
    if channels == 0 || rate == 0 {
        return Err("invalid fmt: zero channels or sample rate".into());
    }
    let ch = channels as usize;
    let decode: fn(&[u8]) -> f32 = match (format, bits) {
        (1, 16) => |b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0,
        (1, 24) => |b| {
            let v = ((b[2] as i32) << 24 | (b[1] as i32) << 16 | (b[0] as i32) << 8) >> 8;
            v as f32 / 8388608.0
        },
        (1, 32) => |b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2147483648.0,
        (3, 32) => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        (1, 8) => |b| (b[0] as f32 - 128.0) / 128.0,
        _ => {
            return Err(format!(
                "unsupported WAV encoding: format {format}, {bits}-bit"
            ));
        }
    };
    let bytes_per = (bits as usize) / 8;
    let stride = bytes_per * ch;
    let n = data.len() / stride;
    let mut samples = Vec::with_capacity(n);
    for i in 0..n {
        let mut acc = 0.0f32;
        for c in 0..ch {
            acc += decode(&data[i * stride + c * bytes_per..]);
        }
        samples.push(acc / ch as f32);
    }
    Ok(WavAudio {
        samples,
        sample_rate: rate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_s16(rate: u32, pcm: &[i16]) -> Vec<u8> {
        let data_len = pcm.len() * 2;
        let mut v = Vec::new();
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
        v.extend_from_slice(b"WAVEfmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes()); // PCM
        v.extend_from_slice(&1u16.to_le_bytes()); // mono
        v.extend_from_slice(&rate.to_le_bytes());
        v.extend_from_slice(&(rate * 2).to_le_bytes());
        v.extend_from_slice(&2u16.to_le_bytes());
        v.extend_from_slice(&16u16.to_le_bytes());
        v.extend_from_slice(b"data");
        v.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in pcm {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    #[test]
    fn parses_pcm16_mono() {
        let w = decode_wav(&wav_s16(16000, &[0, 16384, -16384, 32767])).unwrap();
        assert_eq!(w.sample_rate, 16000);
        assert_eq!(w.samples.len(), 4);
        assert!((w.samples[1] - 0.5).abs() < 1e-6);
        assert!((w.samples[2] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn rejects_non_wav() {
        assert!(decode_wav(b"OggS\0\0\0\0\0\0\0\0").is_err());
    }
}
