//! Polyphase windowed-sinc sample-rate conversion for the transcription
//! endpoints (arbitrary input rate -> the model's 16 kHz). In-house on
//! purpose (parity discipline: every sample-level transform is part of the
//! numeric contract). Standard construction: reduce the ratio to p/q, build
//! a Kaiser-windowed sinc low-pass at the narrower Nyquist with p polyphase
//! branches, then each output sample is one dot product. f64 accumulation.
//!
//! Quality target is the soxr/librosa class (Kaiser beta 14 ≈ 110 dB
//! stopband, 32 zero-crossings per side scaled by the decimation factor) -
//! well past 16-bit source material. Every common source rate (8/11.025/
//! 16/22.05/24/32/44.1/48 kHz) reduces to p ≤ 441.

/// Modified Bessel function of the first kind, order 0 (power series; the
/// Kaiser window's normalizer). Converges in ~25 terms for beta 14.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let half = x / 2.0;
    for k in 1..40 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < sum * 1e-18 {
            break;
        }
    }
    sum
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Resample mono f32 samples from `from_hz` to `to_hz`. Identity when the
/// rates match. Errors on degenerate rates.
pub fn resample(samples: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>, String> {
    if from_hz == 0 || to_hz == 0 {
        return Err("zero sample rate".into());
    }
    if from_hz == to_hz || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    let g = gcd(to_hz as usize, from_hz as usize);
    let (p, q) = (to_hz as usize / g, from_hz as usize / g); // out = in * p / q
    if p > 4096 {
        return Err(format!(
            "unsupported rate pair {from_hz}->{to_hz} (phase count {p})"
        ));
    }

    // Anti-alias cutoff at the narrower Nyquist, pulled in 8% for the
    // transition band; taps scale with the decimation factor so the filter
    // stays as sharp in the downsampled domain.
    const LOBES: usize = 32;
    const BETA: f64 = 14.0;
    let stretch = (q as f64 / p as f64).max(1.0);
    let half_taps = (LOBES as f64 * stretch).ceil() as isize;
    let fc = 0.92 * 0.5 * (p as f64 / q as f64).min(1.0);

    // Polyphase bank: output n reads input around n*q/p; phase = (n*q) % p,
    // base = (n*q) / p. Branch coefficients h[phase][j] = kaiser-sinc at
    // (j - phase/p) taps offset, normalized per branch for exact DC gain.
    let i0b = bessel_i0(BETA);
    let mut bank = vec![0f64; p * (2 * half_taps as usize + 1)];
    let width = 2 * half_taps as usize + 1;
    for phase in 0..p {
        let frac = phase as f64 / p as f64;
        let row = &mut bank[phase * width..(phase + 1) * width];
        let mut sum = 0.0f64;
        for (j, w) in row.iter_mut().enumerate() {
            let t = (j as isize - half_taps) as f64 - frac;
            let x = t / half_taps as f64;
            if x.abs() <= 1.0 {
                let window = bessel_i0(BETA * (1.0 - x * x).sqrt()) / i0b;
                let s = if t == 0.0 {
                    2.0 * fc
                } else {
                    (2.0 * std::f64::consts::PI * fc * t).sin() / (std::f64::consts::PI * t)
                };
                *w = s * window;
                sum += *w;
            }
        }
        for w in row.iter_mut() {
            *w /= sum;
        }
    }

    let n_out = samples.len() * p / q;
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let base = (n * q) / p;
        let phase = (n * q) % p;
        let row = &bank[phase * width..(phase + 1) * width];
        let mut acc = 0.0f64;
        for (j, &w) in row.iter().enumerate() {
            let idx = base as isize + (j as isize - half_taps);
            if idx >= 0 && (idx as usize) < samples.len() {
                acc += samples[idx as usize] as f64 * w;
            }
        }
        out.push(acc as f32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(hz: f64, rate: f64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * hz * i as f64 / rate).sin() as f32)
            .collect()
    }

    #[test]
    fn identity_at_same_rate() {
        let x = tone(440.0, 16000.0, 1600);
        assert_eq!(resample(&x, 16000, 16000).unwrap(), x);
    }

    #[test]
    fn downsample_48k_preserves_a_band_limited_tone() {
        // 1 kHz tone at 48 kHz -> 16 kHz: interior must match the directly
        // synthesized 16 kHz tone to ~-60 dB (Kaiser-14 is far better; edge
        // taps see zeros so exclude the boundary)
        let x = tone(1000.0, 48000.0, 48000);
        let y = resample(&x, 48000, 16000).unwrap();
        assert_eq!(y.len(), 16000);
        let want = tone(1000.0, 16000.0, 16000);
        let mut max_d = 0f32;
        for i in 200..15800 {
            max_d = max_d.max((y[i] - want[i]).abs());
        }
        assert!(max_d < 1e-3, "max deviation {max_d}");
    }

    #[test]
    fn downsample_44100_rejects_above_nyquist() {
        // 10 kHz tone at 44.1 kHz is above the 16 kHz lane's 8 kHz Nyquist -
        // it must be attenuated to near-zero, not aliased into band
        let x = tone(10000.0, 44100.0, 44100);
        let y = resample(&x, 44100, 16000).unwrap();
        let peak = y[400..y.len() - 400]
            .iter()
            .fold(0f32, |m, &v| m.max(v.abs()));
        assert!(peak < 5e-3, "alias peak {peak}");
    }

    #[test]
    fn upsample_8k_doubles_length() {
        let x = tone(500.0, 8000.0, 8000);
        let y = resample(&x, 8000, 16000).unwrap();
        assert_eq!(y.len(), 16000);
        let want = tone(500.0, 16000.0, 16000);
        let mut max_d = 0f32;
        for i in 200..15800 {
            max_d = max_d.max((y[i] - want[i]).abs());
        }
        assert!(max_d < 1e-3, "max deviation {max_d}");
    }
}
