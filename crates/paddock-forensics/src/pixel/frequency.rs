//! Frequency-domain AI-generation analysis (2D FFT), ported verbatim from
//! the CPU reference. CPU-only: uses `rustfft` - a GPU cuFFT would not
//! reproduce it bit-for-bit, so `gpu()` delegates to `cpu()` (same rationale as
//! cfa; the reference's GPU FFT path is a separate approximation, not the canonical).
//!
//! Signals: radial power spectrum 1/f^β falloff (natural β≈2), Wiener-entropy
//! spectral flatness, azimuthal GAN transposed-conv peaks, mid/high-frequency
//! energy ratios (diffusion / AI-upscaling signatures). Photo-oriented ->
//! skipped for PDFs.

use num_complex::Complex;
use rustfft::FftPlanner;

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct FrequencyAnalyzer;

struct SpectrumStats {
    spectral_exponent: f64,
    spectral_flatness: f64,
    high_freq_ratio: f64,
    mid_freq_ratio: f64,
    periodic_peaks_detected: bool,
    azimuthal_peak_ratio: f64,
}

impl Analyzer for FrequencyAnalyzer {
    fn name(&self) -> &'static str {
        "frequency"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        !ctx.is_pdf()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        if ctx.width < 64 || ctx.height < 64 {
            return vec![];
        }

        // Downsample large images before FFT; spectral stats are scale-invariant.
        const MAX_W: u32 = 800;
        const MAX_H: u32 = 600;
        let (fft_gray, fft_w, fft_h);
        if ctx.width > MAX_W || ctx.height > MAX_H {
            let scale = (MAX_W as f64 / ctx.width as f64).min(MAX_H as f64 / ctx.height as f64);
            let new_w = ((ctx.width as f64 * scale) as u32).max(64);
            let new_h = ((ctx.height as f64 * scale) as u32).max(64);
            let luma = ctx.image.to_luma8();
            let resized =
                image::imageops::resize(&luma, new_w, new_h, image::imageops::FilterType::Triangle);
            fft_gray = resized.into_raw();
            fft_w = new_w;
            fft_h = new_h;
        } else {
            fft_gray = ctx.gray().to_vec();
            fft_w = ctx.width;
            fft_h = ctx.height;
        }

        let stats = Self::analyze_2d_fft(&fft_gray, fft_w, fft_h);
        let mut findings = Vec::new();

        if stats.spectral_exponent < 1.3 || stats.spectral_exponent > 3.2 {
            findings.push(Finding::new(
                "frequency",
                "spectral_exponent_anomaly",
                format!(
                    "Spectral falloff exponent β={:.2} deviates from natural range [1.5, 2.5]",
                    stats.spectral_exponent
                ),
                Severity::High,
                0.70,
            ));
        }

        if stats.spectral_flatness > 0.65 {
            findings.push(Finding::new(
                "frequency",
                "flat_spectrum",
                format!(
                    "Wiener entropy {:.3} indicates unusually flat frequency spectrum",
                    stats.spectral_flatness
                ),
                Severity::High,
                0.65,
            ));
        }

        if stats.periodic_peaks_detected && stats.azimuthal_peak_ratio > 3.0 {
            findings.push(Finding::new(
                "frequency",
                "gan_spectral_peaks",
                format!(
                    "Periodic peaks in azimuthal spectrum (peak/median ratio {:.1}x) - \
                     characteristic of GAN transposed-convolution artifacts",
                    stats.azimuthal_peak_ratio
                ),
                Severity::Critical,
                0.75,
            ));
        }

        if stats.mid_freq_ratio > 0.55 {
            findings.push(Finding::new(
                "frequency",
                "mid_freq_anomaly",
                format!(
                    "Elevated mid-frequency energy ratio ({:.3}) - \
                     may indicate diffusion model generation",
                    stats.mid_freq_ratio
                ),
                Severity::Medium,
                0.55,
            ));
        }

        if stats.high_freq_ratio < 0.03 {
            findings.push(Finding::new(
                "frequency",
                "low_high_freq_energy",
                format!(
                    "Abnormally low high-frequency energy ({:.4}) - \
                     may indicate AI upscaling or synthetic generation",
                    stats.high_freq_ratio
                ),
                Severity::Medium,
                0.55,
            ));
        }

        findings
    }

    #[cfg(feature = "cuda")]
    fn gpu(
        &self,
        _gpu: &crate::gpu::ForensicGpu,
        ctx: &Context,
    ) -> Result<Vec<Finding>, crate::gpu::GpuError> {
        Ok(self.cpu(ctx))
    }
}

impl FrequencyAnalyzer {
    fn analyze_2d_fft(pixels: &[u8], width: u32, height: u32) -> SpectrumStats {
        let w = width as usize;
        let h = height as usize;

        let mut planner = FftPlanner::<f64>::new();
        let fft_row = planner.plan_fft_forward(w);
        let fft_col = planner.plan_fft_forward(h);

        let mut data: Vec<Complex<f64>> = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let hann_x = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * x as f64 / w as f64).cos());
                let hann_y = 0.5 * (1.0 - (2.0 * std::f64::consts::PI * y as f64 / h as f64).cos());
                let val = pixels[y * w + x] as f64 * hann_x * hann_y;
                data.push(Complex::new(val, 0.0));
            }
        }

        for y in 0..h {
            let row = &mut data[y * w..(y + 1) * w];
            fft_row.process(row);
        }

        let mut col_buf = vec![Complex::new(0.0, 0.0); h];
        for x in 0..w {
            for y in 0..h {
                col_buf[y] = data[y * w + x];
            }
            fft_col.process(&mut col_buf);
            for y in 0..h {
                data[y * w + x] = col_buf[y];
            }
        }

        let cx = w / 2;
        let cy = h / 2;
        let max_radius = (cx.min(cy)) as f64;

        let num_bins = 64.min(max_radius as usize);
        let mut radial_power = vec![0.0_f64; num_bins];
        let mut radial_count = vec![0_u64; num_bins];

        let num_angle_bins = 360;
        let mut azimuthal_power = vec![0.0_f64; num_angle_bins];
        let mut azimuthal_count = vec![0_u64; num_angle_bins];

        let mut total_energy = 0.0_f64;
        let mut high_freq_energy = 0.0_f64;
        let mut mid_freq_energy = 0.0_f64;

        for y in 0..h {
            for x in 0..w {
                let sx = ((x + cx) % w) as f64 - cx as f64;
                let sy = ((y + cy) % h) as f64 - cy as f64;
                let radius = (sx * sx + sy * sy).sqrt();

                if radius < 1.0 || radius >= max_radius {
                    continue;
                }

                let mag_sq = data[y * w + x].norm_sqr();
                total_energy += mag_sq;

                let bin = ((radius / max_radius) * num_bins as f64) as usize;
                let bin = bin.min(num_bins - 1);
                radial_power[bin] += mag_sq;
                radial_count[bin] += 1;

                let freq_ratio = radius / max_radius;
                if freq_ratio > 0.75 {
                    high_freq_energy += mag_sq;
                } else if freq_ratio > 0.25 && freq_ratio <= 0.75 {
                    mid_freq_energy += mag_sq;
                }

                let angle = sy.atan2(sx);
                let angle_deg = ((angle.to_degrees() + 360.0) % 360.0) as usize;
                let angle_bin = angle_deg.min(num_angle_bins - 1);
                azimuthal_power[angle_bin] += mag_sq;
                azimuthal_count[angle_bin] += 1;
            }
        }

        let radial_avg: Vec<f64> = radial_power
            .iter()
            .zip(&radial_count)
            .map(|(&p, &c)| if c > 0 { p / c as f64 } else { 0.0 })
            .collect();

        let spectral_exponent = Self::estimate_spectral_exponent(&radial_avg, num_bins, max_radius);

        let non_zero: Vec<f64> = radial_avg.iter().copied().filter(|&v| v > 0.0).collect();
        let spectral_flatness = if non_zero.len() >= 2 {
            let n = non_zero.len() as f64;
            let arith_mean = non_zero.iter().sum::<f64>() / n;
            let log_sum: f64 = non_zero.iter().map(|&v| v.ln()).sum();
            let geo_mean = (log_sum / n).exp();
            if arith_mean > 0.0 {
                (geo_mean / arith_mean).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        let azimuthal_avg: Vec<f64> = azimuthal_power
            .iter()
            .zip(&azimuthal_count)
            .map(|(&p, &c)| if c > 0 { p / c as f64 } else { 0.0 })
            .collect();

        let (periodic_peaks_detected, azimuthal_peak_ratio) =
            Self::detect_periodic_peaks(&azimuthal_avg);

        let high_freq_ratio = if total_energy > 0.0 {
            high_freq_energy / total_energy
        } else {
            0.0
        };
        let mid_freq_ratio = if total_energy > 0.0 {
            mid_freq_energy / total_energy
        } else {
            0.0
        };

        SpectrumStats {
            spectral_exponent,
            spectral_flatness,
            high_freq_ratio,
            mid_freq_ratio,
            periodic_peaks_detected,
            azimuthal_peak_ratio,
        }
    }

    fn estimate_spectral_exponent(radial_avg: &[f64], num_bins: usize, max_radius: f64) -> f64 {
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut n = 0.0_f64;

        for i in 2..num_bins.saturating_sub(2) {
            let power = radial_avg[i];
            if power <= 0.0 {
                continue;
            }
            let freq = (i as f64 + 0.5) / num_bins as f64 * max_radius;
            let log_freq = freq.ln();
            let log_power = power.ln();

            sum_x += log_freq;
            sum_y += log_power;
            sum_xx += log_freq * log_freq;
            sum_xy += log_freq * log_power;
            n += 1.0;
        }

        if n < 3.0 {
            return 2.0;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
        -slope
    }

    fn detect_periodic_peaks(azimuthal_avg: &[f64]) -> (bool, f64) {
        let non_zero: Vec<f64> = azimuthal_avg.iter().copied().filter(|&v| v > 0.0).collect();
        if non_zero.len() < 10 {
            return (false, 1.0);
        }

        let mut sorted = non_zero.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];

        if median <= 0.0 {
            return (false, 1.0);
        }

        let max_val = sorted.last().copied().unwrap_or(0.0);
        let peak_ratio = max_val / median;

        let threshold = median * 2.5;
        let peak_count = non_zero.iter().filter(|&&v| v > threshold).count();

        let detected = peak_ratio > 3.0 && (2..=16).contains(&peak_count);

        (detected, peak_ratio)
    }
}
