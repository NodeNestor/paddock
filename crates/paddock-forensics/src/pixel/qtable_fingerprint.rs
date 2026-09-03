//! JPEG quantization-table source fingerprinting, ported verbatim from
//! the CPU reference. CPU-only (byte parsing + small correlations, no
//! GPU kernel); `gpu()` delegates to `cpu()`.
//!
//! Different cameras and editors ship different JPEG quantization tables.
//! A table that matches no standard IJG/camera profile, or an unusual
//! luma/chroma ratio, points at editing software. This is distinct from
//! double-JPEG detection (recompression artifacts) - here we fingerprint the
//! *specific source* of the tables. JPEG-only (skipped for non-JPEG).

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct QtableFingerprintAnalyzer;

impl Analyzer for QtableFingerprintAnalyzer {
    fn name(&self) -> &'static str {
        "qtable_fingerprint"
    }

    fn applies_to(&self, ctx: &Context) -> bool {
        ctx.is_jpeg()
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let mut findings = Vec::new();

        // Belt-and-braces: applies_to already gates on JPEG, but keep the raw
        // SOI check so the parser never runs on non-JPEG bytes.
        if ctx.raw_bytes.len() < 4 || ctx.raw_bytes[0] != 0xFF || ctx.raw_bytes[1] != 0xD8 {
            return findings;
        }

        let qtables = extract_qtables(&ctx.raw_bytes);
        if qtables.is_empty() {
            return findings;
        }

        let origin = identify_qtable_origin(&qtables[0]);
        let standard_score = qtable_standard_similarity(&qtables[0]);

        if standard_score < 0.5 {
            findings.push(Finding::new(
                "qtable_fingerprint",
                "qtable_non_standard",
                format!(
                    "JPEG quantization table does not match standard IJG/camera profiles \
                     (similarity {:.0}%) - indicates image processing software or editing \
                     tool was used{}",
                    standard_score * 100.0,
                    if let Some(ref o) = origin {
                        format!(", Q-table consistent with {o}")
                    } else {
                        String::new()
                    },
                ),
                Severity::Info,
                0.60,
            ));
        }

        if let Some(ref origin_name) = origin
            && (origin_name.contains("Photoshop") || origin_name.contains("editor"))
        {
            findings.push(Finding::new(
                "qtable_fingerprint",
                "qtable_editor_signature",
                format!(
                    "JPEG quantization table matches {origin_name} - \
                         image was processed by editing software"
                ),
                Severity::Low,
                0.65,
            ));
        }

        // Multi-table inconsistency (luminance vs chrominance sum ratio).
        if qtables.len() >= 2 {
            let luma_sum: u32 = qtables[0].iter().map(|&v| v as u32).sum();
            let chroma_sum: u32 = qtables[1].iter().map(|&v| v as u32).sum();

            if luma_sum > 0 {
                let ratio = chroma_sum as f64 / luma_sum as f64;
                // Standard IJG chroma/luma ratio sits around 1.5-2.0.
                if !(0.8..=3.5).contains(&ratio) {
                    findings.push(Finding::new(
                        "qtable_fingerprint",
                        "qtable_unusual_chroma_ratio",
                        format!(
                            "Unusual luminance/chrominance Q-table ratio ({ratio:.2}) - \
                             non-standard JPEG encoder or custom processing pipeline"
                        ),
                        Severity::Low,
                        0.50,
                    ));
                }
            }
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

fn extract_qtables(data: &[u8]) -> Vec<[u16; 64]> {
    let mut tables = Vec::new();
    let mut pos = 2;

    while pos + 4 < data.len() {
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }
        let marker = data[pos + 1];
        pos += 2;

        if marker == 0x00 || marker == 0xFF || (0xD0..=0xD9).contains(&marker) {
            continue;
        }
        if pos + 2 > data.len() {
            break;
        }

        let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        if length < 2 || pos + length > data.len() {
            break;
        }

        if marker == 0xDB {
            let mut tpos = pos + 2;
            while tpos < pos + length {
                if tpos >= data.len() {
                    break;
                }
                let precision = (data[tpos] >> 4) & 0x0F;
                tpos += 1;

                let mut table = [0u16; 64];
                for entry in &mut table {
                    if precision == 0 {
                        if tpos >= data.len() {
                            break;
                        }
                        *entry = data[tpos] as u16;
                        tpos += 1;
                    } else {
                        if tpos + 1 >= data.len() {
                            break;
                        }
                        *entry = u16::from_be_bytes([data[tpos], data[tpos + 1]]);
                        tpos += 2;
                    }
                }
                tables.push(table);
            }
        }

        if marker == 0xDA {
            break;
        }
        pos += length;
    }

    tables
}

/// Best correlation of the table against the standard IJG luminance table
/// scaled across quality levels.
fn qtable_standard_similarity(qtable: &[u16; 64]) -> f64 {
    let standard: [u16; 64] = [
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
    ];

    let mut best_sim = 0.0_f64;
    for q in (1..=100).step_by(5) {
        let scaled = scale_qtable(&standard, q);
        let sim = table_correlation(qtable, &scaled);
        best_sim = best_sim.max(sim);
    }
    best_sim
}

fn scale_qtable(base: &[u16; 64], quality: u32) -> [u16; 64] {
    let scale = if quality < 50 {
        5000 / quality
    } else {
        200 - quality * 2
    };

    let mut result = [0u16; 64];
    for (i, &base_val) in base.iter().enumerate() {
        let val = (base_val as u32 * scale + 50) / 100;
        result[i] = val.clamp(1, 255) as u16;
    }
    result
}

fn table_correlation(a: &[u16; 64], b: &[u16; 64]) -> f64 {
    let mean_a: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / 64.0;
    let mean_b: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / 64.0;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;

    for i in 0..64 {
        let da = a[i] as f64 - mean_a;
        let db = b[i] as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).max(1e-10).sqrt();
    (cov / denom).max(0.0)
}

fn identify_qtable_origin(qtable: &[u16; 64]) -> Option<String> {
    // Photoshop's high-quality tables start with a distinctive low triple.
    let ps_signature = qtable[0] == 2 && qtable[1] == 1 && qtable[2] == 1;
    if ps_signature {
        return Some("Adobe Photoshop (high quality)".into());
    }

    // A very flat table (all values close) points at heavy editing.
    let min_val = *qtable.iter().min().unwrap_or(&1);
    let max_val = *qtable.iter().max().unwrap_or(&255);
    if max_val > 0 && (max_val as f64 / min_val.max(1) as f64) < 1.5 {
        return Some("image editor (flat Q-table)".into());
    }

    None
}
