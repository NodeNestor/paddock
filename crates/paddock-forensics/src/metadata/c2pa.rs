//! C2PA content-credential checker (JUMBF/JPEG APP11, PNG caBX/iTXt), ported
//! verbatim from the reference. Detects presence / absence-with-removal-traces /
//! malformed manifests; presence is a positive trust signal. CPU-only.
//!
//! Full cryptographic signature verification (certificate chain against a trust
//! list) is out of scope here, exactly as in the reference.

use crate::analyzer::Analyzer;
use crate::{Context, Finding, Severity};

pub struct C2paChecker;

impl Analyzer for C2paChecker {
    fn name(&self) -> &'static str {
        "c2pa"
    }

    fn cpu(&self, ctx: &Context) -> Vec<Finding> {
        let mut findings = Vec::new();

        let c2pa_status = self.detect_c2pa(&ctx.raw_bytes);

        match c2pa_status {
            C2paStatus::Present { claim_info } => {
                findings.push(Finding::new(
                    "c2pa",
                    "c2pa_manifest_present",
                    format!(
                        "C2PA content credentials found: {claim_info} - image has provenance \
                         information embedded (positive trust signal)"
                    ),
                    Severity::Info,
                    0.95,
                ));
            }
            C2paStatus::Absent => {
                let had_c2pa = self.detect_removed_c2pa(&ctx.raw_bytes, &ctx.tags);

                if had_c2pa {
                    findings.push(Finding::new(
                        "c2pa",
                        "c2pa_removed",
                        "Evidence of removed C2PA content credentials - provenance \
                         data was stripped, which is suspicious for claimed originals",
                        Severity::Medium,
                        0.6,
                    ));
                }
                // Absence alone is not a finding - most images have no C2PA yet.
            }
            C2paStatus::Malformed => {
                findings.push(Finding::new(
                    "c2pa",
                    "c2pa_malformed",
                    "Malformed C2PA manifest detected - credentials are present but \
                     corrupted, possibly due to re-encoding or manipulation",
                    Severity::Medium,
                    0.7,
                ));
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

enum C2paStatus {
    Present { claim_info: String },
    Absent,
    Malformed,
}

impl C2paChecker {
    fn detect_c2pa(&self, raw_bytes: &[u8]) -> C2paStatus {
        if raw_bytes.len() >= 2 && raw_bytes[0] == 0xFF && raw_bytes[1] == 0xD8 {
            return self.detect_c2pa_jpeg(raw_bytes);
        }

        if raw_bytes.len() >= 8 && &raw_bytes[0..8] == b"\x89PNG\r\n\x1a\n" {
            return self.detect_c2pa_png(raw_bytes);
        }

        C2paStatus::Absent
    }

    fn detect_c2pa_jpeg(&self, data: &[u8]) -> C2paStatus {
        let mut pos = 2;
        let mut found_jumbf = false;
        let mut c2pa_data = Vec::new();

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

            // APP11 = 0xEB - JUMBF container.
            if marker == 0xEB && length > 10 {
                let payload = &data[pos + 2..pos + length];

                if Self::contains_c2pa_uuid(payload) {
                    found_jumbf = true;
                    c2pa_data.extend_from_slice(payload);
                }

                if payload.len() >= 2 && payload[0] == b'J' && payload[1] == b'P' {
                    found_jumbf = true;
                }
            }

            if marker == 0xDA {
                break;
            }

            pos += length;
        }

        if found_jumbf {
            let claim_info = self.extract_basic_claim_info(&c2pa_data);
            if claim_info.is_empty() {
                C2paStatus::Malformed
            } else {
                C2paStatus::Present { claim_info }
            }
        } else {
            C2paStatus::Absent
        }
    }

    fn detect_c2pa_png(&self, data: &[u8]) -> C2paStatus {
        let mut pos = 8; // Skip PNG signature.

        while pos + 12 < data.len() {
            let chunk_len =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            let chunk_type = &data[pos + 4..pos + 8];

            if chunk_type == b"caBX" {
                let payload = &data[pos + 8..pos + 8 + chunk_len.min(data.len() - pos - 8)];
                let claim_info = self.extract_basic_claim_info(payload);
                return if claim_info.is_empty() {
                    C2paStatus::Malformed
                } else {
                    C2paStatus::Present { claim_info }
                };
            }

            if chunk_type == b"iTXt" && chunk_len > 4 {
                let payload = &data[pos + 8..pos + 8 + chunk_len.min(data.len() - pos - 8)];
                if payload.windows(4).any(|w| w == b"c2pa") {
                    return C2paStatus::Present {
                        claim_info: "C2PA manifest in iTXt chunk".into(),
                    };
                }
            }

            pos += 12 + chunk_len; // 4 (len) + 4 (type) + data + 4 (CRC)
        }

        C2paStatus::Absent
    }

    fn contains_c2pa_uuid(payload: &[u8]) -> bool {
        // C2PA manifest store UUID.
        let c2pa_uuid: [u8; 16] = [
            0x63, 0x32, 0x70, 0x61, 0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
            0x9B, 0x71,
        ];

        // C2PA claim UUID.
        let c2pa_claim_uuid: [u8; 16] = [
            0x63, 0x32, 0x63, 0x6C, 0x00, 0x11, 0x00, 0x10, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38,
            0x9B, 0x71,
        ];

        payload
            .windows(16)
            .any(|w| w == c2pa_uuid || w == c2pa_claim_uuid)
    }

    fn extract_basic_claim_info(&self, data: &[u8]) -> String {
        let mut info_parts = Vec::new();

        let searchable = String::from_utf8_lossy(data);

        if searchable.contains("c2pa.action") || searchable.contains("c2pa.claim") {
            info_parts.push("C2PA claim manifest");
        }

        if searchable.contains("dc:creator") {
            info_parts.push("creator attribution present");
        }

        if searchable.contains("stds.exif") {
            info_parts.push("EXIF assertion present");
        }

        if searchable.contains("c2pa.hash") {
            info_parts.push("content hash binding");
        }

        if info_parts.is_empty() {
            "C2PA JUMBF container detected".into()
        } else {
            info_parts.join(", ")
        }
    }

    fn detect_removed_c2pa(&self, raw_bytes: &[u8], tags: &[sift::Tag]) -> bool {
        let has_c2pa_xmp_ref = tags.iter().any(|t| {
            let v = t.value.to_lowercase();
            v.contains("c2pa") || v.contains("content credentials") || v.contains("jumbf")
        });

        let has_c2pa_tool = tags.iter().any(|t| {
            let v = t.value.to_lowercase();
            t.name.to_lowercase().contains("software")
                && (v.contains("content authenticity")
                    || v.contains("cai ")
                    || v.contains("verify"))
        });

        let has_residual = raw_bytes
            .windows(4)
            .any(|w| w == b"jumb" || w == b"jumd" || w == b"jums");

        has_c2pa_xmp_ref || has_c2pa_tool || has_residual
    }
}
