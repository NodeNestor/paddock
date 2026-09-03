/**
 * PDF digital signing operations.
 *
 * Orchestrates the full signing flow:
 *   1. Parse PKCS#12 certificate → extract key + cert chain
 *   2. Prepare signature field (placeholder /Contents + /ByteRange)
 *   3. Save PDF with placeholder
 *   4. Find and patch /ByteRange with actual offsets
 *   5. Hash the byte ranges
 *   6. Create CMS/PKCS#7 SignedData envelope
 *   7. Inject CMS into /Contents placeholder
 */
import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
/**
 * Placement and metadata for a signature placeholder. These options
 * describe the *visual and structural* aspects of the signature field —
 * where it goes, what it looks like, what reason is recorded — but say
 * nothing about *who* is signing or *how* the cryptographic signature
 * is produced.
 *
 * Used by `prepareSignaturePlaceholder` to create the field and compute
 * the byte-range hash that a downstream signer (local PFX, remote
 * provider, etc.) will sign.
 */
export interface SignaturePlaceholderOptions {
    readonly pageIndex: number;
    readonly rectLeft: number;
    readonly rectBottom: number;
    readonly rectRight: number;
    readonly rectTop: number;
    readonly reason?: string;
    readonly signerName?: string;
    /**
     * MDP (Modification Detection and Prevention) level for certifying signatures:
     *   0 = approval signature, no restrictions (default)
     *   1 = no changes allowed after signing
     *   2 = form fill-in and signing allowed
     *   3 = form fill-in, signing, and commenting allowed
     */
    readonly mdpLevel?: 0 | 1 | 2 | 3;
    /**
     * Optional visible signature appearance (JPEG bytes). If provided,
     * a Form XObject containing the image is attached as the widget's
     * normal appearance (/AP /N), so PDF readers display the signature
     * graphic next to the certificate info.
     */
    readonly appearanceJpeg?: ArrayBuffer;
    /** Width of the appearance image in pixels (required if appearanceJpeg is set). */
    readonly appearanceWidth?: number;
    /** Height of the appearance image in pixels (required if appearanceJpeg is set). */
    readonly appearanceHeight?: number;
    /**
     * Size of the /Contents hex placeholder in bytes (= half the number of
     * hex chars). Default 16384 = 32768 hex chars, sufficient for typical
     * RSA-2048 / RSA-4096 PAdES B-T signatures including a TSA token.
     * Increase if embedding large OCSP responses for B-LT.
     */
    readonly placeholderSize?: number;
}
/**
 * Result of `prepareSignaturePlaceholder`. Carries the patched PDF bytes
 * (with the signature widget placed and `/ByteRange` already filled in)
 * plus the byte-range hash that needs to be signed and the offsets needed
 * later by `embedSignatureCms` to inject the resulting CMS.
 *
 * This is the bridge between the placement phase (which must run inside
 * the worker, using pdfium APIs) and the signing phase (which can run
 * anywhere — locally with PFX, or remotely via a network round-trip).
 */
export interface PreparedSignature {
    /** PDF bytes with the placeholder field placed and `/ByteRange` patched. */
    readonly patchedPdf: Uint8Array;
    /** Offset of the first hex character inside the `/Contents <...>` placeholder. */
    readonly contentsHexStart: number;
    /** Number of hex characters in the placeholder (excluding the `<` and `>`). */
    readonly contentsHexLen: number;
    /** The four `/ByteRange` values: [0, contentsStart, contentsEnd, remaining]. */
    readonly byteRange: readonly [number, number, number, number];
    /** Digest of the byte-range — the value that must be signed. */
    readonly hash: Uint8Array;
    /** Hash algorithm used to produce `hash`. */
    readonly hashAlgorithm: 'SHA-256';
    /**
     * Signing time captured at prepare-time. Use this exact value when
     * producing the CMS — the second call (with TSA token) must reuse it
     * so the signedAttrs SET is byte-identical and the TSA imprint stays
     * valid.
     */
    readonly signingTime: Date;
}
/**
 * Full signing options for the local-PFX path: placement options plus
 * the PFX credential and optional TSA URL. Remote-provider signing flows
 * use `SignaturePlaceholderOptions` directly and bring their own credential
 * material.
 */
export interface SigningOptions extends SignaturePlaceholderOptions {
    readonly pfxData: ArrayBuffer;
    readonly pfxPassword: string;
    /**
     * RFC 3161 Time-Stamp Authority URL. If provided, the signature is
     * upgraded to PAdES B-T by embedding a TSA timestamp token. Note that
     * most public TSA servers do not return CORS headers, so a CORS-friendly
     * TSA URL or a backend proxy is required for browser-based signing.
     */
    readonly tsaUrl?: string;
}
export interface SigningResult {
    readonly signedPdf: ArrayBuffer;
    readonly signerSubject: string;
}
/**
 * Hex-encode a CMS SignedData blob and write it into the placeholder
 * region inside the patched PDF, padding any unused space with zeros.
 *
 * Pure function — no pdfium, no async. Safe to call from any context.
 * Mutates `pdfBytes` in place and returns the same buffer for chaining.
 *
 * Used as the final phase after a signer (local PFX or remote provider)
 * has produced the CMS DER bytes for the byte-range hash returned by
 * `prepareSignaturePlaceholder`.
 *
 * @throws if the CMS is larger than the reserved placeholder.
 */
export declare function embedSignatureCms(pdfBytes: Uint8Array, cmsDer: Uint8Array, contentsHexStart: number, contentsHexLen: number): Uint8Array;
/**
 * Phase 1 of PDF signing: place the signature widget, save the document
 * with a `/Contents` hex placeholder, patch `/ByteRange` with the actual
 * offsets, and hash the byte ranges.
 *
 * Returns everything a downstream signer needs:
 *  - the patched PDF bytes (everything except the still-empty placeholder)
 *  - the byte-range hash to be signed
 *  - the offsets needed by `embedSignatureCms` to inject the resulting CMS
 *  - the signing time captured at prepare-time (must be reused by the
 *    signer to ensure signedAttrs are byte-identical across the optional
 *    TSA re-sign step)
 *
 * Mutates `docHandle` (places the signature widget). Does not consume any
 * credentials — that is the next phase.
 */
export declare function prepareSignaturePlaceholder(pdfium: PdfiumInstance, docHandle: FpdfDocument, options: SignaturePlaceholderOptions): PreparedSignature;
/**
 * End-to-end document signing with a local PFX/P12 credential.
 *
 * Composes the three signing phases:
 *  1. `prepareSignaturePlaceholder` — place widget, hash byte-range
 *  2. `produceCmsWithPfx`           — sign the hash with the PFX (in WASM)
 *  3. `embedSignatureCms`           — write CMS into the placeholder
 *
 * Future remote-signing flows reuse phases 1 and 3 directly and substitute
 * a different phase 2 (HTTP/SignalR call to a signing service that returns
 * a CMS). See `docs/architecture/eidas-bankid-signing.md`.
 */
export declare function signDocument(pdfium: PdfiumInstance, docHandle: FpdfDocument, options: SigningOptions): Promise<SigningResult>;
//# sourceMappingURL=signing-ops.d.ts.map