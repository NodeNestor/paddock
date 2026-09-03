/**
 * Signature operations — read digital signature data from PDFs.
 *
 * pdfium can READ signature data (contents, byte range, sub-filter, reason,
 * time, permissions) but does NOT validate signatures cryptographically.
 * Actual PKCS#7/CMS verification requires a separate crypto library.
 */
import type { PdfiumInstance, FpdfDocument } from '@truespar/lector-pdfium-wasm';
/** Information about a single digital signature in a PDF. */
export interface SignatureInfo {
    /** Zero-based index of the signature in the document. */
    readonly index: number;
    /** Raw DER-encoded PKCS#7 signature contents. */
    readonly contents: Uint8Array;
    /** Signed byte ranges [offset, length, ...]. */
    readonly byteRange: readonly number[];
    /** Signature sub-filter (e.g., "adbe.pkcs7.detached", "ETSI.CAdES.detached"). */
    readonly subFilter: string;
    /** Signer's stated reason for signing. */
    readonly reason?: string;
    /** Signing time string from the signature dictionary (not cryptographically verified). */
    readonly time?: string;
    /**
     * Document modification detection permission level:
     * 0 = unknown, 1 = no changes allowed, 2 = form fill + sign,
     * 3 = annotations + form fill + sign.
     */
    readonly docMDPPermission: number;
}
/** Get the number of digital signatures in a document. */
export declare function getSignatureCount(pdfium: PdfiumInstance, docHandle: FpdfDocument): number;
/** Read detailed information about a specific signature. */
export declare function getSignatureInfo(pdfium: PdfiumInstance, docHandle: FpdfDocument, sigIndex: number): SignatureInfo;
//# sourceMappingURL=signature-ops.d.ts.map