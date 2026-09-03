/**
 * Cryptographic signature validation operations.
 *
 * Wraps the Botan-based WASM crypto functions for PKCS#7/CMS
 * signature verification. All operations run in the worker thread.
 */
import type { PdfiumInstance, WasmPointer } from '@truespar/lector-pdfium-wasm';
export type SignatureValidationStatus = 'valid' | 'invalid' | 'unknown' | 'error';
export interface CertificateInfo {
    readonly subject: string;
    readonly issuer: string;
    readonly serialNumber: string;
    readonly isExpired: boolean;
    readonly isSelfSigned: boolean;
}
export interface SignatureValidationResult {
    readonly status: SignatureValidationStatus;
    readonly signerCertificate: CertificateInfo | null;
    readonly hashAlgorithm: string;
    readonly isTimestamped: boolean;
    readonly integrityValid: boolean;
    readonly signatureValid: boolean;
    readonly certificateValid: boolean;
    readonly errorMessage?: string;
}
/**
 * Validate a PKCS#7/CMS signature against PDF bytes already resident on
 * the WASM heap. This is the fast path used by validateAllSignatures —
 * it avoids a per-call malloc+memcpy of the entire PDF (which becomes
 * the dominant cost on multi-signature documents).
 *
 * @param pdfium      Pdfium WASM instance.
 * @param pkcs7Der    Raw DER-encoded PKCS#7 contents from the signature.
 * @param pdfHeapPtr  Pointer to the original PDF bytes already on the heap.
 * @param pdfHeapLen  Length of the PDF in bytes.
 * @param byteRange   Byte range array from the signature dictionary.
 */
export declare function validateSignatureOnHeap(pdfium: PdfiumInstance, pkcs7Der: Uint8Array, pdfHeapPtr: WasmPointer, pdfHeapLen: number, byteRange: readonly number[]): SignatureValidationResult;
/**
 * Legacy entry point: validate a signature against PDF bytes that live
 * in JS memory. Allocates a temporary WASM heap copy. New code should
 * prefer validateSignatureOnHeap when the bytes are already loaded.
 */
export declare function validateSignature(pdfium: PdfiumInstance, pkcs7Der: Uint8Array, pdfBytes: Uint8Array, byteRange: readonly number[]): SignatureValidationResult;
//# sourceMappingURL=crypto-ops.d.ts.map