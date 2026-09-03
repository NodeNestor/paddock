/**
 * Document security (password protection) operations.
 *
 * Wraps the custom pdfium C API for setting AES-256 encryption
 * on a PDF document. After calling setPassword(), the next
 * FPDF_SaveAsCopy will produce an encrypted PDF.
 */
import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
export interface PasswordProtectionOptions {
    /** Password required to open the document. */
    readonly userPassword: string;
    /** Password for full access (remove encryption, change permissions). Defaults to userPassword. */
    readonly ownerPassword?: string;
    /** Allow printing. Default: true. */
    readonly allowPrint?: boolean;
    /** Allow modifying document contents. Default: false. */
    readonly allowModify?: boolean;
    /** Allow copying/extracting text and graphics. Default: true. */
    readonly allowExtract?: boolean;
    /** Allow adding annotations and filling forms. Default: true. */
    readonly allowAnnotate?: boolean;
}
/**
 * Set AES-256 password protection on a document.
 *
 * After calling this, saving the document will produce an encrypted PDF.
 */
export declare function setDocumentPassword(pdfium: PdfiumInstance, docHandle: FpdfDocument, options: PasswordProtectionOptions): void;
//# sourceMappingURL=security-ops.d.ts.map