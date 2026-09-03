/**
 * Attachment operations — manage embedded file attachments in PDFs.
 *
 * Wraps pdfium's FPDFDoc_* and FPDFAttachment_* functions for reading,
 * adding, and deleting file attachments embedded in the PDF document.
 */
import type { PdfiumInstance, FpdfDocument } from '@truespar/lector-pdfium-wasm';
/** Information about an embedded file attachment. */
export interface AttachmentInfo {
    /** Zero-based index in the document's attachment list. */
    readonly index: number;
    /** File name of the attachment. */
    readonly name: string;
    /** File size in bytes. */
    readonly size: number;
    /** Creation date string from attachment metadata. */
    readonly creationDate?: string;
    /** Modification date string from attachment metadata. */
    readonly modDate?: string;
}
/** Get the number of embedded file attachments in a document. */
export declare function getAttachmentCount(pdfium: PdfiumInstance, docHandle: FpdfDocument): number;
/** Read metadata for a specific attachment. */
export declare function getAttachmentInfo(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number): AttachmentInfo;
/** Read the file content of an attachment as an ArrayBuffer. */
export declare function getAttachmentData(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number): ArrayBuffer;
/** Add a new file attachment to the document. */
export declare function addAttachment(pdfium: PdfiumInstance, docHandle: FpdfDocument, name: string, data: ArrayBuffer): void;
/** Delete an attachment from the document by index. */
export declare function deleteAttachment(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number): void;
//# sourceMappingURL=attachment-ops.d.ts.map