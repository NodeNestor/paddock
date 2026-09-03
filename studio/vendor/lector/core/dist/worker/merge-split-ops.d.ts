/**
 * Document merging and splitting operations.
 *
 * Uses pdfium's FPDF_ImportPages / FPDF_ImportPagesByIndex to copy pages
 * between documents, and FPDF_CreateNewDocument for fresh targets.
 */
import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
/**
 * Merge multiple source documents into a single new PDF.
 *
 * Pages from each source are appended in order. The caller is responsible
 * for the source document handles — they are NOT closed by this function.
 *
 * @returns The merged PDF as an ArrayBuffer.
 */
export declare function mergeDocuments(pdfium: PdfiumInstance, sourceHandles: readonly FpdfDocument[]): ArrayBuffer;
/**
 * Split a document into multiple PDFs by page ranges.
 *
 * Each range is `{ start, end }` with 0-based inclusive page indices.
 * Returns one ArrayBuffer per range.
 */
export declare function splitDocument(pdfium: PdfiumInstance, sourceDoc: FpdfDocument, ranges: ReadonlyArray<{
    start: number;
    end: number;
}>): ArrayBuffer[];
/**
 * Extract specific pages from a document into a new PDF.
 *
 * @param pageIndices 0-based page indices to extract.
 * @returns The extracted pages as a single PDF ArrayBuffer.
 */
export declare function extractPages(pdfium: PdfiumInstance, sourceDoc: FpdfDocument, pageIndices: readonly number[]): ArrayBuffer;
//# sourceMappingURL=merge-split-ops.d.ts.map