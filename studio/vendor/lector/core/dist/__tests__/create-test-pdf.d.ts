/**
 * Create test PDFs programmatically using pdfium's document creation API.
 *
 * This avoids external fixture files — tests are fully self-contained.
 */
import type { PdfiumInstance, FpdfDocument } from '@truespar/lector-pdfium-wasm';
/**
 * Result of creating a test document — holds the live pdfium document handle.
 * The caller is responsible for closing via `closeTestDocument`.
 */
export interface TestPdfResult {
    /** The live pdfium document handle. */
    readonly doc: FpdfDocument;
    /** Number of pages created. */
    readonly pageCount: number;
}
/**
 * Create a minimal PDF in-memory with the given number of pages.
 * Each page is Letter size (612 x 792 points) with text content.
 *
 * Returns a live document handle — do NOT pass through save/reload.
 * Call `pdfium.fn._FPDF_CloseDocument(result.doc)` when done.
 */
export declare function createTestPdf(pdfium: PdfiumInstance, options?: {
    pageCount?: number;
    /** Add a text annotation on page 0 if true. */
    withAnnotations?: boolean;
}): TestPdfResult;
//# sourceMappingURL=create-test-pdf.d.ts.map