import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
import type { AnnotationData } from '../data/types.js';
/**
 * Save the document as a new PDF. Returns the PDF bytes as ArrayBuffer.
 *
 * Uses pdfium's FPDF_SaveAsCopy with an FPDF_FILEWRITE callback that
 * accumulates chunks written by pdfium into a JS buffer. The callback is
 * registered via Emscripten's addFunction (requires ALLOW_TABLE_GROWTH or
 * a reserved function table slot in the WASM build).
 */
export declare function saveDocumentAsCopy(pdfium: PdfiumInstance, docHandle: FpdfDocument): ArrayBuffer;
/**
 * Export annotations as an XFDF XML string.
 *
 * This is a pure TypeScript serialization of AnnotationData into the
 * Adobe XFDF format. No pdfium calls are required.
 *
 * @param _pdfium - Unused; present for API consistency with other ops.
 * @param _docHandle - Unused; present for API consistency.
 * @param annotations - The annotations to serialize.
 * @returns A complete XFDF XML document string.
 */
export declare function exportXfdf(_pdfium: PdfiumInstance, _docHandle: FpdfDocument, annotations: AnnotationData[]): string;
//# sourceMappingURL=save-ops.d.ts.map