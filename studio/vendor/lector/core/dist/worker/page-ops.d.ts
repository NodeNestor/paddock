/**
 * Page-level operations — delete, insert, rotate, reorder, duplicate, flatten.
 *
 * Wraps pdfium C API page manipulation functions. These are destructive
 * operations that modify the document in-place.
 */
import type { PdfiumInstance, FpdfDocument } from '@truespar/lector-pdfium-wasm';
import type { RedactionSpec } from '../types/worker-api.js';
/**
 * Delete a page from the document.
 *
 * After deletion, all page indices after `pageIndex` shift down by one.
 * The caller must re-read page sizes and update any cached state.
 */
export declare function deletePage(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number): void;
/**
 * Insert a new blank page at the given index.
 *
 * @param width Page width in PDF points (72 points = 1 inch).
 * @param height Page height in PDF points.
 */
export declare function insertBlankPage(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, width: number, height: number): void;
/**
 * Set the rotation of a page.
 *
 * @param rotation 0 = 0°, 1 = 90° CW, 2 = 180°, 3 = 270° CW.
 */
export declare function rotatePage(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, rotation: 0 | 1 | 2 | 3): void;
/**
 * Get the rotation of a page.
 *
 * @returns 0 = 0°, 1 = 90° CW, 2 = 180°, 3 = 270° CW.
 */
export declare function getPageRotation(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number): number;
/**
 * Move a page from one position to another.
 *
 * Uses FPDF_MovePages which takes an array of page indices and a
 * destination index. For a single page move, we pass a 1-element array.
 */
export declare function movePage(pdfium: PdfiumInstance, docHandle: FpdfDocument, fromIndex: number, toIndex: number): void;
/**
 * Duplicate a page within the same document.
 *
 * Uses FPDF_ImportPages with the document as both source and target.
 * The duplicated page is inserted after the source page.
 */
export declare function duplicatePage(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number): void;
/**
 * Import pages from a source document into a target document.
 *
 * @param pageRange Comma-separated 1-indexed page range (e.g., "1,3,5-7")
 *                  or empty string for all pages.
 * @param insertIndex 0-based insertion point in the target document.
 */
export declare function importPages(pdfium: PdfiumInstance, sourceDoc: FpdfDocument, targetDoc: FpdfDocument, pageRange: string, insertIndex: number): void;
/**
 * Flatten annotations on a page, merging them into the page content.
 *
 * @param flag 0 = flatten for display, 1 = flatten for print.
 * @returns 0 = success, 1 = nothing to flatten, 2 = failure.
 */
export declare function flattenPage(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, flag?: 0 | 1): number;
/**
 * Apply redactions on a page from a set of explicit specs.
 *
 * This is a true content-destructive redaction. Two strategies are used:
 *
 * - **Top-level content** is removed object-by-object: every page object that
 *   overlaps a rect is deleted and a fill box is drawn. Text elsewhere on the
 *   page stays selectable.
 * - **Form-nested content** (objects inside a Form XObject) cannot be purged
 *   by object removal — `FPDFPage_GenerateContent` does not rewrite nested
 *   form streams, so the content survives reload. When a redaction overlaps
 *   such content the whole page is rasterized (see {@link rasterizeRedactions})
 *   so the content is physically destroyed. This is detected automatically.
 *
 * After content removal it draws optional overlay text, optionally deletes the
 * redaction annotations, then regenerates the page content stream.
 *
 * @param removeAnnots When true, redaction annotations on the page are also
 *   deleted. The primary worker passes true; render-pool workers pass false
 *   because their document copies never received the annotations.
 */
export declare function applyRedactions(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, specs: readonly RedactionSpec[], removeAnnots: boolean): void;
//# sourceMappingURL=page-ops.d.ts.map