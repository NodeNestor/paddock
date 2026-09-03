import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
/** A single character's position and bounding box on a page. */
export interface TextCharInfo {
    readonly charCode: number;
    readonly char: string;
    readonly left: number;
    readonly right: number;
    readonly top: number;
    readonly bottom: number;
    readonly fontSize: number;
}
/** A rectangular region on a page. */
export interface TextRect {
    readonly left: number;
    readonly top: number;
    readonly right: number;
    readonly bottom: number;
}
/** A single search match result. */
export interface TextSearchMatch {
    readonly pageIndex: number;
    readonly charIndex: number;
    readonly length: number;
    readonly rects: TextRect[];
}
/**
 * Extract the full text content of a page as a single string.
 *
 * Loads the page and its text page, retrieves the character count, and reads
 * the entire text via `FPDFText_GetText` into a UTF-16 buffer which is then
 * converted to a JS string.
 */
export declare function extractPageText(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number): string;
/**
 * Extract detailed character information for every character on a page.
 *
 * For each character, retrieves the Unicode code point, bounding box
 * (left, right, top, bottom in page coordinates), and font size.
 */
export declare function extractPageCharInfo(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number): TextCharInfo[];
/**
 * Search for text on a page and return all matches with their bounding rects.
 *
 * @param flags Pdfium search flags (bitmask).
 *   - `0x00000001` — Match case
 *   - `0x00000002` — Match whole word
 *   - `0x00000004` — Match consecutive (no whitespace between search words)
 */
export declare function searchPageText(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, query: string, flags: number): TextSearchMatch[];
/**
 * Get bounding rectangles for a range of characters on a page.
 *
 * @param charIndex The starting character index.
 * @param count The number of characters in the range.
 */
export declare function getTextRects(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, charIndex: number, count: number): TextRect[];
/**
 * Get the character index at a given position on a page.
 *
 * @param x The x-coordinate in page space.
 * @param y The y-coordinate in page space.
 * @param tolerance The hit-test tolerance in both x and y directions.
 * @returns The character index, or -1 if no character is found at the position.
 */
export declare function getCharIndexAtPos(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number, x: number, y: number, tolerance: number): number;
//# sourceMappingURL=text-ops.d.ts.map