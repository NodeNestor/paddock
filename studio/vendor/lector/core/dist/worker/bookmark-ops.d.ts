/**
 * Bookmark (outline) editing operations.
 *
 * Wraps custom pdfium C API extensions for creating, deleting, moving,
 * and editing bookmarks. These functions modify the document's /Outlines
 * dictionary in-place.
 */
import type { PdfiumInstance, FpdfDocument } from '@truespar/lector-pdfium-wasm';
/**
 * Add a new top-level bookmark.
 *
 * @param title Display title for the bookmark.
 * @param pageIndex Target page (0-based). Use -1 for no destination.
 * @param insertIndex Position among siblings (0 = first, -1 = append).
 * @returns true on success.
 */
export declare function addBookmark(pdfium: PdfiumInstance, docHandle: FpdfDocument, title: string, pageIndex: number, insertIndex: number): boolean;
/**
 * Delete a top-level bookmark by index.
 */
export declare function deleteBookmark(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number): boolean;
/**
 * Move a top-level bookmark from one index to another.
 */
export declare function moveBookmark(pdfium: PdfiumInstance, docHandle: FpdfDocument, fromIndex: number, toIndex: number): boolean;
/**
 * Set the title of a top-level bookmark.
 */
export declare function setBookmarkTitle(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number, title: string): boolean;
/**
 * Set the destination page of a top-level bookmark.
 */
export declare function setBookmarkDest(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number, pageIndex: number): boolean;
/**
 * Get the number of top-level bookmarks.
 */
export declare function getBookmarkCount(pdfium: PdfiumInstance, docHandle: FpdfDocument): number;
//# sourceMappingURL=bookmark-ops.d.ts.map