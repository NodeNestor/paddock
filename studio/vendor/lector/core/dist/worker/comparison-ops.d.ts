/**
 * Document comparison operations.
 *
 * Computes a structured diff between two PDF documents:
 *   1. Page-level alignment via LCS on per-page text fingerprints
 *      (handles inserted / deleted / reordered pages).
 *   2. Per-aligned-page comparison in one of three modes:
 *      - `text`        — both pages have extractable text → word-level
 *                        Myers diff with char-rect mapping
 *      - `region`      — both pages are image-only → low-DPI pixel diff
 *                        with tile grouping into bounding rectangles
 *      - `mismatched`  — one page has text, the other doesn't → emit
 *                        a single warning change, do not attempt diff
 *
 * Pure data output: the result has no DOM/UI dependencies and can be
 * serialised across the worker boundary without modification.
 */
import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
import type { TextRect } from './text-ops.js';
/** How a single aligned page-pair was compared. */
export type PageComparisonMode = 'text' | 'region' | 'mismatched' | 'identical' | 'inserted' | 'deleted';
/** A single change inside a page. */
export interface ComparisonChange {
    /** What kind of edit this represents. */
    readonly type: 'insert' | 'delete' | 'replace' | 'region';
    /** Page index in document A, or null if the change is "inserted in B". */
    readonly pageA: number | null;
    /** Page index in document B, or null if the change is "deleted from A". */
    readonly pageB: number | null;
    /** Bounding rect on page A, in PDF points. */
    readonly rectA?: TextRect;
    /** Bounding rect on page B, in PDF points. */
    readonly rectB?: TextRect;
    /** Old text (for delete / replace). */
    readonly textBefore?: string;
    /** New text (for insert / replace). */
    readonly textAfter?: string;
    /**
     * Estimated proportion of pixels that differ in this region (0–1).
     * Only set for `region` changes.
     */
    readonly pixelDelta?: number;
}
/** Per-page comparison result. */
export interface PageDiff {
    readonly pageA: number | null;
    readonly pageB: number | null;
    readonly mode: PageComparisonMode;
    readonly changes: readonly ComparisonChange[];
}
/** Top-level comparison result. */
export interface ComparisonResult {
    readonly pageCountA: number;
    readonly pageCountB: number;
    /** Per-page diffs, in document-A order with inserts spliced in. */
    readonly pageDiffs: readonly PageDiff[];
    /** Total number of changes across the whole comparison. */
    readonly totalChanges: number;
}
/**
 * Compare two open documents and produce a structured diff.
 *
 * Both `docHandleA` and `docHandleB` must remain valid for the
 * duration of the call. Pages are loaded on demand from each handle
 * and closed immediately after use.
 */
export declare function compareDocuments(pdfium: PdfiumInstance, docHandleA: FpdfDocument, pageCountA: number, pageSizesA: ReadonlyArray<{
    width: number;
    height: number;
}>, docHandleB: FpdfDocument, pageCountB: number, pageSizesB: ReadonlyArray<{
    width: number;
    height: number;
}>): ComparisonResult;
//# sourceMappingURL=comparison-ops.d.ts.map