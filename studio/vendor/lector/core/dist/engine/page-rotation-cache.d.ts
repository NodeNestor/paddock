import type { DocumentId } from '../types/handle-id.js';
/** Page rotation in 90° clockwise steps: 0=0°, 1=90°, 2=180°, 3=270°. */
export type PageRotation = 0 | 1 | 2 | 3;
/**
 * Shared, engine-level cache of per-page rotation.
 *
 * pdfium only exposes a page's rotation through a loaded page handle
 * (`FPDFPage_GetRotation`), and the document-open path deliberately avoids
 * loading every page. So rotation is fetched lazily the first time a consumer
 * needs it and cached here for the rest of the session.
 *
 * Several subsystems need the same value — the overlay layer (to place
 * rendered overlays), text selection (to hit-test), and annotation creation
 * (to map a click to PDF space). Annotation creation runs inside synchronous
 * pointer handlers, so it needs a *synchronous* read ({@link get}); the overlay
 * layer warms the cache for visible pages via {@link resolve}, and a page can
 * only be drawn on while it is visible, so the synchronous read is warm in
 * practice. Unknown pages read as `0` (the common, unrotated case).
 */
export declare class PageRotationCache implements Disposable {
    #private;
    /**
     * @param fetch Loads a page's raw rotation from the worker
     *   (`FPDFPage_GetRotation`). Only invoked after the engine is initialized.
     */
    constructor(fetch: (docId: DocumentId, pageIndex: number) => Promise<number>);
    /** Synchronous read. Returns the cached rotation, or 0 if not yet resolved. */
    get(docId: DocumentId, pageIndex: number): PageRotation;
    /** Whether this page's rotation has been resolved (vs. defaulting to 0). */
    has(docId: DocumentId, pageIndex: number): boolean;
    /**
     * Resolve and cache a page's rotation. Concurrent calls for the same page
     * share one worker request. Returns 0 if the worker call fails.
     */
    resolve(docId: DocumentId, pageIndex: number): Promise<PageRotation>;
    /** Invalidate one page's cached rotation (e.g. after a rotate operation). */
    invalidate(docId: DocumentId, pageIndex: number): void;
    /** Drop all cached rotations for a document (on close). */
    clearDocument(docId: DocumentId): void;
    [Symbol.dispose](): void;
}
//# sourceMappingURL=page-rotation-cache.d.ts.map