import type { DocumentId } from '../types/handle-id.js';
/**
 * Page manipulation capability.
 *
 * Provides operations to structurally modify documents: delete pages,
 * insert blank pages, rotate pages, reorder pages, duplicate pages,
 * and flatten annotations into page content.
 *
 * After every mutation, page sizes are re-read from the worker and
 * a `page-ops:pages-changed` event is emitted so the viewport and
 * render plugins can relayout.
 */
export interface PageOperationsCapability {
    /** Delete a page from the document. */
    deletePage(docId: DocumentId, pageIndex: number): Promise<void>;
    /** Insert a new blank page at the given index. Width/height in PDF points. */
    insertBlankPage(docId: DocumentId, pageIndex: number, width: number, height: number): Promise<void>;
    /**
     * Rotate a page.
     * @param degrees 0, 90, 180, or 270 (clockwise).
     */
    rotatePage(docId: DocumentId, pageIndex: number, degrees: 0 | 90 | 180 | 270): Promise<void>;
    /** Move a page from one index to another. */
    movePage(docId: DocumentId, fromIndex: number, toIndex: number): Promise<void>;
    /** Duplicate a page (inserted after the source page). */
    duplicatePage(docId: DocumentId, pageIndex: number): Promise<void>;
    /** Flatten annotations on a page into the page content (irreversible). */
    flattenPage(docId: DocumentId, pageIndex: number): Promise<void>;
}
export declare const pageOpsPlugin: import("../index.js").PluginDefinition<PageOperationsCapability, Record<string, never>>;
//# sourceMappingURL=page-ops-plugin.d.ts.map