/**
 * Render Worker Pool — parallel page rendering across multiple pdfium workers.
 *
 * Each worker has its own pdfium WASM instance and independently-loaded
 * copies of open documents. Page render requests are distributed round-robin
 * across the pool. Document lifecycle (open/close) is broadcast to all workers.
 *
 * This is an opt-in performance feature. When `renderPoolSize` is 0 (default),
 * the primary worker handles all rendering (serial dispatch). When > 0, the
 * pool workers handle rendering while the primary worker handles text, search,
 * annotations, and other stateful operations.
 */
import type { DocumentId } from '../types/handle-id.js';
import type { RenderOptions } from '../types/render.js';
import type { RedactionSpec } from '../types/worker-api.js';
/**
 * Pool of render workers for parallel page rendering.
 */
export declare class RenderPool implements Disposable {
    #private;
    get size(): number;
    /**
     * Create and initialize the pool.
     *
     * @param poolSize - Number of render workers (0 = disabled)
     * @param workerUrl - URL of the worker script (same as primary worker)
     * @param wasmUrl - URL of pdfium.wasm
     * @param wasmJsUrl - URL of pdfium.js
     */
    init(poolSize: number, workerUrl: string | URL, wasmUrl: string, wasmJsUrl: string): Promise<void>;
    /**
     * Open a document on all pool workers. Must be called after the primary
     * worker opens the document (so we have the data + docId).
     *
     * @param docId - The document ID assigned by the primary worker
     * @param data - Raw PDF bytes
     * @param password - Optional password
     */
    openDocument(docId: DocumentId, data: ArrayBuffer, password?: string): Promise<void>;
    /**
     * Close a document on all pool workers.
     */
    closeDocument(docId: DocumentId): Promise<void>;
    /**
     * Re-run a destructive page mutation on every pool worker so their
     * pdfium copies stay in sync with the primary worker. Caller is
     * responsible for having already executed the same op on the primary.
     * Resolves only after all pool workers complete.
     */
    applyRedactions(docId: DocumentId, pageIndex: number, specs: readonly RedactionSpec[]): Promise<void>;
    /**
     * Render a page on an available pool worker (round-robin over the workers
     * that actually hold this document). Returns null when no pool worker can
     * serve it — the document isn't mapped, every copy is stale, or the pool is
     * empty — and the scheduler then renders on the primary worker. The
     * primary's `docId` is translated to each worker's own id; we never assume
     * the ids coincide.
     */
    renderPage(docId: DocumentId, pageIndex: number, widthPx: number, heightPx: number, options?: RenderOptions): Promise<ImageBitmap> | null;
    /**
     * Mark a document's pool copies stale — used after a structural page
     * mutation (delete/insert/move/rotate/duplicate/flatten) that the pool
     * cannot replay. Renders fall back to the primary worker until the document
     * is closed, guaranteeing correct content at the cost of pool parallelism
     * for that document.
     */
    invalidate(docId: DocumentId): void;
    /** Dispose all pool workers. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=render-pool.d.ts.map