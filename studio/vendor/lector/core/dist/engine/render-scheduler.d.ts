import type * as Comlink from 'comlink';
import type { DocumentId, TaskId } from '../types/handle-id.js';
import type { RenderOptions, RenderPriority } from '../types/render.js';
import type { PdfiumWorkerApi } from '../types/worker-api.js';
import type { RenderPool } from './render-pool.js';
/** A request to render a single page at specific pixel dimensions. */
export interface RenderRequest {
    readonly docId: DocumentId;
    readonly pageIndex: number;
    readonly width: number;
    readonly height: number;
    readonly options?: RenderOptions;
    readonly priority?: RenderPriority;
    readonly signal?: AbortSignal;
}
/**
 * Priority queue with cancellation for background page rendering.
 *
 * Renders are dispatched one at a time to the worker. The queue is sorted by
 * priority (lower = higher priority), then by insertion order (FIFO within
 * the same priority level).
 *
 * Identical requests (same doc + page + dimensions) are deduplicated: the
 * existing promise is returned instead of enqueueing a duplicate.
 */
export declare class RenderScheduler implements Disposable {
    #private;
    constructor(proxy: Comlink.Remote<PdfiumWorkerApi>, pool?: RenderPool);
    /**
     * Enqueue a page render request.
     *
     * Returns a promise that resolves with the rendered ImageBitmap.
     * If an identical request is already pending, returns the existing promise.
     */
    enqueue(request: RenderRequest): Promise<ImageBitmap>;
    /** Cancel a task by ID. Removes it from the queue or discards the active result. */
    cancel(taskId: TaskId): void;
    /**
     * Cancel every queued and in-flight render for a document. Call this when a
     * document is closing so outstanding renders don't resolve against a closed
     * document: their promises reject with AbortError, and any late-arriving
     * pool bitmap is dropped and closed by the success guard (no leak).
     */
    cancelDocument(docId: DocumentId): void;
    /**
     * Change the priority of all pending tasks matching a specific document and page.
     * Tasks that are already actively rendering are not affected.
     */
    reprioritize(docId: DocumentId, pageIndex: number, newPriority: RenderPriority): void;
    /** Dispose the scheduler, rejecting all pending tasks. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=render-scheduler.d.ts.map