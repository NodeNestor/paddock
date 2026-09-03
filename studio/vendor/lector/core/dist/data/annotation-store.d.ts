import type { ReadonlySignal, Unsubscribe } from '@truespar/lector-utils';
import type { AnnotationData, DataEvent, TrackedObject } from './types.js';
import { OperationLog } from './operation-log.js';
import type { EventBus } from '../plugin/event-bus.js';
/**
 * Per-document annotation state on the main thread.
 *
 * Bridges the pdfium worker (which extracts annotation data) and the
 * consumer application (which reacts to annotation events). Every mutation
 * is recorded in an {@link OperationLog} and broadcast via the
 * {@link EventBus}.
 *
 * Each document gets its own {@link DirtyTracker} instance so commit state
 * is isolated per document.
 */
export declare class AnnotationStore implements Disposable {
    #private;
    constructor(eventBus: EventBus, userId?: string);
    /**
     * Load annotations for a page from the worker result.
     *
     * Existing annotations for the same page are replaced. Each loaded
     * annotation starts in the 'synced' commit state because it already
     * exists in the PDF.
     */
    loadPage(documentId: string, pageIndex: number, annotations: AnnotationData[]): void;
    /** Record a new annotation (from user action or API). */
    create(documentId: string, annotation: AnnotationData): TrackedObject<AnnotationData>;
    /** Record an annotation update. */
    update(documentId: string, annotationId: string, patch: Partial<AnnotationData>): void;
    /** Record an annotation deletion. */
    delete(documentId: string, annotationId: string): void;
    /** Mark a single annotation as synced. */
    markSynced(documentId: string, annotationId: string): void;
    /** Mark all annotations for a document as synced. */
    markAllSynced(documentId: string): void;
    /** Get all annotations for a document. */
    getForDocument(documentId: string): TrackedObject<AnnotationData>[];
    /** Get annotations for a specific page. */
    getForPage(documentId: string, pageIndex: number): TrackedObject<AnnotationData>[];
    /** Get dirty annotations for a document. */
    getDirty(documentId: string): TrackedObject<AnnotationData>[];
    /** Check if a document has unsaved annotation changes. */
    hasDirty(documentId: string): ReadonlySignal<boolean>;
    /** Subscribe to annotation events from the operation log. */
    subscribe(fn: (event: DataEvent<AnnotationData>) => void): Unsubscribe;
    /** Get the operation log (for sync/replay). */
    get log(): OperationLog<AnnotationData>;
    /** Clean up all state for a document. */
    clearDocument(documentId: string): void;
    /** Dispose: clean up all documents. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=annotation-store.d.ts.map