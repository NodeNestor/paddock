import type { ReadonlySignal, Unsubscribe } from '@truespar/lector-utils';
import type { DataEvent, TrackedObject, WidgetData } from './types.js';
import { OperationLog } from './operation-log.js';
import type { EventBus } from '../plugin/event-bus.js';
/**
 * Per-document form field state on the main thread.
 *
 * Same structural pattern as {@link AnnotationStore} but specialised for
 * PDF form fields ({@link WidgetData}). Tracks field value changes, emits
 * events via the {@link EventBus}, and records every mutation in an
 * {@link OperationLog} for sync/replay.
 *
 * Form fields are keyed by `fieldName` (the PDF field name is unique within
 * a document, though the same field may appear on multiple pages as
 * different widget annotations).
 */
export declare class FormStore implements Disposable {
    #private;
    constructor(eventBus: EventBus, userId?: string);
    /**
     * Load form fields for a page from the worker result.
     *
     * Existing fields for the same page are replaced. Each loaded field
     * starts in the 'synced' commit state because it reflects the current
     * PDF state.
     */
    loadPage(documentId: string, pageIndex: number, fields: WidgetData[]): void;
    /**
     * Record a form field value change.
     *
     * Builds a new {@link WidgetData} with the updated `fieldValue`, records
     * it in the tracker and operation log, and emits a `'form:field-changed'`
     * event.
     */
    updateField(documentId: string, fieldName: string, value: string, pageIndex: number): void;
    /** Get all form fields for a document. */
    getForDocument(documentId: string): TrackedObject<WidgetData>[];
    /** Get form fields for a specific page. */
    getForPage(documentId: string, pageIndex: number): TrackedObject<WidgetData>[];
    /** Get all changed fields for a document. */
    getDirty(documentId: string): TrackedObject<WidgetData>[];
    /** Check if a document's form has unsaved changes. */
    hasDirty(documentId: string): ReadonlySignal<boolean>;
    /** Mark a single field as synced. */
    markSynced(documentId: string, fieldName: string): void;
    /** Convenience alias for `updateField` — matches the plugin API naming. */
    setFieldValue(documentId: string, pageIndex: number, fieldName: string, value: string): void;
    /** Get the current value of a field by name. Returns undefined if not loaded. */
    getFieldValue(documentId: string, fieldName: string): string | undefined;
    /** Extract all form data as a flat record (fieldName → fieldValue). */
    extractFormData(documentId: string): Record<string, string>;
    /** Mark all fields as synced for a document. */
    markAllSynced(documentId: string): void;
    /** Subscribe to form events from the operation log. */
    subscribe(fn: (event: DataEvent<WidgetData>) => void): Unsubscribe;
    /** Get the operation log (for sync/replay). */
    get log(): OperationLog<WidgetData>;
    /** Clean up all state for a document. */
    clearDocument(documentId: string): void;
    /** Dispose: clean up all documents. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=form-store.d.ts.map