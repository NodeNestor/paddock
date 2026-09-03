import type { ReadonlySignal, Unsubscribe } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { WidgetData, DataEvent, TrackedObject } from '../data/types.js';
import { FormStore } from '../data/form-store.js';
/**
 * Capability provided by the form layer plugin.
 *
 * Manages interactive PDF form fields: reads field data from the worker,
 * supports programmatic value changes, tracks dirty state, and provides
 * tab navigation between fields.
 */
export interface FormCapability {
    /** The form store instance. Consumers subscribe to field change events here. */
    readonly store: FormStore;
    /**
     * Load form fields for a page from the worker into the store.
     * Typically called when a page becomes visible.
     */
    loadPage(docId: DocumentId, pageIndex: number): Promise<void>;
    /** Get all form fields on a page (already loaded). */
    getPageFields(docId: DocumentId, pageIndex: number): TrackedObject<WidgetData>[];
    /** Get all form fields across the entire document (loaded pages only). */
    getDocumentFields(docId: DocumentId): TrackedObject<WidgetData>[];
    /**
     * Set a form field value.
     * Updates in both the worker (pdfium) and the local store.
     */
    setFieldValue(docId: DocumentId, pageIndex: number, fieldName: string, value: string): Promise<void>;
    /**
     * Get the value of a specific form field.
     * Returns undefined if the field is not loaded.
     */
    getFieldValue(docId: DocumentId, fieldName: string): string | undefined;
    /**
     * Populate multiple form fields at once (batch operation).
     * Each entry maps field name to value.
     */
    populateFields(docId: DocumentId, fields: ReadonlyArray<{
        pageIndex: number;
        fieldName: string;
        value: string;
    }>): Promise<void>;
    /**
     * Extract all form data as a flat record.
     * Keys are field names, values are field values.
     */
    extractFormData(docId: DocumentId): Record<string, string>;
    /** Whether the document has unsaved form changes. */
    hasDirty(docId: DocumentId): ReadonlySignal<boolean>;
    /** Mark all form fields as synced for a document. */
    markAllSynced(docId: DocumentId): void;
    /** Subscribe to form field change events. */
    subscribe(fn: (event: DataEvent<WidgetData>) => void): Unsubscribe;
    /** Whether the form layer is read-only. */
    readOnly: ReadonlySignal<boolean>;
    /** Set whether forms are read-only. */
    setReadOnly(readOnly: boolean): void;
    /** Currently focused field name, or null. */
    focusedField: ReadonlySignal<string | null>;
    /** Focus a field by name (for tab navigation). */
    focusField(fieldName: string | null): void;
    /** Set of page indices that have been loaded. */
    isPageLoaded(docId: DocumentId, pageIndex: number): boolean;
    /**
     * Simulate a click on a checkbox or radio button widget.
     * Returns updated form field data for the page after the click.
     */
    clickWidget(docId: DocumentId, pageIndex: number, pageX: number, pageY: number): Promise<WidgetData[]>;
}
/**
 * Form layer plugin.
 *
 * Bridges the pdfium worker (which reads/writes form field values in the PDF)
 * with the FormStore (which provides event sourcing and dirty tracking on
 * the main thread). Supports batch population for pre-filling forms.
 */
export declare const formPlugin: import("../index.js").PluginDefinition<FormCapability, Record<string, never>>;
//# sourceMappingURL=form-plugin.d.ts.map