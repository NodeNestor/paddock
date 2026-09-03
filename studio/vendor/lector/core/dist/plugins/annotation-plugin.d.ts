import type { ReadonlySignal, Unsubscribe } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
import type { LectorUser } from '../engine/lector-engine.js';
import type { AnnotationData, CommentStatus, DataEvent, TrackedObject } from '../data/types.js';
import { AnnotationStore } from '../data/annotation-store.js';
import { type AnnotationTool, type AnnotationStyleState } from './annotation-tools.js';
/** Lock modes for the annotation layer. */
export type AnnotationLockMode = 'none' | 'read-only' | 'no-create' | 'no-delete';
/**
 * Capability provided by the annotation layer plugin.
 *
 * Manages annotation lifecycle: load from worker, create/edit/delete with
 * deferred commit model, dirty tracking, and lock modes.
 */
export interface AnnotationCapability {
    /** The annotation store instance. Consumers subscribe to events here. */
    readonly store: AnnotationStore;
    /**
     * Load annotations for a page from the worker into the store.
     * Typically called when a page becomes visible.
     */
    loadPage(docId: DocumentId, pageIndex: number): Promise<void>;
    /**
     * Create a new annotation on a page.
     * Creates in both the worker (pdfium) and the local store.
     */
    create(docId: DocumentId, pageIndex: number, data: Partial<AnnotationData>): Promise<TrackedObject<AnnotationData>>;
    /**
     * Update an existing annotation.
     * Updates in both the worker and the local store.
     */
    update(docId: DocumentId, annotationId: string, patch: Partial<AnnotationData>): Promise<void>;
    /**
     * Delete an annotation.
     * Deletes from both the worker and the local store.
     */
    delete(docId: DocumentId, annotationId: string): Promise<void>;
    /** Get all annotations for a page from the store (already loaded). */
    getForPage(docId: DocumentId, pageIndex: number): TrackedObject<AnnotationData>[];
    /** Get all annotations for a document. */
    getForDocument(docId: DocumentId): TrackedObject<AnnotationData>[];
    /** Get all dirty (unsaved) annotations for a document. */
    getDirty(docId: DocumentId): TrackedObject<AnnotationData>[];
    /** Whether the document has unsaved annotation changes. */
    hasDirty(docId: DocumentId): ReadonlySignal<boolean>;
    /** Mark a single annotation as synced (saved by consumer). */
    markSynced(docId: DocumentId, annotationId: string): void;
    /** Mark all annotations for a document as synced. */
    markAllSynced(docId: DocumentId): void;
    /** Subscribe to annotation data events. */
    subscribe(fn: (event: DataEvent<AnnotationData>) => void): Unsubscribe;
    /** Current lock mode. */
    lockMode: ReadonlySignal<AnnotationLockMode>;
    /** Set the lock mode. */
    setLockMode(mode: AnnotationLockMode): void;
    /**
     * Currently primary-selected annotation ID, or null. The "primary"
     * is the most recently added member of the multi-selection set —
     * single-selection callers (the popover, the comments sidebar, drag
     * handles) keep working unchanged.
     */
    selectedAnnotation: ReadonlySignal<string | null>;
    /**
     * The full multi-selection set. Always contains `selectedAnnotation`
     * as its last entry when single-selected; can hold 2+ ids when the
     * user shift-clicks additional annotations to build a group selection.
     */
    selectedAnnotations: ReadonlySignal<readonly string[]>;
    /** Select an annotation (or null to deselect). Replaces the multi-selection. */
    selectAnnotation(annotationId: string | null): void;
    /**
     * Toggle an annotation's membership in the multi-selection. If the
     * annotation isn't selected, it's added; if it is, it's removed. Used
     * by shift-click on the canvas to build group selections.
     */
    toggleAnnotationSelection(annotationId: string): void;
    /** Clear the entire multi-selection. */
    clearAnnotationSelection(): void;
    /** Set of page indices that have been loaded into the store. */
    isPageLoaded(docId: DocumentId, pageIndex: number): boolean;
    /**
     * Force-reload annotations for a page from pdfium.
     * Clears existing store/index data for the page and re-reads from the worker.
     * Use after operations that externally mutate pdfium annotations (e.g. flatten).
     */
    reloadPage(docId: DocumentId, pageIndex: number): Promise<void>;
    /** Active annotation creation tool, or null if not in drawing mode. */
    activeTool: ReadonlySignal<AnnotationTool | null>;
    /** Activate an annotation creation tool (switches to draw mode). */
    setActiveTool(tool: AnnotationTool | null): void;
    /** Current annotation style (color, border width, etc.). */
    toolStyle: ReadonlySignal<AnnotationStyleState>;
    /** Update annotation style properties. */
    setToolStyle(patch: Partial<AnnotationStyleState>): void;
    /** Current user identity, if configured on the engine. */
    readonly user: LectorUser | undefined;
    /** Set the review status of an annotation's comment thread. */
    setCommentStatus(docId: DocumentId, annotationId: string, status: CommentStatus): Promise<void>;
    /** Toggle the resolved state of an annotation's comment thread. */
    toggleResolved(docId: DocumentId, annotationId: string): Promise<void>;
    /** Edit an existing comment's text. Only the comment author can edit. */
    editComment(docId: DocumentId, annotationId: string, commentId: string, newText: string): Promise<void>;
    /** Delete a comment from the thread. Only the comment author can delete. */
    deleteComment(docId: DocumentId, annotationId: string, commentId: string): Promise<void>;
    /** Mark an annotation and its comments as read by the current user. */
    markAsRead(docId: DocumentId, annotationId: string): void;
    /** Bring an annotation to the front (highest z-index on its page). */
    bringToFront(docId: DocumentId, annotationId: string): Promise<void>;
    /** Send an annotation to the back (lowest z-index on its page). */
    sendToBack(docId: DocumentId, annotationId: string): Promise<void>;
    /** Group annotations together. All get the same groupId. */
    groupAnnotations(docId: DocumentId, annotationIds: string[]): Promise<void>;
    /** Ungroup annotations (remove groupId). */
    ungroupAnnotations(docId: DocumentId, groupId: string): Promise<void>;
    /** Check if the current user can edit/delete an annotation. */
    canEdit(docId: DocumentId, annotationId: string): boolean;
    canDelete(docId: DocumentId, annotationId: string): boolean;
    /** Active stamp name for the stamp tool. */
    activeStampName: string;
    /**
     * Stage an image for the next click of the image tool. The viewer's
     * image-tool button calls this after the user picks a file and the
     * file is decoded into a base64 data URI; the next click on a page
     * places it. Pass null to clear the staged image.
     */
    setStagedImage(image: {
        dataUri: string;
        naturalWidth: number;
        naturalHeight: number;
    } | null): void;
    /** Read the currently staged image, if any. */
    getStagedImage(): {
        dataUri: string;
        naturalWidth: number;
        naturalHeight: number;
    } | null;
}
/**
 * Annotation layer plugin.
 *
 * Bridges the pdfium worker (which reads/writes annotations in the PDF)
 * with the AnnotationStore (which provides event sourcing and dirty tracking
 * on the main thread). Enforces lock modes for enterprise use cases.
 */
export declare const annotationPlugin: import("../index.js").PluginDefinition<AnnotationCapability, Record<string, never>>;
//# sourceMappingURL=annotation-plugin.d.ts.map