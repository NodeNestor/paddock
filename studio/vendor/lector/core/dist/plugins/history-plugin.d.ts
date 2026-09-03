import type { ReadonlySignal } from '@truespar/lector-utils';
import type { DocumentId } from '../types/handle-id.js';
/** A single undoable/redoable operation. */
export interface HistoryEntry {
    /** Unique ID for this operation. */
    readonly id: string;
    /** Human-readable label for the UI (e.g., "Move annotation"). */
    readonly label: string;
    /** Topic for grouping related operations (e.g., "annotation", "form"). */
    readonly topic: string;
    /** Timestamp when the operation was recorded. */
    readonly timestamp: number;
    /** Execute the operation (apply the change). */
    execute(): void | Promise<void>;
    /** Reverse the operation (undo the change). */
    undo(): void | Promise<void>;
}
/**
 * Capability provided by the history plugin.
 *
 * Provides undo/redo stacks per document with topic-based grouping
 * and batch operations.
 */
export interface HistoryCapability {
    /**
     * Push an operation onto the undo stack.
     *
     * The operation's `execute()` is NOT called — it is assumed the caller
     * has already applied the change. This is the "deferred commit" pattern:
     * the caller does the work, then records it for undo support.
     */
    push(docId: DocumentId, entry: HistoryEntry): void;
    /** Undo the most recent operation for a document. */
    undo(docId: DocumentId): Promise<void>;
    /** Redo the most recently undone operation. */
    redo(docId: DocumentId): Promise<void>;
    /** Whether undo is available for the active document. */
    canUndo: ReadonlySignal<boolean>;
    /** Whether redo is available for the active document. */
    canRedo: ReadonlySignal<boolean>;
    /** The label of the next undoable operation, or null. */
    undoLabel: ReadonlySignal<string | null>;
    /** The label of the next redoable operation, or null. */
    redoLabel: ReadonlySignal<string | null>;
    /**
     * Start a batch operation. All entries pushed between `beginBatch`
     * and `endBatch` are grouped as a single undo/redo step.
     */
    beginBatch(docId: DocumentId, label: string): void;
    /** End the current batch and push the grouped operation. */
    endBatch(docId: DocumentId): void;
    /** Clear all history for a document. */
    clear(docId: DocumentId): void;
    /** Get the undo stack size for a document. */
    undoSize(docId: DocumentId): number;
    /** Get the redo stack size for a document. */
    redoSize(docId: DocumentId): number;
}
/**
 * History (undo/redo) plugin.
 *
 * Provides per-document undo/redo stacks. Operations are recorded by other
 * plugins (annotations, forms, etc.) using the `push()` method. Batch
 * operations group multiple entries into a single undo step.
 *
 * The history is topic-aware, so the UI can display context like
 * "Undo: Move annotation" or "Undo: Change form field".
 */
export declare const historyPlugin: import("../index.js").PluginDefinition<HistoryCapability, Record<string, never>>;
//# sourceMappingURL=history-plugin.d.ts.map