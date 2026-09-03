import type { Unsubscribe } from '@truespar/lector-utils';
import type { DataEvent } from './types.js';
/**
 * Immutable, append-only event log.
 *
 * Every data mutation (annotation create/update/delete, form field change)
 * is recorded as a {@link DataEvent}. The log is subscribable so consumers
 * can react to events in real time, and replayable so the full history can
 * be sent to a server for persistence or conflict resolution.
 */
export declare class OperationLog<T> implements Disposable {
    #private;
    /** Append an event to the log. Notifies all subscribers synchronously. */
    append(event: DataEvent<T>): void;
    /** Subscribe to new events. Returns an unsubscribe function. */
    subscribe(fn: (event: DataEvent<T>) => void): Unsubscribe;
    /** Get all events (shallow copy -- event objects themselves are readonly). */
    getAll(): ReadonlyArray<DataEvent<T>>;
    /** Get events for a specific document. */
    getForDocument(documentId: string): ReadonlyArray<DataEvent<T>>;
    /**
     * Get events appended after the event with the given operation ID.
     *
     * If the operation ID is not found the full log is returned, which is the
     * safe default for an initial sync.
     */
    getSince(operationId: string): ReadonlyArray<DataEvent<T>>;
    /** Get the total number of events in the log. */
    get size(): number;
    /** Remove all events for a document (typically when the document is closed). */
    clearDocument(documentId: string): void;
    /** Dispose: clear all events and subscribers. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=operation-log.d.ts.map