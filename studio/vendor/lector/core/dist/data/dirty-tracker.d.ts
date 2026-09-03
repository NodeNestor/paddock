import type { ReadonlySignal } from '@truespar/lector-utils';
import type { CommitState, TrackedObject } from './types.js';
/**
 * Per-object commit state management.
 *
 * Tracks objects by ID and maintains a reactive {@link CommitState} signal
 * for each one. The aggregate `hasDirty` signal reactively reflects whether
 * any tracked object has unsaved changes.
 */
export declare class DirtyTracker<T> implements Disposable {
    #private;
    /** Reactive signal that is `true` when at least one object is dirty. */
    readonly hasDirty: ReadonlySignal<boolean>;
    /** Track a new object. */
    add(id: string, data: T, state?: CommitState): TrackedObject<T>;
    /** Update a tracked object's data. Sets commitState to 'dirty' if currently 'synced'. */
    update(id: string, data: T): void;
    /** Mark an object as deleted. */
    delete(id: string): void;
    /** Mark an object as synced (consumer has persisted it). */
    markSynced(id: string): void;
    /** Mark all tracked objects as synced. */
    markAllSynced(): void;
    /** Get a tracked object by ID. */
    get(id: string): TrackedObject<T> | undefined;
    /** Get all objects with a specific commit state. */
    getByState(state: CommitState): TrackedObject<T>[];
    /** Get all dirty objects (new + dirty + deleted). */
    getDirty(): TrackedObject<T>[];
    /** Remove an object completely (after deletion is confirmed by the server). */
    remove(id: string): void;
    /** Clear all tracked objects. */
    clear(): void;
    /** Dispose: clear all tracked objects. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=dirty-tracker.d.ts.map