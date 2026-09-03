import type { TaskId } from '../types/handle-id.js';
/**
 * Maps TaskId to AbortSignal listeners.
 *
 * When a task is tracked, an 'abort' listener is attached to its signal.
 * When the task completes (or is cancelled), call `untrack` to remove the listener.
 * Disposing the tracker removes all outstanding listeners.
 */
export declare class AbortTracker implements Disposable {
    #private;
    /**
     * Register an abort listener for a task.
     *
     * If the signal is already aborted, `onAbort` is called synchronously
     * and the task is not stored.
     */
    track(taskId: TaskId, signal: AbortSignal, onAbort: () => void): void;
    /** Remove the abort listener for a task. No-op if the task is not tracked. */
    untrack(taskId: TaskId): void;
    /** Remove all abort listeners. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=abort-tracker.d.ts.map