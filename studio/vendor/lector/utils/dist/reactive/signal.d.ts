/** Subscription callback. */
export type Subscriber<T> = (value: T) => void;
/** Unsubscribe function. */
export type Unsubscribe = () => void;
/** Readable reactive value. */
export interface ReadonlySignal<T> {
    /** Current value. Reading inside an effect auto-tracks this signal. */
    readonly value: T;
    /** Subscribe to value changes. Returns unsubscribe function. */
    subscribe(fn: Subscriber<T>): Unsubscribe;
    /** Read value without tracking (won't register in effects). */
    peek(): T;
}
/** Writable reactive value. */
export interface Signal<T> extends ReadonlySignal<T> {
    value: T;
    /** Update value via a function of the current value. */
    update(fn: (current: T) => T): void;
}
/** @internal Get the currently running effect (for dependency tracking). */
export declare function _getActiveEffect(): (() => void) | null;
/** @internal Set the currently running effect. */
export declare function _setActiveEffect(fn: (() => void) | null): void;
/** Create a writable reactive signal. */
export declare function signal<T>(initial: T): Signal<T>;
/** Create a read-only computed signal derived from other signals. */
export declare function computed<T>(fn: () => T): ReadonlySignal<T>;
/**
 * Run a function that auto-tracks signal dependencies. Re-runs whenever
 * any tracked signal changes. Returns a dispose function to stop the effect.
 */
export declare function effect(fn: () => void | (() => void)): Unsubscribe;
/**
 * Batch multiple signal updates. Subscribers are notified once at the end.
 */
export declare function batch(fn: () => void): void;
//# sourceMappingURL=signal.d.ts.map