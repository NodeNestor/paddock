/**
 * Generic Disposable registry mapping opaque branded string IDs to real handles.
 *
 * Used to give the main thread stable, serializable identifiers for objects
 * that live in the worker's WASM memory (documents, pages, bitmaps, etc.)
 * without ever exposing raw pointers across the boundary.
 */
export declare class HandleRegistry<TId extends string, THandle> implements Disposable {
    #private;
    constructor(prefix: string, disposer: (handle: THandle) => void);
    /** Register a handle and return a new opaque ID. */
    register(handle: THandle): TId;
    /** Resolve an ID to its handle. Throws if not found. */
    resolve(id: TId): THandle;
    /** Remove a handle by ID and return it. Throws if not found. */
    release(id: TId): THandle;
    /** Replace the handle for an existing ID (does NOT call the disposer). */
    replace(id: TId, handle: THandle): void;
    /** Check whether an ID is registered. */
    has(id: TId): boolean;
    /** Number of currently registered handles. */
    get size(): number;
    /** Dispose all remaining handles via the configured disposer. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=handle-registry.d.ts.map