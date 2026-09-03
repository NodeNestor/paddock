/* tslint:disable */
/* eslint-disable */

/**
 * In-process graph database. Single-graph, single-tab. All
 * methods are synchronous from Rust's perspective; the JS wrapper
 * presents them as `async` so the host page can switch in a Web
 * Worker without changing the API.
 */
export class TraverseDb {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Returns the algorithm catalog — same shape as the
     * `/api/algorithms` HTTP endpoint. Studio's GDS drawer uses
     * this to populate its picker.
     */
    algorithms(): any;
    /**
     * Reset the database to an empty graph. The previous graph is
     * dropped, including any indexes and properties.
     */
    clear(): void;
    /**
     * Returns the current edge count.
     */
    edgeCount(): number;
    /**
     * Approximate in-memory size of the loaded graph in bytes —
     * node + edge property + index storage, not the JS heap or
     * the WASM linear-memory total. Matches the native server's
     * per-database `memory_bytes` so Studio's header readout has
     * meaningful numbers in WASM mode.
     */
    estimatedMemory(): number;
    exportTvdb(): Uint8Array;
    /**
     * Replace the current graph with one deserialized from a
     * `.tvdb` byte buffer produced by `exportTvdb` or by the
     * native Traverse server. The existing graph is dropped.
     */
    loadTvdb(bytes: Uint8Array): void;
    /**
     * Construct an empty graph. Populate via `query()` with
     * `CREATE` statements or via the (future) `loadTvdb()` API.
     */
    constructor();
    /**
     * Returns the current node count.
     */
    nodeCount(): number;
    /**
     * Run a Cypher query (read or write). `params_json` is an
     * optional JSON-stringified parameter map; pass `null` if no
     * parameters. `timeout_ms` (optional, > 0) installs a thread-
     * local deadline the executor honors at iteration boundaries;
     * pass 0 / null for no timeout.
     *
     * Returns a JS object matching the HTTP `QueryResponse` shape
     * — columns, rows, hydrated nodes/edges, mutation stats,
     * query type classification, elapsed time.
     *
     * Examples (from JS):
     * ```js
     * await db.query("MATCH (n) RETURN n", null, 30000)
     * await db.query(
     *   "CALL traverse.pageRank.stream({maxIterations: 20}) " +
     *   "YIELD nodeId, score RETURN nodeId, score",
     *   null,
     *   null
     * )
     * ```
     */
    query(cypher_text: string, params_json: any, timeout_ms?: number | null, dialect?: string | null): any;
    /**
     * Schema summary — labels, relationship types, property keys,
     * per-label / per-edge-type property maps, indexes, and
     * constraints. Same shape as the HTTP `/api/schema` response
     * so Studio's SchemaBrowser renders WASM-mode databases
     * identically.
     */
    schema(): any;
    /**
     * Serialize the entire graph as a `.tvdb` byte buffer —
     * **the same binary format the native server writes to
     * disk**. Identical paged layout, identical record encoding,
     * identical compression. A tvdb file written here loads
     * cleanly in the native server via `Database::open` and vice
     * versa.
     *
     * The returned bytes are ready to be persisted (typically to
     * OPFS by the Worker) and round-tripped via `loadTvdb`.
     * Split a script into individual statements, the way the server's
     * import endpoint and the CLI do.
     *
     * `query` takes exactly one statement, so a caller with a pasted
     * script has to cut it up first. Doing that in JS would be a third
     * implementation of a thing that is easy to get subtly wrong: a
     * bare split on `;` breaks `{name: 'Smith; Jones'}` in half. This
     * hands out the same splitter the other two use.
     *
     * Semicolons inside string literals and line comments are not
     * separators; empty statements are dropped; a trailing statement
     * with no semicolon is kept.
     */
    static splitStatements(script: string): string[];
    /**
     * Library version (matches the workspace Cargo version).
     */
    static version(): string;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_traversedb_free: (a: number, b: number) => void;
    readonly traversedb_algorithms: (a: number, b: number) => void;
    readonly traversedb_clear: (a: number, b: number) => void;
    readonly traversedb_edgeCount: (a: number, b: number) => void;
    readonly traversedb_estimatedMemory: (a: number, b: number) => void;
    readonly traversedb_exportTvdb: (a: number, b: number) => void;
    readonly traversedb_loadTvdb: (a: number, b: number, c: number, d: number) => void;
    readonly traversedb_new: () => number;
    readonly traversedb_nodeCount: (a: number, b: number) => void;
    readonly traversedb_query: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
    readonly traversedb_schema: (a: number, b: number) => void;
    readonly traversedb_splitStatements: (a: number, b: number, c: number) => void;
    readonly traversedb_version: (a: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
