// @truespar/traverse-wasm — public entry point.
//
// Spawns a dedicated Web Worker that hosts the Traverse WASM engine,
// exposes a Promise-based API the host page can call from the main
// thread, and presents the same shape Studio's HttpApiClient does so
// the WasmApiClient adapter drops in transparently.
//
// Usage:
//   import { TraverseDb } from '@truespar/traverse-wasm'
//   const db = await TraverseDb.open({ numThreads: navigator.hardwareConcurrency })
//   const r = await db.query("CALL traverse.pageRank.stream({}) YIELD nodeId, score RETURN nodeId, score LIMIT 10")
//   console.log(r.rows)
//
// Cancellation:
//   const ctl = new AbortController()
//   setTimeout(() => ctl.abort('user cancelled'), 5000)
//   await db.query(cypher, params, { signal: ctl.signal })
// (Note: in-flight cancellation hits the Rust-side cancellation token
//  at iteration boundaries — long-running pure CPU loops between
//  checks may take a moment to notice.)

/**
 * Per-instance counter for matching requests with responses.
 * @private
 */
let _nextRequestId = 0

/**
 * In-tab Traverse graph engine, hosted in a dedicated Web Worker so the
 * main thread never blocks on Cypher execution.
 *
 * Construct via `TraverseDb.open()` — the constructor itself does no
 * async work, so call `open()` and `await` the returned Promise.
 */
export class TraverseDb {
  /** @type {Worker} */
  #worker
  /** @type {Map<number, { resolve: (v: any) => void, reject: (e: any) => void, signal?: AbortSignal, onAbort?: () => void }>} */
  #pending = new Map()
  /** @type {string | null} Library version reported by the worker post-init. */
  version = null

  /**
   * @param {Worker} worker
   * @private
   */
  constructor(worker) {
    this.#worker = worker
    worker.addEventListener('message', (ev) => this.#onMessage(ev))
  }

  /**
   * Spawn a fresh worker, initialize the WASM module + thread pool,
   * and optionally restore a previously-committed database from OPFS.
   *
   * @param {{
   *   numThreads?: number,
   *   workerUrl?: URL | string,
   *   name?: string,
   * }} [options]
   *   `numThreads` — passed to wasm-bindgen-rayon's initThreadPool.
   *     Defaults to `navigator.hardwareConcurrency` clamped to 16.
   *   `workerUrl` — override the worker script URL. Most users won't
   *     need this; the default resolves `./worker.js` via import.meta.
   *   `name` — OPFS database to restore. If absent, opens an empty
   *     graph. If present but no such database exists in OPFS, opens
   *     empty and the caller can populate then `commit(name)`.
   * @returns {Promise<TraverseDb>}
   */
  static async open(options = {}) {
    // Bundler-friendly: the literal `new Worker(new URL('./worker.js',
    // import.meta.url), …)` is what Vite / webpack / Parcel recognize
    // as a worker entry. The variable form (`options.workerUrl ?? …`)
    // hid the static pattern from those tools and they stopped
    // bundling the worker as a module — left it as a raw asset with
    // unresolved imports. So we keep the two paths explicit.
    const worker = options.workerUrl
      ? new Worker(options.workerUrl, { type: 'module' })
      : new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
    const db = new TraverseDb(worker)
    const numThreads = Math.min(
      options.numThreads ?? (globalThis.navigator?.hardwareConcurrency ?? 4),
      16,
    )
    const initResult = await db.#call('init', { numThreads })
    db.version = initResult?.version ?? null
    if (options.name) {
      await db.load(options.name).catch(() => undefined)
    }
    return db
  }

  /**
   * List OPFS-resident databases without opening a `TraverseDb`. Spawns
   * a short-lived worker; cheaper than opening a full database. Useful
   * for connection-panel UIs that show "previously saved" databases.
   * @returns {Promise<Array<{ name: string, size: number, lastModified: number }>>}
   */
  static async listDatabases(options = {}) {
    const worker = options.workerUrl
      ? new Worker(options.workerUrl, { type: 'module' })
      : new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
    const db = new TraverseDb(worker)
    try {
      await db.#call('init', { numThreads: 1 })
      return await db.listDatabases()
    } finally {
      db.close()
    }
  }

  /**
   * Delete an OPFS-resident database without opening it. Spawns a
   * short-lived worker. No-op if the database doesn't exist.
   */
  static async deleteDatabase(name, options = {}) {
    const worker = options.workerUrl
      ? new Worker(options.workerUrl, { type: 'module' })
      : new Worker(new URL('./worker.js', import.meta.url), { type: 'module' })
    const db = new TraverseDb(worker)
    try {
      await db.#call('init', { numThreads: 1 })
      await db.deleteDatabase(name)
    } finally {
      db.close()
    }
  }

  /**
   * Run a Cypher query. Returns the same `QueryResponse` shape as the
   * HTTP `/api/query` endpoint (columns + rows + timing + mutation
   * stats + query type + hydrated entities).
   *
   * @param {string} cypher
   * @param {Record<string, unknown> | null} [params]
   * @param {{ signal?: AbortSignal, timeoutMs?: number, dialect?: 'cypher' | 'gql' }} [options]
   *   `timeoutMs` — install a deadline that the Cypher executor
   *   checks at iteration boundaries. Pair with `signal` for a
   *   responsive cancel: the deadline aborts the worker side, the
   *   signal rejects the JS promise.
   *   `dialect` — `'gql'` routes through the ISO/IEC 39075 parser
   *   (lowered onto the same engine); default is openCypher.
   * @returns {Promise<QueryResponse>}
   */
  query(cypher, params = null, options = {}) {
    return this.#call(
      'query',
      {
        cypher,
        params,
        timeoutMs: options.timeoutMs ?? null,
        dialect: options.dialect ?? null,
      },
      options,
    )
  }

  /**
   * Returns the algorithm catalog metadata (same shape as
   * `/api/algorithms`). Used by Studio to populate the GDS drawer.
   *
   * @returns {Promise<AlgorithmsResponse>}
   */
  algorithms() {
    return this.#call('algorithms', {})
  }

  /**
   * Returns the schema summary — labels, relationship types, property
   * keys, per-label / per-type property maps. Same shape as the HTTP
   * `/api/schema` response.
   * @returns {Promise<SchemaResponse>}
   */
  schema() {
    return this.#call('schema', {})
  }

  /**
   * Execute a Cypher import (multi-statement OK). Returns an
   * `ImportResult` matching the HTTP `/api/import` response.
   * @param {string} cypher
   * @returns {Promise<ImportResult>}
   */
  importCypher(cypher) {
    return this.#call('import', { cypher })
  }

  /**
   * Reset to an empty graph. Indexes and properties are dropped.
   * @returns {Promise<void>}
   */
  async clear() {
    await this.#call('clear', {})
  }

  /** @returns {Promise<number>} */
  nodeCount() {
    return this.#call('nodeCount', {})
  }

  /** @returns {Promise<number>} */
  edgeCount() {
    return this.#call('edgeCount', {})
  }

  /** @returns {Promise<{ nodes: number, edges: number }>} */
  stats() {
    return this.#call('stats', {})
  }

  /** Approximate in-memory size of the graph in bytes (node + edge
   *  property + index storage). Matches the native server's per-DB
   *  `memory_bytes` field. Does *not* report the WASM linear-memory
   *  total or the JS heap — just the engine's accounting.
   *  @returns {Promise<number>} */
  estimatedMemory() {
    return this.#call('estimatedMemory', {})
  }

  // ── OPFS persistence ──────────────────────────────────────────

  /**
   * Persist the current graph to OPFS under `name`. Subsequent reloads
   * of the page can restore it via `TraverseDb.open({ name })`.
   *
   * @param {string} name
   * @returns {Promise<{ bytes: number }>}
   */
  commit(name) {
    return this.#call('commit', { name })
  }

  /**
   * Persist a caller-supplied `.tvdb` byte buffer to OPFS under
   * `name` WITHOUT loading it into the engine. Pair with `load(name)`
   * to actually parse it into the graph. Lets large-file uploads
   * progress through "uploading" and "loading into memory" as
   * distinct phases instead of one long opaque pause.
   *
   * @param {string} name
   * @param {Uint8Array} bytes
   * @returns {Promise<{ ok: true, bytes: number }>}
   */
  writeBytes(name, bytes) {
    return this.#call('writeBytes', { name, bytes })
  }

  /**
   * Restore the graph from OPFS under `name`. Replaces the current
   * graph state. Returns `{ ok: false, missing: true }` if no
   * database with that name exists.
   *
   * @param {string} name
   * @returns {Promise<{ ok: boolean, missing?: boolean, nodes?: number, edges?: number }>}
   */
  load(name) {
    return this.#call('open', { name })
  }

  /**
   * List databases currently stored in OPFS for this origin. Newest
   * first.
   * @returns {Promise<Array<{ name: string, size: number, lastModified: number }>>}
   */
  listDatabases() {
    return this.#call('listDatabases', {})
  }

  /**
   * Delete a database from OPFS. No-op if it doesn't exist.
   * @param {string} name
   * @returns {Promise<void>}
   */
  async deleteDatabase(name) {
    await this.#call('deleteDatabase', { name })
  }

  /**
   * Export the current graph as a `.tvdb` binary buffer — the same
   * format the native Traverse server uses on disk. Useful for
   * "Download .tvdb" buttons or shipping via the Clipboard API. To
   * persist within the origin, use `commit()` instead — that goes to
   * OPFS without a user file-dialog.
   *
   * A tvdb file exported here loads cleanly in the native server,
   * and vice versa.
   * @returns {Promise<Uint8Array>}
   */
  async exportTvdb() {
    const { bytes } = await this.#call('exportTvdb', {})
    return bytes
  }

  /**
   * Replace the current graph with one deserialized from a `.tvdb`
   * binary buffer (typically produced by `exportTvdb` or by the
   * native Traverse server). Pair with `exportTvdb` for cross-tab /
   * cross-origin transfer via files.
   * @param {Uint8Array} bytes
   */
  async loadTvdb(bytes) {
    return this.#call('loadTvdb', { bytes })
  }

  /**
   * Tear down the worker. Pending requests reject with `AbortError`.
   */
  close() {
    for (const [, entry] of this.#pending) {
      entry.reject(new DOMException('Database closed', 'AbortError'))
    }
    this.#pending.clear()
    this.#worker.terminate()
  }

  // ── Internal ────────────────────────────────────────────────────

  /**
   * @template T
   * @param {string} kind
   * @param {object} payload
   * @param {{ signal?: AbortSignal, transfer?: Transferable[] }} [opts]
   * @returns {Promise<T>}
   * @private
   */
  #call(kind, payload, opts = {}) {
    const id = ++_nextRequestId
    return new Promise((resolve, reject) => {
      const signal = opts.signal
      let onAbort
      if (signal) {
        if (signal.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }
        onAbort = () => {
          // Best-effort: tell the worker to flip its cancellation
          // token; the resolve/reject path is local.
          this.#worker.postMessage({ id: -1, kind: 'abort', payload: { requestId: id } })
          const entry = this.#pending.get(id)
          if (entry) {
            this.#pending.delete(id)
            entry.reject(new DOMException('Aborted', 'AbortError'))
          }
        }
        signal.addEventListener('abort', onAbort, { once: true })
      }
      this.#pending.set(id, { resolve, reject, signal, onAbort })
      const msg = { id, kind, payload }
      if (opts.transfer && opts.transfer.length > 0) {
        this.#worker.postMessage(msg, opts.transfer)
      } else {
        this.#worker.postMessage(msg)
      }
    })
  }

  #onMessage(ev) {
    const { id, ok, result, error } = ev.data ?? {}
    const entry = this.#pending.get(id)
    if (!entry) return
    this.#pending.delete(id)
    if (entry.signal && entry.onAbort) {
      entry.signal.removeEventListener('abort', entry.onAbort)
    }
    if (ok) {
      entry.resolve(result)
    } else {
      const err = new Error(error?.message ?? 'unknown WASM error')
      // @ts-ignore — attach the structured error info as non-standard
      // properties so callers can inspect `err.code` if they want.
      err.code = error?.code
      err.stack = error?.stack ?? err.stack
      entry.reject(err)
    }
  }
}

/**
 * Cheap accessor for the library version (matches the workspace Cargo
 * version). Returns null until `open()` resolves.
 */
export function version() {
  return TraverseDb.prototype.version
}

// ── Type re-exports for JSDoc consumers ───────────────────────────

/**
 * @typedef {object} QueryResponse
 * @property {string[]} columns
 * @property {unknown[][]} rows
 * @property {number} total_rows
 * @property {boolean} truncated
 * @property {'Read' | 'Write'} query_type
 * @property {number} time_ms
 */

/**
 * @typedef {object} AlgorithmsResponse
 * @property {GdsAlgorithm[]} algorithms
 */

/**
 * @typedef {object} GdsAlgorithm
 * @property {string} name
 * @property {string} displayName
 * @property {string} category
 * @property {string} outputKind
 * @property {string} stability
 * @property {string} description
 * @property {boolean} writable
 * @property {string[]} modes
 * @property {string[]} outputColumns
 * @property {string[]} statsColumns
 * @property {string[]} writeColumns
 */
