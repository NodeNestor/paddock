// Public types for @truespar/traverse-wasm.
//
// The JS implementation in `index.js` is JSDoc-typed; this `.d.ts`
// re-states the surface in pure TypeScript for downstream consumers
// that don't run TypeScript's checkJs on third-party code.

export interface QueryStats {
  nodes_created: number
  nodes_deleted: number
  relationships_created: number
  relationships_deleted: number
  properties_set: number
  labels_added: number
  labels_removed: number
}

export interface QueryStats {
  nodes_created: number
  nodes_deleted: number
  relationships_created: number
  relationships_deleted: number
  properties_set: number
  labels_added: number
  labels_removed: number
  indexes_added: number
  indexes_removed: number
  constraints_added: number
  constraints_removed: number
}

export interface QueryResponse {
  columns: string[]
  rows: unknown[][]
  total_rows: number
  truncated: boolean
  nodes: unknown[]
  edges: unknown[]
  entities_truncated: boolean
  /** Entities the query's patterns BOUND, returned or not — the provenance
   *  behind a projection like `RETURN t.amount`, hydrated to the same shape
   *  as `nodes`/`edges`. A graph view highlights these instead of guessing
   *  from the rows. Superset of the returned entities; dedupe by id. */
  bound_nodes: unknown[]
  bound_edges: unknown[]
  /** Read, Write, ReadWrite, or Schema — surfaced from the
   *  Cypher executor's classification. */
  query_type: 'Read' | 'Write' | 'ReadWrite' | 'Schema'
  stats: QueryStats
  time_ms: number
}

export type GdsCategory =
  | 'centrality'
  | 'community'
  | 'paths'
  | 'dag'
  | 'similarity'
  | 'linkPrediction'
  | 'embeddings'

export type GdsOutputKind =
  | 'nodeScalar'
  | 'nodeCommunity'
  | 'nodeMulti'
  | 'nodeVector'
  | 'path'
  | 'nodePair'
  | 'entitySet'
  | 'statsOnly'

export type GdsStability = 'stable' | 'beta' | 'experimental'

export type GdsConfigParamType =
  | 'bool'
  | 'integer'
  | 'float'
  | 'string'
  | 'enum'
  | 'stringList'

export interface GdsConfigParam {
  key: string
  type: GdsConfigParamType
  default: unknown
  description: string
  advanced: boolean
}

export interface GdsAlgorithm {
  name: string
  displayName: string
  category: GdsCategory
  outputKind: GdsOutputKind
  stability: GdsStability
  description: string
  writable: boolean
  modes: ('stream' | 'stats' | 'write')[]
  configSchema: GdsConfigParam[]
  outputColumns: string[]
  statsColumns: string[]
  writeColumns: string[]
}

export interface AlgorithmsResponse {
  algorithms: GdsAlgorithm[]
}

export interface PropertyKeyInfo {
  name: string
  type: string
}

export interface OwnerProperties {
  name: string
  properties: PropertyKeyInfo[]
}

export interface SchemaResponse {
  labels: string[]
  relationship_types: string[]
  property_keys: PropertyKeyInfo[]
  indexes: unknown[]
  constraints: unknown[]
  labels_detail: OwnerProperties[]
  relationship_types_detail: OwnerProperties[]
}

export interface ImportResult {
  total_statements: number
  executed: number
  errors: { statement: string; error: string }[]
  stats: QueryStats
  time_ms: number
}

export interface TraverseDbOpenOptions {
  /** Thread pool size for wasm-bindgen-rayon. Default:
   * `Math.min(navigator.hardwareConcurrency, 16)`. */
  numThreads?: number
  /** Override the worker script URL. Most users don't need this. */
  workerUrl?: URL | string
  /** OPFS database name to restore on open. If absent, opens empty.
   *  If present but the database doesn't exist yet, opens empty and
   *  the caller can populate then `commit(name)`. */
  name?: string
}

export interface DatabaseListEntry {
  name: string
  size: number
  lastModified: number
}

export interface CallOptions {
  /** Abort signal — propagates to the worker's cancellation token. */
  signal?: AbortSignal
}

export interface QueryOptions extends CallOptions {
  /** Per-query timeout in milliseconds. Installs a thread-local
   *  deadline the executor checks at iteration boundaries. Pair with
   *  `signal` for end-to-end cancellation (deadline aborts the
   *  worker side; the signal rejects the promise on the main thread). */
  timeoutMs?: number
  /** Query dialect. `'gql'` routes through the ISO/IEC 39075 parser
   *  (lowered onto the same engine); default is openCypher. */
  dialect?: 'cypher' | 'gql'
}

/**
 * In-tab Traverse graph engine, hosted in a dedicated Web Worker so
 * the main thread never blocks on Cypher execution.
 */
export class TraverseDb {
  /** Library version reported by the worker post-init. */
  readonly version: string | null

  /** Open a new database. Spawns a fresh worker and initializes the
   *  WASM module + thread pool. If `options.name` is set, restores
   *  the OPFS-resident database with that name. */
  static open(options?: TraverseDbOpenOptions): Promise<TraverseDb>

  /** List OPFS-resident databases without opening a TraverseDb.
   *  Lightweight — spawns a short-lived worker. */
  static listDatabases(options?: { workerUrl?: URL | string }): Promise<DatabaseListEntry[]>

  /** Delete an OPFS-resident database without opening it. No-op if it
   *  doesn't exist. */
  static deleteDatabase(name: string, options?: { workerUrl?: URL | string }): Promise<void>

  /** Run a Cypher query. Returns the standard QueryResponse shape. */
  query(
    cypher: string,
    params?: Record<string, unknown> | null,
    options?: QueryOptions,
  ): Promise<QueryResponse>

  /** Returns the algorithm catalog metadata. Same shape as the HTTP
   *  `/api/algorithms` endpoint. */
  algorithms(): Promise<AlgorithmsResponse>

  /** Schema summary — labels, relationship types, property keys,
   *  per-label / per-edge-type property listings. Same shape as the
   *  HTTP `/api/schema` response. Indexes and constraints are emitted
   *  as empty arrays in v1 — the WASM build does not yet surface
   *  those registries even though the Cypher executor uses them. */
  schema(): Promise<SchemaResponse>

  /** Execute a Cypher import (multi-statement OK). Returns the same
   *  shape as the HTTP `/api/import` endpoint — total / executed /
   *  errors / stats / time_ms. Use this rather than `query()` when
   *  Studio's import panels are wired up. */
  importCypher(cypher: string): Promise<ImportResult>

  /** Split a script into individual statements, the way the server's import
   *  endpoint and the CLI do. Semicolons inside string literals and line
   *  comments are not separators, so `{name: 'Smith; Jones'}` survives.
   *
   *  `query()` takes exactly one statement; this is how a caller holding a
   *  pasted script cuts it up without writing a fourth splitter. */
  static splitStatements(script: string): string[]

  /** Reset to an empty graph. Indexes and properties are dropped. */
  clear(): Promise<void>

  /** Current node count in the graph. */
  nodeCount(): Promise<number>

  /** Current edge count in the graph. */
  edgeCount(): Promise<number>

  /** Current node + edge counts as a single round-trip. */
  stats(): Promise<{ nodes: number; edges: number }>

  /** Approximate in-memory size of the graph in bytes (node + edge +
   *  property + index storage). Mirrors the native server's per-DB
   *  `memory_bytes` field. Does not include the WASM linear-memory
   *  total or the JS heap. */
  estimatedMemory(): Promise<number>

  /** Persist the current graph to OPFS under `name`. */
  commit(name: string): Promise<{ bytes: number }>

  /** Write a `.tvdb` byte buffer straight to OPFS without loading it
   *  into the engine. Pair with `load(name)` to parse it. Lets large
   *  uploads show distinct "uploading" / "loading" phases. */
  writeBytes(name: string, bytes: Uint8Array): Promise<{ ok: true; bytes: number }>

  /** Restore the graph from OPFS under `name`. Returns `{ ok: false,
   *  missing: true }` if no such database exists. */
  load(name: string): Promise<{ ok: boolean; missing?: boolean; nodes?: number; edges?: number }>

  /** List OPFS-resident databases for this origin. */
  listDatabases(): Promise<DatabaseListEntry[]>

  /** Delete a database from OPFS. */
  deleteDatabase(name: string): Promise<void>

  /** Export the current graph as a `.tvdb` binary buffer — same
   *  format the native server uses on disk. Round-trips with
   *  `Database::open` on native. Use for "Download .tvdb" or
   *  cross-origin transfer; for same-origin persistence call
   *  `commit()` (which uses OPFS internally). */
  exportTvdb(): Promise<Uint8Array>

  /** Replace the current graph with one deserialized from a `.tvdb`
   *  binary buffer. */
  loadTvdb(bytes: Uint8Array): Promise<{ ok: boolean; nodes?: number; edges?: number }>

  /** Tear down the worker. Pending requests reject with AbortError. */
  close(): void
}

/** Cheap accessor for the library version. Returns null until at least
 *  one TraverseDb has been opened in this realm. */
export function version(): string | null
