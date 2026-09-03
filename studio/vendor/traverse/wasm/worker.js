// Web Worker that hosts the WASM module. The main thread never calls
// the WASM directly — every operation funnels through this worker via
// postMessage so the host page's UI stays responsive even when an
// algorithm takes seconds to run.
//
// Wire protocol (see `index.js` for the matching client side):
//   request:  { id: number, kind: string, payload: object, transfer?: Transferable[] }
//   response: { id: number, ok: true, result: unknown }
//           | { id: number, ok: false, error: { code, message, stack? } }
//
// All requests are serialised — only one WASM call is in flight at a
// time per worker. Studio's gds-store + the WasmApiClient layer above
// observe responses in order.

// Use namespace import so the single-threaded build (which doesn't
// export `initThreadPool`) still bundles cleanly with strict module
// resolvers like Vite/rolldown. The threaded build is detected at
// runtime via `wasm.initThreadPool`.
import * as wasm from './pkg/traverse_wasm.js'
const init = wasm.default
const TraverseDb = wasm.TraverseDb
// Looked up at runtime on purpose: a static `wasm.initThreadPool` (or a
// literal-keyed `wasm['initThreadPool']`) is reported by rolldown as an
// import that is always undefined on the single-threaded build.
const initThreadPool = Reflect.get(wasm, 'initThreadPool')

/** @type {TraverseDb | null} */
let db = null

/** Initialization promise — resolved once `init` + `initThreadPool`
 *  have completed for this worker. Subsequent requests await it. */
let readyPromise = null

async function bootstrap(numThreads) {
  await init()
  if (typeof initThreadPool === 'function') {
    // Threaded build: spin up the rayon pool. JS provides the worker
    // count; default to navigator.hardwareConcurrency in the client.
    await initThreadPool(numThreads ?? 4)
  }
  db = new TraverseDb()
}

function ensureReady(numThreads) {
  if (!readyPromise) readyPromise = bootstrap(numThreads)
  return readyPromise
}

self.addEventListener('message', async (ev) => {
  const { id, kind, payload } = ev.data ?? {}
  try {
    const result = await handle(kind, payload)
    self.postMessage({ id, ok: true, result })
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err)
    const stack = err instanceof Error ? err.stack : undefined
    self.postMessage({
      id,
      ok: false,
      error: { code: 'WasmError', message: msg, stack },
    })
  }
})

/** Dispatch a single request kind. Returns a structured-cloneable
 *  value that the main thread can hand off to its caller. */
async function handle(kind, payload) {
  switch (kind) {
    case 'init':
      await ensureReady(payload?.numThreads)
      // `TraverseDb.version` is a static method (returns a string),
      // not a getter — must invoke it. Posting the bare function back
      // to the main thread crashed on structured-clone with a
      // misleading "could not be cloned" error.
      return { ready: true, version: TraverseDb.version() }

    case 'query': {
      await ensureReady()
      // `timeoutMs` (optional, > 0) installs the executor's
      // thread-local deadline before running. Pair with the JS-side
      // AbortController so a runaway algorithm can be cancelled.
      const timeoutMs = typeof payload.timeoutMs === 'number' ? payload.timeoutMs : null
      // `dialect` (optional) — 'gql' routes through the ISO/IEC 39075
      // parser; anything else (or absent) is openCypher.
      const dialect = typeof payload.dialect === 'string' ? payload.dialect : undefined
      return db.query(payload.cypher, payload.params ?? null, timeoutMs, dialect)
    }

    case 'algorithms': {
      await ensureReady()
      return db.algorithms()
    }

    case 'clear': {
      await ensureReady()
      db.clear()
      return { ok: true }
    }

    case 'nodeCount': {
      await ensureReady()
      return db.nodeCount()
    }

    case 'edgeCount': {
      await ensureReady()
      return db.edgeCount()
    }

    case 'stats': {
      await ensureReady()
      return { nodes: db.nodeCount(), edges: db.edgeCount() }
    }

    case 'estimatedMemory': {
      await ensureReady()
      return db.estimatedMemory()
    }

    case 'schema': {
      await ensureReady()
      return db.schema()
    }

    case 'import': {
      // Cypher import: split the script, execute each statement, then
      // synthesize an ImportResult (matches the HTTP /api/import response
      // shape, including its statement loop).
      //
      // This used to pass the whole script to `query` on the belief that the
      // executor handled multi-statement strings the way the native server
      // does. It does not, and neither does the server: the server's import
      // endpoint SPLITS first, and `query` takes exactly one statement, so a
      // pasted script died at its first semicolon while the same script
      // imported fine against a real server. Splitting is done in Rust so
      // there is one implementation of it rather than a JS fourth.
      await ensureReady()
      const statements = TraverseDb.splitStatements(payload.cypher)
      const before = { n: db.nodeCount(), e: db.edgeCount() }
      const t0 = performance.now()
      let stats = emptyStats()
      let executed = 0
      const errors = []
      for (const statement of statements) {
        try {
          const r = db.query(statement, null)
          executed += 1
          // Accumulate against the ORIGINAL counts so the totals describe the
          // whole import rather than only its last statement.
          stats = stats0FromResponse(r, before, db)
        } catch (err) {
          // Keep going, like the server does: one bad statement in a long
          // script should not discard the ones that worked, and the caller
          // gets told exactly which failed.
          errors.push({ statement, error: String(err.message ?? err) })
        }
      }
      return {
        total_statements: statements.length,
        executed,
        errors,
        stats,
        time_ms: performance.now() - t0,
      }
    }

    // ── OPFS persistence ────────────────────────────────────────

    case 'commit': {
      await ensureReady()
      const bytes = db.exportTvdb()
      await opfsWrite(payload.name, bytes)
      return { ok: true, bytes: bytes.byteLength }
    }

    case 'writeBytes': {
      // Persist a caller-provided `.tvdb` byte buffer to OPFS WITHOUT
      // touching the engine. Used by the upload flow: lets the file
      // manager show "uploading" then "loading into memory" as two
      // separate phases, instead of one long opaque pause while the
      // engine parses a 500 MB tvdb.
      await ensureReady()
      const bytes = payload.bytes
      await opfsWrite(payload.name, bytes)
      return { ok: true, bytes: bytes.byteLength }
    }

    case 'open': {
      await ensureReady()
      const bytes = await opfsRead(payload.name)
      if (!bytes) {
        return { ok: false, missing: true }
      }
      db.loadTvdb(bytes)
      return { ok: true, nodes: db.nodeCount(), edges: db.edgeCount() }
    }

    case 'listDatabases':
      return await opfsList()

    case 'deleteDatabase':
      await opfsDelete(payload.name)
      return { ok: true }

    // ── Manual byte-level I/O for File-API import / Download export ──
    // The bytes are the same `.tvdb` binary format the native server
    // writes — round-trips cleanly between this WASM build and the
    // native build.

    case 'exportTvdb': {
      await ensureReady()
      const bytes = db.exportTvdb()
      return { bytes }
    }

    case 'loadTvdb': {
      await ensureReady()
      db.loadTvdb(payload.bytes)
      return { ok: true, nodes: db.nodeCount(), edges: db.edgeCount() }
    }

    default:
      throw new Error(`Unknown worker request kind: ${kind}`)
  }
}

// ── OPFS helpers ────────────────────────────────────────────────────

/** Get-or-create the Traverse subdirectory in OPFS root. Databases
 *  live here as `<name>.tvdb` (binary) files; per-database studio
 *  metadata (styles, saved queries) lives as `<name>.studio.json`
 *  sidecar files. */
async function opfsDir() {
  const root = await navigator.storage.getDirectory()
  return root.getDirectoryHandle('traverse', { create: true })
}

/** Empty QueryStats — Cypher executor doesn't emit one yet on WASM,
 *  so import stats are derived from before/after counts. */
function emptyStats() {
  return {
    nodes_created: 0,
    nodes_deleted: 0,
    relationships_created: 0,
    relationships_deleted: 0,
    properties_set: 0,
    labels_added: 0,
    labels_removed: 0,
  }
}

/** Best-effort stat derivation: compare entity counts before/after.
 *  Doesn't capture property/label churn — the WASM Cypher executor
 *  will surface that more accurately once it threads stats through. */
function stats0FromResponse(_response, before, db) {
  const after = { n: db.nodeCount(), e: db.edgeCount() }
  const stats = emptyStats()
  const dn = after.n - before.n
  const de = after.e - before.e
  if (dn > 0) stats.nodes_created = dn
  else if (dn < 0) stats.nodes_deleted = -dn
  if (de > 0) stats.relationships_created = de
  else if (de < 0) stats.relationships_deleted = -de
  return stats
}

/** Write bytes to OPFS via a sync access handle (worker-only API). */
async function opfsWrite(name, bytes) {
  const dir = await opfsDir()
  const file = await dir.getFileHandle(filename(name), { create: true })
  const handle = await file.createSyncAccessHandle()
  try {
    handle.truncate(0)
    handle.write(bytes, { at: 0 })
    handle.flush()
  } finally {
    handle.close()
  }
}

/** Read bytes from OPFS. Returns null if the file doesn't exist. */
async function opfsRead(name) {
  const dir = await opfsDir()
  let file
  try {
    file = await dir.getFileHandle(filename(name))
  } catch {
    return null
  }
  const handle = await file.createSyncAccessHandle()
  try {
    const size = handle.getSize()
    const buf = new Uint8Array(size)
    handle.read(buf, { at: 0 })
    return buf
  } finally {
    handle.close()
  }
}

/** List databases in OPFS root. Sidecar `.studio.json` files are
 *  hidden from this listing (Studio reaches for them separately). */
async function opfsList() {
  const dir = await opfsDir()
  const items = []
  // @ts-ignore — OPFS async iterator is well-supported in modern browsers.
  for await (const [filenameStr, entry] of dir.entries()) {
    if (entry.kind !== 'file') continue
    if (!filenameStr.endsWith('.tvdb')) continue
    const f = await entry.getFile()
    items.push({
      name: filenameStr.slice(0, -'.tvdb'.length),
      size: f.size,
      lastModified: f.lastModified,
    })
  }
  return items.sort((a, b) => b.lastModified - a.lastModified)
}

/** Delete a database from OPFS. No-op if it doesn't exist. */
async function opfsDelete(name) {
  const dir = await opfsDir()
  try {
    await dir.removeEntry(filename(name))
  } catch {
    // Already gone — fine.
  }
}

/** Filename for an OPFS-stored database. Slashes / dots in the user
 *  name are stripped to keep the OPFS layout flat. The binary tvdb
 *  buffer (paged layout, lz4-per-record, same on-disk format as the
 *  native server) lives under `<name>.tvdb`. */
function filename(name) {
  const safe = String(name).replace(/[\/\\.]/g, '_')
  return `${safe}.tvdb`
}
