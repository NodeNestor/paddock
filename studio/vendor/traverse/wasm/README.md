# @truespar/traverse-wasm

The full [Traverse](https://truespar.com/traverse) graph engine in a
browser tab via WebAssembly. Run Cypher queries and 36+ graph-data-science
algorithms — PageRank, Louvain, Dijkstra, FastRP, etc. — entirely
in-browser, no server required. Hosted in a dedicated Web Worker so the
main thread never blocks. Persists databases to OPFS so they survive
tab reloads.

**Free for any use.** No license key, no telemetry, no upsell — the
WASM build is open for evaluation, demos, prototypes, and production.

Try a live demo: **<https://traverse.truespar.com>**

---

## Install

```sh
npm install @truespar/traverse-wasm
```

Requires Node.js 18+ for the toolchain; the package itself runs in
browsers (see [Browser support](#browser-support)).

## Quickstart

```ts
import { TraverseDb } from '@truespar/traverse-wasm'

// Spawn the engine (one Web Worker + the WASM module)
const db = await TraverseDb.open({ name: 'myapp' })

// Write something
await db.query(`
  CREATE (a:Person {name:'Ada'})-[:KNOWS]->(b:Person {name:'Bob'})
`)

// Read it back
const res = await db.query('MATCH (a)-[:KNOWS]->(b) RETURN a.name, b.name')
console.log(res.columns)   // ['a.name', 'b.name']
console.log(res.rows)      // [['Ada', 'Bob']]

// Persist to the browser's OPFS (survives reload)
await db.commit('myapp')

// ...later, in another tab session:
const restored = await TraverseDb.open({ name: 'myapp' })
const count = await restored.nodeCount()  // 2
```

That's the full lifecycle. The engine is the same one that ships in
the licensed server — same Cypher, same algorithms, same `.tvdb`
binary format.

---

## Required hosting setup: cross-origin isolation

The engine uses `SharedArrayBuffer` for multi-threading. Browsers only
expose `SharedArrayBuffer` on pages that are **cross-origin isolated**,
which means your host must serve every page that loads `@truespar/traverse-wasm`
with these two response headers:

```
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Opener-Policy:   same-origin
```

Without them you'll get `ReferenceError: SharedArrayBuffer is not defined`
when calling `TraverseDb.open()`. Recipes for common hosts below — pick
the one matching your stack.

### Vite (dev + preview)

```ts
// vite.config.ts
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin',
    },
  },
})
```

### Vercel

```json
// vercel.json
{
  "headers": [{
    "source": "/(.*)",
    "headers": [
      { "key": "Cross-Origin-Embedder-Policy", "value": "require-corp" },
      { "key": "Cross-Origin-Opener-Policy", "value": "same-origin" }
    ]
  }]
}
```

### Netlify

```
# public/_headers
/*
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
```

### Cloudflare Pages

```
# public/_headers (same syntax as Netlify)
/*
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Opener-Policy: same-origin
```

### nginx

```nginx
add_header Cross-Origin-Embedder-Policy "require-corp" always;
add_header Cross-Origin-Opener-Policy   "same-origin" always;
```

### Apache

```apache
Header always set Cross-Origin-Embedder-Policy "require-corp"
Header always set Cross-Origin-Opener-Policy   "same-origin"
```

### IIS

```xml
<!-- web.config -->
<configuration>
  <system.webServer>
    <httpProtocol>
      <customHeaders>
        <add name="Cross-Origin-Embedder-Policy" value="require-corp" />
        <add name="Cross-Origin-Opener-Policy"   value="same-origin" />
      </customHeaders>
    </httpProtocol>
  </system.webServer>
</configuration>
```

### Test it

After deploying, open the page and check:

```js
console.log(typeof SharedArrayBuffer)  // 'function' = isolated
console.log(crossOriginIsolated)        // true
```

If either is missing, the headers didn't take effect.

---

## API

### `TraverseDb.open(options?)`

Spawns the worker, initializes the WASM module and thread pool, and
optionally restores an OPFS-resident database.

```ts
const db = await TraverseDb.open({
  name: 'myapp',           // optional: OPFS db to restore
  numThreads: 8,           // optional: default = min(hardwareConcurrency, 16)
  workerUrl: '/my-worker', // optional: override worker script URL
})
```

If `name` is set but no such database exists, opens empty — call
`commit(name)` later to create it.

### `db.query(cypher, params?, options?)`

```ts
const res = await db.query(
  'MATCH (n:Person) WHERE n.age > $minAge RETURN n.name, n.age',
  { minAge: 30 },
  { signal: controller.signal, timeoutMs: 5000 },
)
// res.columns: string[]
// res.rows: unknown[][]                — each cell may be a primitive,
//                                        or a tagged { _type: 'node'|'edge'|'path', ... }
// res.nodes, res.edges                 — deduplicated entities for graph rendering
// res.stats.nodes_created etc.         — write counters
// res.parsing_time_ms / planning_time_ms / execution_time_ms / time_ms
// res.query_type: 'Read'|'Write'|'ReadWrite'|'Schema'
```

`timeoutMs` installs a thread-local deadline the executor checks at
iteration boundaries. `signal` lets you AbortController-cancel from
the main thread (the worker also receives the cancellation).

### `db.algorithms()`

Returns the live GDS catalog — every algorithm with its supported
modes, required/optional config, and output schema. Use this to drive
a UI catalog or to introspect what's runnable:

```ts
const { algorithms } = await db.algorithms()
for (const a of algorithms) {
  console.log(`${a.name}: modes=${a.modes.join('/')} — ${a.description}`)
}
```

Then invoke any of them with regular Cypher:

```ts
await db.query(`
  CALL traverse.pageRank.write({writeProperty: 'rank', maxIterations: 20})
  YIELD nodeCount, mean, computeMillis
  RETURN nodeCount, mean, computeMillis
`)
```

### `db.schema()`

Returns labels, relationship types, property keys (with types), and
the per-label / per-edge-type property listings. Mirrors the HTTP
`/api/schema` response.

### `db.importCypher(text)`

Run a multi-statement Cypher import script. Returns per-statement
results plus aggregate stats. Same shape as the licensed server's
`/api/import` endpoint.

### Persistence

```ts
await db.commit('myapp')                       // → OPFS
const list = await db.listDatabases()          // → [{name, size, lastModified}]
await db.load('myapp')                          // ← OPFS
await db.deleteDatabase('myapp')

const bytes = await db.exportTvdb()             // → Uint8Array (.tvdb format)
await db.loadTvdb(bytes)                        // ← Uint8Array

await db.writeBytes('upload', uploadedFile)    // write straight to OPFS, then load() separately
```

The `.tvdb` format is identical to the one the licensed server reads
and writes — you can `db.exportTvdb()` in the browser, save the bytes,
and `Database::open` the same file from a server-side embedded binding.

### Static helpers

```ts
await TraverseDb.listDatabases()                // OPFS list without opening a TraverseDb
await TraverseDb.deleteDatabase('myapp')        // delete without opening
```

### Cleanup

```ts
db.close()    // terminates the worker; pending requests reject with AbortError
```

---

## OPFS persistence

The package writes databases to the browser's
[Origin Private File System](https://web.dev/articles/origin-private-file-system) —
a sandboxed filesystem visible only to your origin. It is:

- Persistent across tab reloads and browser restarts
- Subject to the browser's storage quota (typically gigabytes)
- Not visible from other origins or via DevTools as a regular filesystem
- Wiped when the user clears site data

To check available quota:

```ts
const { quota, usage } = await navigator.storage.estimate()
console.log(`OPFS: ${usage} / ${quota} bytes`)
```

---

## TypeScript

The package ships full `.d.ts` files. Your editor should autocomplete
everything (`QueryResponse`, `GdsAlgorithm`, etc.). Strict-mode safe.

```ts
import { TraverseDb, type QueryResponse, type GdsAlgorithm } from '@truespar/traverse-wasm'
```

---

## Browser support

| Browser | Minimum | Notes |
|---|---|---|
| Chrome / Edge | 92+ | Full support |
| Firefox | 89+ | Full support |
| Safari | 16.4+ | Full support; iOS has tight memory caps (see below) |

Hard requirements (all browsers):

- `SharedArrayBuffer` (gated by COEP/COOP headers — see above)
- `WebAssembly.Memory` with `shared: true`
- OPFS (`navigator.storage.getDirectory`) for persistence

**iOS Safari memory.** Phones cap WebAssembly memory at roughly 300–500
MiB, much lower than desktop's ~4 GiB. Plan dataset sizes accordingly.

---

## Bundle size & runtime memory

- **Wire size**: ~1.4 MB gzipped (~4.2 MB unpacked, mostly the WASM
  binary itself)
- **Linear-memory ceiling**: ~4 GiB on desktop (wasm32 hard cap);
  ~300–500 MiB on iOS Safari. Memory64 raises the ceiling to ~16 GiB
  on Chrome 133+ / Firefox 134+ but isn't yet enabled in this build.
- **Boot time**: typically 200–600 ms on desktop, 1–3 s on mobile

The engine streams the WASM module on `TraverseDb.open()`. Subsequent
`open()` calls reuse the browser's WASM cache.

---

## Related

- **Live demo**: <https://traverse.truespar.com>
- **Integration guide** (long-form): <https://truespar.com/traverse/docs/v1/wasm-integration>
- **Cypher reference**: <https://truespar.com/traverse/docs/v1/cypher>
- **GDS reference**: <https://truespar.com/traverse/docs/v1/gds>
- **Licensed server** (multi-user, server-side persistence, more):
  <https://truespar.com/traverse/docs/v1/getting-started>

---

## License

Dual-licensed MIT OR Apache-2.0, at your option. See the LICENSE,
LICENSE-MIT and LICENSE-APACHE files at the root of this vendored drop.
No license key, no runtime telemetry, no usage limits.

For server-side Traverse — multi-user, network protocols (Bolt/HTTP/gRPC),
embedded SDKs for Python/Java/Node/Go/.NET, or production support — see
the [server docs](https://truespar.com/traverse/docs/v1/getting-started).
