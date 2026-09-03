import { fileURLToPath, URL } from 'node:url'
import { readFileSync } from 'node:fs'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { compression } from 'vite-plugin-compression2'

// The lector pdfium worker loads its emscripten glue at runtime via a URL under
// /pdfium/. Vite dev REFUSES to import a /public file as a module, so serve the
// pdfium loader + wasm from a middleware registered before Vite's transform
// middleware (directly on server.middlewares intercepts first). Single-threaded
// pdfium build -> no COOP/COEP headers needed. Mirrors hq's web-ui config.
function servePdfium() {
  const pubRoot = fileURLToPath(new URL('./public', import.meta.url))
  const mime: Record<string, string> = {
    '.js': 'application/javascript',
    '.wasm': 'application/wasm',
  }
  return {
    name: 'serve-pdfium',
    configureServer(server: {
      middlewares: {
        use: (
          fn: (
            req: { url?: string },
            res: { setHeader: (k: string, v: string) => void; end: (b: Buffer) => void },
            next: () => void,
          ) => void,
        ) => void
      }
    }) {
      server.middlewares.use((req, res, next) => {
        const p = req.url?.split('?')[0] ?? ''
        const ext = p.slice(p.lastIndexOf('.'))
        if (p.startsWith('/pdfium/') && !p.includes('..') && mime[ext]) {
          try {
            const body = readFileSync(`${pubRoot}${p}`)
            res.setHeader('Content-Type', mime[ext])
            res.setHeader('Cache-Control', 'no-cache')
            res.end(body)
            return
          } catch {
            /* fall through to vite default */
          }
        }
        next()
      })
    },
  }
}

// The browser tab's icon, taken from the same file the two exes are branded
// with (assets/paddock.ico, via winresource in each crate's build.rs) rather
// than a copy in public/. One source of truth: a duplicate would let the
// desktop icon change while the tab kept the old one, and nobody would notice
// for months.
//
// Until this existed the Studio had no favicon at all, which was worse than it
// sounds - the manager falls unmatched paths back to the SPA shell, so
// /favicon.ico answered with the whole index.html as text/html. The browser
// cannot render that, shows its generic icon, and pays for a wasted round trip
// on every load.
//
// The .ico carries 16..256 including a 256 that is already PNG-encoded inside,
// so apple-touch-icon is SLICED out of it rather than re-encoded - no quality
// loss and no second art file to keep in step.
function serveIcons() {
  const ico = fileURLToPath(new URL('../assets/paddock.ico', import.meta.url))
  /** The largest PNG-encoded image inside an .ico, or null if it holds none. */
  const embeddedPng = (buf: Buffer): Buffer | null => {
    const count = buf.readUInt16LE(4)
    let best: Buffer | null = null
    let bestPx = 0
    for (let i = 0; i < count; i++) {
      const e = 6 + i * 16
      // 0 in the width byte means 256 - the format cannot express it directly.
      const px = buf[e] === 0 ? 256 : buf[e]
      const size = buf.readUInt32LE(e + 8)
      const off = buf.readUInt32LE(e + 12)
      const isPng = buf[off] === 0x89 && buf[off + 1] === 0x50
      if (isPng && px > bestPx) {
        best = buf.subarray(off, off + size)
        bestPx = px
      }
    }
    return best
  }
  const files = () => {
    const buf = readFileSync(ico)
    const out: Record<string, Buffer> = { 'favicon.ico': buf }
    const png = embeddedPng(buf)
    if (png) out['apple-touch-icon.png'] = png
    return out
  }
  return {
    name: 'serve-icons',
    // dev: the file lives outside public/, so serve it by hand
    configureServer(server: {
      middlewares: {
        use: (
          fn: (
            req: { url?: string },
            res: { setHeader: (k: string, v: string) => void; end: (b: Buffer) => void },
            next: () => void,
          ) => void,
        ) => void
      }
    }) {
      server.middlewares.use((req, res, next) => {
        const p = req.url?.split('?')[0]?.replace(/^\//, '') ?? ''
        const f = files()[p]
        if (!f) return next()
        res.setHeader('Content-Type', p.endsWith('.ico') ? 'image/x-icon' : 'image/png')
        res.end(f)
      })
    },
    generateBundle(this: { emitFile: (f: unknown) => void }) {
      for (const [fileName, source] of Object.entries(files())) {
        this.emitFile({ type: 'asset', fileName, source })
      }
    },
  }
}

// The Studio builds into the server crate's `static/` dir; `paddock` serves it.
// In dev, proxy the store/telemetry API to a locally running `paddock` manager
// (:11500) and inference to a `paddock-runner` (:11540) - the manager/runner
// split.
export default defineConfig({
  plugins: [
    vue(),
    compression({
      algorithms: ['gzip'],
      include: /\.(js|css|html|json|svg)$/,
    }),
    servePdfium(),
    serveIcons(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      // Monaco's internals, reachable. Its package exports map rewrites deep
      // specifiers (`./*` -> `./esm/vs/*.js`) and cannot express a .css import
      // at all, so src/lib/monaco-entry.ts - our trimmed build of monaco's own
      // entry - goes through this instead. See scripts/gen-monaco-entry.mjs.
      'monaco-vs': fileURLToPath(new URL('./node_modules/monaco-editor/esm/vs', import.meta.url)),
      // Scriptor (the Word/.docx engine) is vendored under vendor/scriptor from
      // github.com/truespar/scriptor at the revision its VERSION file records,
      // the same one the Rust side pins. Its packages use workspace:*
      // protocols npm can't install, so the sources are aliased straight in
      // rather than listed as file: dependencies. The wasm dist is the
      // wasm-pack build of that revision; its wasm-bindgen glue references
      // scriptor_wasm_bg.wasm by URL, which the build copies into assets.
      '@truespar/scriptor-vue': fileURLToPath(
        new URL('./vendor/scriptor/vue/src/index.ts', import.meta.url),
      ),
      '@truespar/scriptor-core': fileURLToPath(
        new URL('./vendor/scriptor/core/src/index.ts', import.meta.url),
      ),
      '@truespar/scriptor-wasm': fileURLToPath(
        new URL('./vendor/scriptor/scriptor-wasm/dist/scriptor_wasm.js', import.meta.url),
      ),
    },
  },
  // Lector's pdfium worker is a MODULE worker (engine does
  // `new Worker(url, { type: 'module' })`) - build it as ESM. The wasm bindings
  // package must not be pre-bundled: it loads the .wasm by URL at runtime.
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['@truespar/lector-pdfium-wasm', '@truespar/traverse-wasm'] },
  // Treat .wasm as an asset so the traverse glue's
  // `new URL('traverse_wasm_bg.wasm', import.meta.url)` copies the binary
  // through unchanged instead of trying to parse it as a module.
  assetsInclude: ['**/*.wasm'],
  server: {
    port: 5273,
    proxy: {
      // Both targets are overridable, because "which manager" and "which
      // runner" are per-box facts: PADDOCK_DEV_MANAGER / PADDOCK_DEV_RUNNER.
      //
      // The manager defaults to https (made TLS the default, so
      // LAN browsers get a secure context) and serves a certificate signed by
      // paddock's own CA in `data/tls`. Node rejects that as self-signed, so
      // `secure: false` is required or every /api call in dev fails with
      // DEPTH_ZERO_SELF_SIGNED_CERT - which reads as "the manager is down".
      // This proxy still said `http://` long after that switch, so the dev server
      // could not talk to a default manager at all.
      '/v1': {
        target: process.env.PADDOCK_DEV_RUNNER ?? 'http://localhost:11540',
        secure: false,
        changeOrigin: true,
      },
      '/api': {
        target: process.env.PADDOCK_DEV_MANAGER ?? 'https://localhost:11500',
        secure: false,
        changeOrigin: true,
      },
    },
    // The telemetry WebSocket connects straight to the server in dev (see
    // VITE_API_WS in .env.development) because Vite's proxy doesn't reliably
    // upgrade WS alongside its own HMR socket. In prod the manager serves the
    // Studio same-origin, so no override is needed.
  },
  build: {
    outDir: '../crates/paddock-manager/static',
    emptyOutDir: true,
    // Two chunks are far above Vite's 500 kB default: monaco-editor (~3.8 MB,
    // its own chunk, fetched when an editor opens) and the app entry (~3.4 MB,
    // everything the shell imports eagerly). The limit records where they
    // stand so the build stays quiet; it is not a target. Splitting the
    // entry with dynamic imports is real work still to do.
    chunkSizeWarningLimit: 4096,
  },
})
