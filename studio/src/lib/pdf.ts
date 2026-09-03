// Client-side PDF (pdfium WASM, via lector) for the Studio: page counts for the
// attachment chip + the in-app viewer. One engine, lazily booted - the ~5.8 MB
// wasm loads once and is reused across chips and viewer opens. Single-threaded
// pdfium build, so no COOP/COEP cross-origin-isolation headers are needed; the
// loader + .wasm are served from /pdfium/ (studio/public/pdfium).
import { LectorEngine } from '@truespar/lector-core'
import { CORE_PLUGINS, READER_PLUGINS } from '@truespar/lector-vue'
// `?worker&url` points straight at the package's module-worker export - a .ts
// re-export or a side-effect import both fail under Vite 8/rolldown (the worker
// gets mis-typed or tree-shaken). See vite.config.ts + hq's PdfPreview notes.
import pdfiumWorkerUrl from '@truespar/lector-core/worker?worker&url'

let enginePromise: Promise<LectorEngine> | null = null

/** The shared, initialised engine (booted on first use, with the CORE plugins
 *  the viewer needs already registered). */
export function pdfEngine(): Promise<LectorEngine> {
  if (!enginePromise) {
    enginePromise = (async () => {
      const engine = new LectorEngine({
        // Both primary + fallback point at the single-threaded build, so it
        // loads regardless of crossOriginIsolated (we never set the headers).
        wasmUrl: '/pdfium/pdfium-st.wasm',
        wasmJsUrl: '/pdfium/pdfium-st.js',
        wasmUrlFallback: '/pdfium/pdfium-st.wasm',
        wasmJsUrlFallback: '/pdfium/pdfium-st.js',
        workerUrl: new URL(pdfiumWorkerUrl, import.meta.url),
      })
      for (const p of CORE_PLUGINS) engine.plugins.register(p)
      await engine.init()
      return engine
    })()
  }
  return enginePromise
}

let viewerEnginePromise: Promise<LectorEngine> | null = null

/** The document pane's viewer engine - Separate from the raster engine above
 *  deliberately: the LectorViewer juggles the engine-wide active document, and
 *  sharing that with the chips' transient open/close raster traffic would
 *  have the two fighting over it. READER preset, not all: with the
 *  annotation/form/signature plugins registered, the PDF's own annotation
 *  widgets rendered as interactive boxes on TOP of the OCR region overlay
 *  and the editing chrome lit up - in a chat pane the document is a reading
 * surface. */
export function pdfViewerEngine(): Promise<LectorEngine> {
  if (!viewerEnginePromise) {
    viewerEnginePromise = (async () => {
      const engine = new LectorEngine({
        wasmUrl: '/pdfium/pdfium-st.wasm',
        wasmJsUrl: '/pdfium/pdfium-st.js',
        wasmUrlFallback: '/pdfium/pdfium-st.wasm',
        wasmJsUrlFallback: '/pdfium/pdfium-st.js',
        workerUrl: new URL(pdfiumWorkerUrl, import.meta.url),
      })
      for (const p of READER_PLUGINS) engine.plugins.register(p)
      await engine.init()
      return engine
    })()
  }
  return viewerEnginePromise
}

/** Page count of a PDF, client-side (no server round-trip). Best-effort:
 *  returns 0 when pdfium can't parse the bytes. `openDocument` TRANSFERS the
 *  buffer to the worker (detaches it), so we hand over a copy. */
export async function pdfPageCount(bytes: ArrayBuffer): Promise<number> {
  try {
    const engine = await pdfEngine()
    const handle = await engine.openDocument(bytes.slice(0))
    const n = handle.pageCount
    await handle.close()
    return n
  } catch {
    return 0
  }
}
