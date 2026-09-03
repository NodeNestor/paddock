/**
 * WASM Module Cache — Cache API + streaming compilation.
 *
 * Uses the browser Cache API to store the WASM fetch Response. On
 * subsequent loads, `WebAssembly.instantiateStreaming()` reads from
 * cache — no network request, and the browser can apply its internal
 * compilation cache on top (V8 caches compiled code alongside the
 * response in the HTTP cache).
 *
 * Fallback chain:
 * 1. Cache API hit → `instantiateStreaming(cachedResponse, imports)`
 * 2. Cache API miss → `instantiateStreaming(fetch(url), imports)` + cache
 * 3. No Cache API → `fetch` + `WebAssembly.instantiate(bytes, imports)`
 */
/**
 * Load and instantiate a WASM module with Cache API + streaming compilation.
 *
 * @param wasmUrl - URL of the .wasm binary
 * @param imports - WebAssembly import object
 * @returns The instantiated module
 */
export declare function loadWasmCached(wasmUrl: string, imports: WebAssembly.Imports): Promise<{
    instance: WebAssembly.Instance;
    module: WebAssembly.Module;
}>;
/**
 * Create an `instantiateWasm` hook for Emscripten that uses
 * Cache API + streaming compilation.
 *
 * @param wasmUrl - URL of the .wasm binary
 * @returns The hook function to pass as `PdfiumModuleConfig.instantiateWasm`
 */
export declare function createInstantiateWasmHook(wasmUrl: string): (imports: WebAssembly.Imports, receiveInstance: (instance: WebAssembly.Instance, module?: WebAssembly.Module) => void) => WebAssembly.Exports;
//# sourceMappingURL=wasm-cache.d.ts.map