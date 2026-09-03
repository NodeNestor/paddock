import "./chunk-UAWBPTDW.js";

// src/worker/wasm-cache.ts
var CACHE_NAME = "lector-wasm-v1";
async function getCachedResponse(url) {
  try {
    if (typeof caches === "undefined") return null;
    const cache = await caches.open(CACHE_NAME);
    const response = await cache.match(url);
    return response ?? null;
  } catch {
    return null;
  }
}
async function cacheResponse(url, response) {
  try {
    if (typeof caches === "undefined") return;
    const cache = await caches.open(CACHE_NAME);
    await cache.put(url, response);
  } catch {
  }
}
async function loadWasmCached(wasmUrl, imports) {
  const cached = await getCachedResponse(wasmUrl);
  if (cached && typeof WebAssembly.instantiateStreaming === "function") {
    const result2 = await WebAssembly.instantiateStreaming(cached, imports);
    return { instance: result2.instance, module: result2.module };
  }
  if (typeof WebAssembly.instantiateStreaming === "function") {
    const response2 = await fetch(wasmUrl);
    const responseForCache = response2.clone();
    const result2 = await WebAssembly.instantiateStreaming(response2, imports);
    void cacheResponse(wasmUrl, responseForCache);
    return { instance: result2.instance, module: result2.module };
  }
  const response = await fetch(wasmUrl);
  const bytes = await response.arrayBuffer();
  const result = await WebAssembly.instantiate(bytes, imports);
  return { instance: result.instance, module: result.module };
}
function createInstantiateWasmHook(wasmUrl) {
  return (imports, receiveInstance) => {
    loadWasmCached(wasmUrl, imports).then(({ instance, module }) => {
      receiveInstance(instance, module);
    }).catch((err) => {
      throw err;
    });
    return {};
  };
}
export {
  createInstantiateWasmHook,
  loadWasmCached
};
