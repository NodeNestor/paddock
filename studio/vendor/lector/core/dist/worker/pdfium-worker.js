import {
  __callDispose,
  __using
} from "./chunk-UAWBPTDW.js";

// ../../node_modules/.pnpm/comlink@4.4.2/node_modules/comlink/dist/esm/comlink.mjs
var proxyMarker = /* @__PURE__ */ Symbol("Comlink.proxy");
var createEndpoint = /* @__PURE__ */ Symbol("Comlink.endpoint");
var releaseProxy = /* @__PURE__ */ Symbol("Comlink.releaseProxy");
var finalizer = /* @__PURE__ */ Symbol("Comlink.finalizer");
var throwMarker = /* @__PURE__ */ Symbol("Comlink.thrown");
var isObject = (val) => typeof val === "object" && val !== null || typeof val === "function";
var proxyTransferHandler = {
  canHandle: (val) => isObject(val) && val[proxyMarker],
  serialize(obj) {
    const { port1, port2 } = new MessageChannel();
    expose(obj, port1);
    return [port2, [port2]];
  },
  deserialize(port) {
    port.start();
    return wrap(port);
  }
};
var throwTransferHandler = {
  canHandle: (value) => isObject(value) && throwMarker in value,
  serialize({ value }) {
    let serialized;
    if (value instanceof Error) {
      serialized = {
        isError: true,
        value: {
          message: value.message,
          name: value.name,
          stack: value.stack
        }
      };
    } else {
      serialized = { isError: false, value };
    }
    return [serialized, []];
  },
  deserialize(serialized) {
    if (serialized.isError) {
      throw Object.assign(new Error(serialized.value.message), serialized.value);
    }
    throw serialized.value;
  }
};
var transferHandlers = /* @__PURE__ */ new Map([
  ["proxy", proxyTransferHandler],
  ["throw", throwTransferHandler]
]);
function isAllowedOrigin(allowedOrigins, origin) {
  for (const allowedOrigin of allowedOrigins) {
    if (origin === allowedOrigin || allowedOrigin === "*") {
      return true;
    }
    if (allowedOrigin instanceof RegExp && allowedOrigin.test(origin)) {
      return true;
    }
  }
  return false;
}
function expose(obj, ep = globalThis, allowedOrigins = ["*"]) {
  ep.addEventListener("message", function callback(ev) {
    if (!ev || !ev.data) {
      return;
    }
    if (!isAllowedOrigin(allowedOrigins, ev.origin)) {
      console.warn(`Invalid origin '${ev.origin}' for comlink proxy`);
      return;
    }
    const { id, type, path } = Object.assign({ path: [] }, ev.data);
    const argumentList = (ev.data.argumentList || []).map(fromWireValue);
    let returnValue;
    try {
      const parent = path.slice(0, -1).reduce((obj2, prop) => obj2[prop], obj);
      const rawValue = path.reduce((obj2, prop) => obj2[prop], obj);
      switch (type) {
        case "GET":
          {
            returnValue = rawValue;
          }
          break;
        case "SET":
          {
            parent[path.slice(-1)[0]] = fromWireValue(ev.data.value);
            returnValue = true;
          }
          break;
        case "APPLY":
          {
            returnValue = rawValue.apply(parent, argumentList);
          }
          break;
        case "CONSTRUCT":
          {
            const value = new rawValue(...argumentList);
            returnValue = proxy(value);
          }
          break;
        case "ENDPOINT":
          {
            const { port1, port2 } = new MessageChannel();
            expose(obj, port2);
            returnValue = transfer(port1, [port1]);
          }
          break;
        case "RELEASE":
          {
            returnValue = void 0;
          }
          break;
        default:
          return;
      }
    } catch (value) {
      returnValue = { value, [throwMarker]: 0 };
    }
    Promise.resolve(returnValue).catch((value) => {
      return { value, [throwMarker]: 0 };
    }).then((returnValue2) => {
      const [wireValue, transferables] = toWireValue(returnValue2);
      ep.postMessage(Object.assign(Object.assign({}, wireValue), { id }), transferables);
      if (type === "RELEASE") {
        ep.removeEventListener("message", callback);
        closeEndPoint(ep);
        if (finalizer in obj && typeof obj[finalizer] === "function") {
          obj[finalizer]();
        }
      }
    }).catch((error) => {
      const [wireValue, transferables] = toWireValue({
        value: new TypeError("Unserializable return value"),
        [throwMarker]: 0
      });
      ep.postMessage(Object.assign(Object.assign({}, wireValue), { id }), transferables);
    });
  });
  if (ep.start) {
    ep.start();
  }
}
function isMessagePort(endpoint) {
  return endpoint.constructor.name === "MessagePort";
}
function closeEndPoint(endpoint) {
  if (isMessagePort(endpoint))
    endpoint.close();
}
function wrap(ep, target) {
  const pendingListeners = /* @__PURE__ */ new Map();
  ep.addEventListener("message", function handleMessage(ev) {
    const { data } = ev;
    if (!data || !data.id) {
      return;
    }
    const resolver = pendingListeners.get(data.id);
    if (!resolver) {
      return;
    }
    try {
      resolver(data);
    } finally {
      pendingListeners.delete(data.id);
    }
  });
  return createProxy(ep, pendingListeners, [], target);
}
function throwIfProxyReleased(isReleased) {
  if (isReleased) {
    throw new Error("Proxy has been released and is not useable");
  }
}
function releaseEndpoint(ep) {
  return requestResponseMessage(ep, /* @__PURE__ */ new Map(), {
    type: "RELEASE"
  }).then(() => {
    closeEndPoint(ep);
  });
}
var proxyCounter = /* @__PURE__ */ new WeakMap();
var proxyFinalizers = "FinalizationRegistry" in globalThis && new FinalizationRegistry((ep) => {
  const newCount = (proxyCounter.get(ep) || 0) - 1;
  proxyCounter.set(ep, newCount);
  if (newCount === 0) {
    releaseEndpoint(ep);
  }
});
function registerProxy(proxy2, ep) {
  const newCount = (proxyCounter.get(ep) || 0) + 1;
  proxyCounter.set(ep, newCount);
  if (proxyFinalizers) {
    proxyFinalizers.register(proxy2, ep, proxy2);
  }
}
function unregisterProxy(proxy2) {
  if (proxyFinalizers) {
    proxyFinalizers.unregister(proxy2);
  }
}
function createProxy(ep, pendingListeners, path = [], target = function() {
}) {
  let isProxyReleased = false;
  const proxy2 = new Proxy(target, {
    get(_target, prop) {
      throwIfProxyReleased(isProxyReleased);
      if (prop === releaseProxy) {
        return () => {
          unregisterProxy(proxy2);
          releaseEndpoint(ep);
          pendingListeners.clear();
          isProxyReleased = true;
        };
      }
      if (prop === "then") {
        if (path.length === 0) {
          return { then: () => proxy2 };
        }
        const r = requestResponseMessage(ep, pendingListeners, {
          type: "GET",
          path: path.map((p) => p.toString())
        }).then(fromWireValue);
        return r.then.bind(r);
      }
      return createProxy(ep, pendingListeners, [...path, prop]);
    },
    set(_target, prop, rawValue) {
      throwIfProxyReleased(isProxyReleased);
      const [value, transferables] = toWireValue(rawValue);
      return requestResponseMessage(ep, pendingListeners, {
        type: "SET",
        path: [...path, prop].map((p) => p.toString()),
        value
      }, transferables).then(fromWireValue);
    },
    apply(_target, _thisArg, rawArgumentList) {
      throwIfProxyReleased(isProxyReleased);
      const last = path[path.length - 1];
      if (last === createEndpoint) {
        return requestResponseMessage(ep, pendingListeners, {
          type: "ENDPOINT"
        }).then(fromWireValue);
      }
      if (last === "bind") {
        return createProxy(ep, pendingListeners, path.slice(0, -1));
      }
      const [argumentList, transferables] = processArguments(rawArgumentList);
      return requestResponseMessage(ep, pendingListeners, {
        type: "APPLY",
        path: path.map((p) => p.toString()),
        argumentList
      }, transferables).then(fromWireValue);
    },
    construct(_target, rawArgumentList) {
      throwIfProxyReleased(isProxyReleased);
      const [argumentList, transferables] = processArguments(rawArgumentList);
      return requestResponseMessage(ep, pendingListeners, {
        type: "CONSTRUCT",
        path: path.map((p) => p.toString()),
        argumentList
      }, transferables).then(fromWireValue);
    }
  });
  registerProxy(proxy2, ep);
  return proxy2;
}
function myFlat(arr) {
  return Array.prototype.concat.apply([], arr);
}
function processArguments(argumentList) {
  const processed = argumentList.map(toWireValue);
  return [processed.map((v) => v[0]), myFlat(processed.map((v) => v[1]))];
}
var transferCache = /* @__PURE__ */ new WeakMap();
function transfer(obj, transfers) {
  transferCache.set(obj, transfers);
  return obj;
}
function proxy(obj) {
  return Object.assign(obj, { [proxyMarker]: true });
}
function toWireValue(value) {
  for (const [name, handler] of transferHandlers) {
    if (handler.canHandle(value)) {
      const [serializedValue, transferables] = handler.serialize(value);
      return [
        {
          type: "HANDLER",
          name,
          value: serializedValue
        },
        transferables
      ];
    }
  }
  return [
    {
      type: "RAW",
      value
    },
    transferCache.get(value) || []
  ];
}
function fromWireValue(value) {
  switch (value.type) {
    case "HANDLER":
      return transferHandlers.get(value.name).deserialize(value.value);
    case "RAW":
      return value.value;
  }
}
function requestResponseMessage(ep, pendingListeners, msg, transfers) {
  return new Promise((resolve) => {
    const id = generateUUID();
    pendingListeners.set(id, resolve);
    if (ep.start) {
      ep.start();
    }
    ep.postMessage(Object.assign({ id }, msg), transfers);
  });
}
function generateUUID() {
  return new Array(4).fill(0).map(() => Math.floor(Math.random() * Number.MAX_SAFE_INTEGER).toString(16)).join("-");
}

// src/worker/pdfium-worker.ts
import {
  createPdfiumInstance,
  checkHandle,
  FS_SIZEF_SIZE
} from "@truespar/lector-pdfium-wasm";

// src/types/errors.ts
function serializePdfiumError(error) {
  if (isSerializedPdfiumError(error)) {
    return error;
  }
  if (error instanceof Error) {
    const code = "code" in error && typeof error.code === "number" ? error.code : 1;
    const context = "context" in error && typeof error.context === "string" ? error.context : void 0;
    return {
      name: "PdfiumError",
      message: error.message,
      code,
      ...context !== void 0 ? { context } : {}
    };
  }
  return {
    name: "PdfiumError",
    message: String(error),
    code: 1
    // UNKNOWN
  };
}
function isSerializedPdfiumError(value) {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const obj = value;
  return obj["name"] === "PdfiumError" && typeof obj["message"] === "string" && typeof obj["code"] === "number";
}

// src/types/render.ts
var DEFAULT_RENDER_OPTIONS = {
  flags: 2,
  // LCD_TEXT only — annotations are rendered as overlays
  rotation: 0,
  devicePixelRatio: 1,
  backgroundColor: 4294967295
  // opaque white
};

// src/worker/handle-registry.ts
var HandleRegistry = class {
  #nextId = 0;
  #prefix;
  #handles = /* @__PURE__ */ new Map();
  #disposer;
  constructor(prefix, disposer) {
    this.#prefix = prefix;
    this.#disposer = disposer;
  }
  /** Register a handle and return a new opaque ID. */
  register(handle) {
    const id = `${this.#prefix}_${this.#nextId++}`;
    this.#handles.set(id, handle);
    return id;
  }
  /** Resolve an ID to its handle. Throws if not found. */
  resolve(id) {
    const handle = this.#handles.get(id);
    if (handle === void 0) {
      throw new Error(`Handle not found: ${id}`);
    }
    return handle;
  }
  /** Remove a handle by ID and return it. Throws if not found. */
  release(id) {
    const handle = this.#handles.get(id);
    if (handle === void 0) {
      throw new Error(`Handle not found: ${id}`);
    }
    this.#handles.delete(id);
    return handle;
  }
  /** Replace the handle for an existing ID (does NOT call the disposer). */
  replace(id, handle) {
    if (!this.#handles.has(id)) {
      throw new Error(`Handle not found: ${id}`);
    }
    this.#handles.set(id, handle);
  }
  /** Check whether an ID is registered. */
  has(id) {
    return this.#handles.has(id);
  }
  /** Number of currently registered handles. */
  get size() {
    return this.#handles.size;
  }
  /** Dispose all remaining handles via the configured disposer. */
  [Symbol.dispose]() {
    for (const handle of this.#handles.values()) {
      this.#disposer(handle);
    }
    this.#handles.clear();
  }
};

// src/worker/document-store.ts
var DocumentStore = class {
  #pdfium;
  #registry;
  constructor(pdfium2) {
    this.#pdfium = pdfium2;
    this.#registry = new HandleRegistry(
      "doc",
      (state) => this.#disposeDocument(state)
    );
  }
  /** Register a new document and return its opaque ID. */
  register(state) {
    return this.#registry.register(state);
  }
  /** Resolve a document ID to its internal state. Throws if not found. */
  resolve(docId) {
    return this.#registry.resolve(docId);
  }
  /** Close a document by ID, freeing all associated resources. */
  release(docId) {
    const state = this.#registry.release(docId);
    this.#disposeDocument(state);
  }
  /** Check whether a document ID is valid. */
  has(docId) {
    return this.#registry.has(docId);
  }
  /**
   * Update cached page info after a page operation (insert, delete, move, rotate).
   * The caller must pass freshly-read page sizes from pdfium.
   */
  updatePageInfo(docId, pageSizes) {
    const state = this.#registry.resolve(docId);
    this.#registry.replace(docId, {
      ...state,
      pageCount: pageSizes.length,
      pageSizes
    });
  }
  /** Number of currently open documents. */
  get size() {
    return this.#registry.size;
  }
  /** Close all open documents and free all WASM resources. */
  [Symbol.dispose]() {
    this.#registry[Symbol.dispose]();
  }
  #disposeDocument(state) {
    if (state.formHandle !== 0) {
      this.#pdfium.fn._FPDFDOC_ExitFormFillEnvironment(state.formHandle);
    }
    this.#pdfium.fn._lector_form_release(state.docHandle);
    this.#pdfium.fn._FPDF_CloseDocument(state.docHandle);
    state.formInfoAlloc[Symbol.dispose]();
    state.pdfAlloc?.[Symbol.dispose]();
  }
};

// src/worker/sha256.ts
var K = new Uint32Array([
  1116352408,
  1899447441,
  3049323471,
  3921009573,
  961987163,
  1508970993,
  2453635748,
  2870763221,
  3624381080,
  310598401,
  607225278,
  1426881987,
  1925078388,
  2162078206,
  2614888103,
  3248222580,
  3835390401,
  4022224774,
  264347078,
  604807628,
  770255983,
  1249150122,
  1555081692,
  1996064986,
  2554220882,
  2821834349,
  2952996808,
  3210313671,
  3336571891,
  3584528711,
  113926993,
  338241895,
  666307205,
  773529912,
  1294757372,
  1396182291,
  1695183700,
  1986661051,
  2177026350,
  2456956037,
  2730485921,
  2820302411,
  3259730800,
  3345764771,
  3516065817,
  3600352804,
  4094571909,
  275423344,
  430227734,
  506948616,
  659060556,
  883997877,
  958139571,
  1322822218,
  1537002063,
  1747873779,
  1955562222,
  2024104815,
  2227730452,
  2361852424,
  2428436474,
  2756734187,
  3204031479,
  3329325298
]);
function sha256(data) {
  const H = new Uint32Array([
    1779033703,
    3144134277,
    1013904242,
    2773480762,
    1359893119,
    2600822924,
    528734635,
    1541459225
  ]);
  const W = new Uint32Array(64);
  const compress = (words) => {
    for (let t = 0; t < 16; t++) W[t] = words(t);
    for (let t = 16; t < 64; t++) {
      const w15 = W[t - 15];
      const w2 = W[t - 2];
      const s0 = (w15 >>> 7 | w15 << 25) ^ (w15 >>> 18 | w15 << 14) ^ w15 >>> 3 | 0;
      const s1 = (w2 >>> 17 | w2 << 15) ^ (w2 >>> 19 | w2 << 13) ^ w2 >>> 10 | 0;
      W[t] = W[t - 16] + s0 + W[t - 7] + s1 | 0;
    }
    let a = H[0], b = H[1], c = H[2], d = H[3];
    let e = H[4], f = H[5], g = H[6], h = H[7];
    for (let t = 0; t < 64; t++) {
      const S1 = (e >>> 6 | e << 26) ^ (e >>> 11 | e << 21) ^ (e >>> 25 | e << 7);
      const ch = e & f ^ ~e & g;
      const t1 = h + S1 + ch + K[t] + W[t] | 0;
      const S0 = (a >>> 2 | a << 30) ^ (a >>> 13 | a << 19) ^ (a >>> 22 | a << 10);
      const maj = a & b ^ a & c ^ b & c;
      const t2 = S0 + maj | 0;
      h = g;
      g = f;
      f = e;
      e = d + t1 | 0;
      d = c;
      c = b;
      b = a;
      a = t1 + t2 | 0;
    }
    H[0] = H[0] + a | 0;
    H[1] = H[1] + b | 0;
    H[2] = H[2] + c | 0;
    H[3] = H[3] + d | 0;
    H[4] = H[4] + e | 0;
    H[5] = H[5] + f | 0;
    H[6] = H[6] + g | 0;
    H[7] = H[7] + h | 0;
  };
  const n = data.byteLength;
  const view = new DataView(data);
  const full = n - n % 64;
  for (let off = 0; off < full; off += 64) {
    const base = off;
    compress((i) => view.getUint32(base + i * 4));
  }
  const rem = n - full;
  const padded = new Uint8Array(rem < 56 ? 64 : 128);
  padded.set(new Uint8Array(data, full, rem));
  padded[rem] = 128;
  const pv = new DataView(padded.buffer);
  pv.setUint32(padded.length - 8, Math.floor(n / 536870912));
  pv.setUint32(padded.length - 4, n << 3 >>> 0);
  for (let off = 0; off < padded.length; off += 64) {
    const base = off;
    compress((i) => pv.getUint32(base + i * 4));
  }
  const out = new DataView(new ArrayBuffer(32));
  for (let i = 0; i < 8; i++) out.setUint32(i * 4, H[i]);
  return out.buffer;
}

// src/worker/render-pipeline.ts
function renderPageToRgba(pdfium2, docHandle, pageIndex, width, height, options = DEFAULT_RENDER_OPTIONS, formHandle) {
  const { fn } = pdfium2;
  let page = 0;
  let bitmap = 0;
  try {
    page = fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      const errCode = fn._FPDF_GetLastError();
      throw new Error(
        `Failed to load page ${pageIndex}: pdfium error ${errCode}`
      );
    }
    bitmap = fn._FPDFBitmap_CreateEx(width, height, 4, 0, 0);
    if (bitmap === 0) {
      throw new Error(
        `Failed to create bitmap (${width}x${height}): out of memory`
      );
    }
    fn._FPDFBitmap_FillRect(bitmap, 0, 0, width, height, options.backgroundColor);
    const flags = options.flags & ~16;
    fn._FPDF_RenderPageBitmap(
      bitmap,
      page,
      0,
      0,
      width,
      height,
      options.rotation,
      flags
    );
    if (formHandle && formHandle !== 0) {
      fn._FORM_OnAfterLoadPage(page, formHandle);
      fn._FPDF_FFLDraw(formHandle, bitmap, page, 0, 0, width, height, options.rotation, flags);
      fn._FORM_OnBeforeClosePage(page, formHandle);
    } else {
      fn._lector_render_form_widgets(docHandle, bitmap, page, 0, 0, width, height, options.rotation, flags);
    }
    const bufferPtr = fn._FPDFBitmap_GetBuffer(bitmap);
    const stride = fn._FPDFBitmap_GetStride(bitmap);
    const totalBytes = stride * height;
    const pixelsCopy = new Uint8Array(totalBytes);
    pixelsCopy.set(pdfium2.memory.heapView(bufferPtr, totalBytes));
    const pixels32 = new Uint32Array(pixelsCopy.buffer);
    for (let i = 0; i < pixels32.length; i++) {
      const v = pixels32[i];
      pixels32[i] = v & 4278255360 | (v & 255) << 16 | (v & 16711680) >>> 16;
    }
    return { width, height, rgba: pixelsCopy };
  } finally {
    if (bitmap !== 0) fn._FPDFBitmap_Destroy(bitmap);
    if (page !== 0) fn._FPDF_ClosePage(page);
  }
}
async function renderPageTileToImageBitmap(pdfium2, docHandle, pageIndex, tileX, tileY, tileW, tileH, fullW, fullH, options = DEFAULT_RENDER_OPTIONS, formHandle) {
  const { fn } = pdfium2;
  let page = 0;
  let bitmap = 0;
  try {
    page = fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}: pdfium error ${fn._FPDF_GetLastError()}`);
    }
    bitmap = fn._FPDFBitmap_CreateEx(tileW, tileH, 4, 0, 0);
    if (bitmap === 0) {
      throw new Error(`Failed to create tile bitmap (${tileW}x${tileH}): out of memory`);
    }
    fn._FPDFBitmap_FillRect(bitmap, 0, 0, tileW, tileH, options.backgroundColor);
    const flags = options.flags & ~16;
    fn._FPDF_RenderPageBitmap(
      bitmap,
      page,
      -tileX,
      -tileY,
      fullW,
      fullH,
      options.rotation,
      flags
    );
    if (formHandle && formHandle !== 0) {
      fn._FORM_OnAfterLoadPage(page, formHandle);
      fn._FPDF_FFLDraw(formHandle, bitmap, page, -tileX, -tileY, fullW, fullH, options.rotation, flags);
      fn._FORM_OnBeforeClosePage(page, formHandle);
    } else {
      fn._lector_render_form_widgets(docHandle, bitmap, page, -tileX, -tileY, fullW, fullH, options.rotation, flags);
    }
    const bufferPtr = fn._FPDFBitmap_GetBuffer(bitmap);
    const stride = fn._FPDFBitmap_GetStride(bitmap);
    const totalBytes = stride * tileH;
    const pixelsCopy = new Uint8Array(totalBytes);
    pixelsCopy.set(pdfium2.memory.heapView(bufferPtr, totalBytes));
    const pixels32 = new Uint32Array(pixelsCopy.buffer);
    for (let i = 0; i < pixels32.length; i++) {
      const v = pixels32[i];
      pixels32[i] = v & 4278255360 | (v & 255) << 16 | (v & 16711680) >>> 16;
    }
    const rgbaBytes = new Uint8ClampedArray(pixelsCopy.buffer);
    const imageData = new ImageData(rgbaBytes, tileW, tileH);
    return await createImageBitmap(imageData);
  } finally {
    if (bitmap !== 0) fn._FPDFBitmap_Destroy(bitmap);
    if (page !== 0) fn._FPDF_ClosePage(page);
  }
}
async function renderPageToImageBitmap(pdfium2, docHandle, pageIndex, width, height, options = DEFAULT_RENDER_OPTIONS, formHandle) {
  const { fn } = pdfium2;
  let page = 0;
  let bitmap = 0;
  try {
    page = fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      const errCode = fn._FPDF_GetLastError();
      throw new Error(
        `Failed to load page ${pageIndex}: pdfium error ${errCode}`
      );
    }
    bitmap = fn._FPDFBitmap_CreateEx(
      width,
      height,
      4,
      // BGRA
      0,
      0
    );
    if (bitmap === 0) {
      throw new Error(
        `Failed to create bitmap (${width}x${height}): out of memory`
      );
    }
    fn._FPDFBitmap_FillRect(
      bitmap,
      0,
      0,
      width,
      height,
      options.backgroundColor
    );
    const flags = options.flags & ~16;
    fn._FPDF_RenderPageBitmap(
      bitmap,
      page,
      0,
      // start_x
      0,
      // start_y
      width,
      // size_x
      height,
      // size_y
      options.rotation,
      flags
    );
    if (formHandle && formHandle !== 0) {
      fn._FORM_OnAfterLoadPage(page, formHandle);
      fn._FPDF_FFLDraw(formHandle, bitmap, page, 0, 0, width, height, options.rotation, flags);
      fn._FORM_OnBeforeClosePage(page, formHandle);
    } else {
      fn._lector_render_form_widgets(docHandle, bitmap, page, 0, 0, width, height, options.rotation, flags);
    }
    const bufferPtr = fn._FPDFBitmap_GetBuffer(bitmap);
    const stride = fn._FPDFBitmap_GetStride(bitmap);
    const totalBytes = stride * height;
    const pixelsCopy = new Uint8Array(totalBytes);
    const heapView = pdfium2.memory.heapView(bufferPtr, totalBytes);
    pixelsCopy.set(heapView);
    const pixels = new Uint32Array(pixelsCopy.buffer);
    const pixelCount = pixels.length;
    for (let i = 0; i < pixelCount; i++) {
      const v = pixels[i];
      pixels[i] = v & 4278255360 | (v & 255) << 16 | (v & 16711680) >>> 16;
    }
    const rgbaBytes = new Uint8ClampedArray(pixelsCopy.buffer);
    const imageData = new ImageData(rgbaBytes, width, height);
    const imageBitmap = await createImageBitmap(imageData);
    return imageBitmap;
  } finally {
    if (bitmap !== 0) {
      fn._FPDFBitmap_Destroy(bitmap);
    }
    if (page !== 0) {
      fn._FPDF_ClosePage(page);
    }
  }
}

// src/utils/uuid.ts
function uuid() {
  const c = crypto;
  if (c.randomUUID) return c.randomUUID();
  const b = c.getRandomValues(new Uint8Array(16));
  b[6] = b[6] & 15 | 64;
  b[8] = b[8] & 63 | 128;
  let h = "";
  for (let i = 0; i < 16; i++) h += b[i].toString(16).padStart(2, "0");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}

// src/worker/annotation-ops.ts
import {
  FpdfAnnotColorType,
  FpdfAnnotSubtype,
  FpdfFormFieldType,
  FS_POINTF_SIZE,
  FS_QUADPOINTSF_SIZE,
  FS_RECTF_SIZE
} from "@truespar/lector-pdfium-wasm";
var CALLOUT_DATA_KEY = "LectorCallout";
var IMAGE_DATA_KEY = "LectorImage";
var IMAGE_SIZE_KEY = "LectorImageSize";
var IMAGE_NATURAL_KEY = "LectorImageNatural";
function readAnnotStringValue(pdfium2, annot, key) {
  var _stack = [];
  try {
    const keyAlloc = __using(_stack, pdfium2.memory.toByteString(key));
    const len = pdfium2.fn._FPDFAnnot_GetStringValue(
      annot,
      keyAlloc.ptr,
      0,
      0
    );
    if (len <= 2) return void 0;
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    pdfium2.fn._FPDFAnnot_GetStringValue(annot, keyAlloc.ptr, buf.ptr, len);
    return pdfium2.memory.fromWideString(buf.ptr) || void 0;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readAnnotColor(pdfium2, annot, colorType) {
  var _stack = [];
  try {
    const rAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const gAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const bAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const aAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const ok = pdfium2.fn._FPDFAnnot_GetColor(
      annot,
      colorType,
      rAlloc.ptr,
      gAlloc.ptr,
      bAlloc.ptr,
      aAlloc.ptr
    );
    if (ok === 0) return void 0;
    return {
      r: pdfium2.module.getValue(rAlloc.ptr, "i32"),
      g: pdfium2.module.getValue(gAlloc.ptr, "i32"),
      b: pdfium2.module.getValue(bAlloc.ptr, "i32"),
      a: pdfium2.module.getValue(aAlloc.ptr, "i32")
    };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readAnnotBorder(pdfium2, annot) {
  var _stack = [];
  try {
    const hAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const vAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const wAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const ok = pdfium2.fn._FPDFAnnot_GetBorder(annot, hAlloc.ptr, vAlloc.ptr, wAlloc.ptr);
    if (ok === 0) return void 0;
    return {
      horizontalRadius: pdfium2.module.getValue(hAlloc.ptr, "float"),
      verticalRadius: pdfium2.module.getValue(vAlloc.ptr, "float"),
      width: pdfium2.module.getValue(wAlloc.ptr, "float")
    };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readMarkupQuadPoints(pdfium2, annot) {
  var _stack = [];
  try {
    const count = pdfium2.fn._FPDFAnnot_CountAttachmentPoints(annot);
    if (count <= 0) return void 0;
    const quadPoints = [];
    const qpAlloc = __using(_stack, pdfium2.memory.alloc(FS_QUADPOINTSF_SIZE));
    for (let i = 0; i < count; i++) {
      const ok = pdfium2.fn._FPDFAnnot_GetAttachmentPoints(annot, i, qpAlloc.ptr);
      if (ok !== 0) {
        quadPoints.push(pdfium2.memory.readQuadPointsF(qpAlloc.ptr));
      }
    }
    return quadPoints.length > 0 ? quadPoints : void 0;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readInkStrokes(pdfium2, annot) {
  const strokeCount = pdfium2.fn._FPDFAnnot_GetInkListCount(annot);
  if (strokeCount <= 0) return void 0;
  const strokes = [];
  for (let s = 0; s < strokeCount; s++) {
    var _stack = [];
    try {
      const pointCount = pdfium2.fn._FPDFAnnot_GetInkListPath(
        annot,
        s,
        0,
        0
      );
      if (pointCount <= 0) {
        strokes.push([]);
        continue;
      }
      const pointsAlloc = __using(_stack, pdfium2.memory.alloc(pointCount * FS_POINTF_SIZE));
      pdfium2.fn._FPDFAnnot_GetInkListPath(annot, s, pointsAlloc.ptr, pointCount);
      const points = [];
      for (let p = 0; p < pointCount; p++) {
        const offset = pointsAlloc.ptr + p * FS_POINTF_SIZE;
        const pt = pdfium2.memory.readPointF(offset);
        points.push({ x: pt.x, y: pt.y });
      }
      strokes.push(points);
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  }
  return strokes.length > 0 ? strokes : void 0;
}
function readLineEndpoints(pdfium2, annot) {
  var _stack = [];
  try {
    const startAlloc = __using(_stack, pdfium2.memory.alloc(FS_POINTF_SIZE));
    const endAlloc = __using(_stack, pdfium2.memory.alloc(FS_POINTF_SIZE));
    const ok = pdfium2.fn._FPDFAnnot_GetLine(annot, startAlloc.ptr, endAlloc.ptr);
    if (ok === 0) return void 0;
    const start = pdfium2.memory.readPointF(startAlloc.ptr);
    const end = pdfium2.memory.readPointF(endAlloc.ptr);
    return {
      start: { x: start.x, y: start.y },
      end: { x: end.x, y: end.y }
    };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readFormWideString(pdfium2, getter) {
  var _stack = [];
  try {
    const len = getter(0, 0);
    if (len <= 2) return "";
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    getter(buf.ptr, len);
    return pdfium2.memory.fromWideString(buf.ptr);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readWidgetData(pdfium2, formHandle, annot, annotIndex) {
  const fieldType = pdfium2.fn._FPDFAnnot_GetFormFieldType(formHandle, annot);
  const fieldName = readFormWideString(
    pdfium2,
    (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetFormFieldName(formHandle, annot, buffer, buflen)
  );
  const fieldValue = readFormWideString(
    pdfium2,
    (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetFormFieldValue(formHandle, annot, buffer, buflen)
  );
  const isCheckable = fieldType === FpdfFormFieldType.CHECKBOX || fieldType === FpdfFormFieldType.RADIOBUTTON;
  const exportValue = isCheckable ? readFormWideString(
    pdfium2,
    (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetFormFieldExportValue(formHandle, annot, buffer, buflen)
  ) : void 0;
  const isChecked = isCheckable ? pdfium2.fn._FPDFAnnot_IsChecked(formHandle, annot) !== 0 : void 0;
  const hasOptions = fieldType === FpdfFormFieldType.COMBOBOX || fieldType === FpdfFormFieldType.LISTBOX;
  let options;
  if (hasOptions) {
    const count = pdfium2.fn._FPDFAnnot_GetOptionCount(formHandle, annot);
    if (count > 0) {
      options = [];
      for (let i = 0; i < count; i++) {
        const label = readFormWideString(
          pdfium2,
          (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetOptionLabel(formHandle, annot, i, buffer, buflen)
        );
        const selected = pdfium2.fn._FPDFAnnot_IsOptionSelected(formHandle, annot, i) !== 0;
        options.push({ label, selected, index: i });
      }
    }
  }
  const fieldFlags = pdfium2.fn._FPDFAnnot_GetFormFieldFlags(formHandle, annot);
  return { fieldType, fieldName, fieldValue, exportValue, isChecked, options, annotIndex, fieldFlags };
}
function readAnnotation(pdfium2, annot, pageIndex, formHandle, annotIndex) {
  var _stack = [];
  try {
    const subtype = pdfium2.fn._FPDFAnnot_GetSubtype(annot);
    const flags = pdfium2.fn._FPDFAnnot_GetFlags(annot);
    const rectAlloc = __using(_stack, pdfium2.memory.alloc(FS_RECTF_SIZE));
    pdfium2.fn._FPDFAnnot_GetRect(annot, rectAlloc.ptr);
    const rect = pdfium2.memory.readRectF(rectAlloc.ptr);
    const color = readAnnotColor(pdfium2, annot, FpdfAnnotColorType.COLOR);
    const interiorColor = readAnnotColor(pdfium2, annot, FpdfAnnotColorType.INTERIOR_COLOR);
    const border = readAnnotBorder(pdfium2, annot);
    const contents = readAnnotStringValue(pdfium2, annot, "Contents");
    const author = readAnnotStringValue(pdfium2, annot, "T");
    const modifiedDate = readAnnotStringValue(pdfium2, annot, "M");
    const createdDate = readAnnotStringValue(pdfium2, annot, "CreationDate");
    const isMarkup = subtype === FpdfAnnotSubtype.HIGHLIGHT || subtype === FpdfAnnotSubtype.UNDERLINE || subtype === FpdfAnnotSubtype.SQUIGGLY || subtype === FpdfAnnotSubtype.STRIKEOUT;
    const markupQuadPoints = isMarkup ? readMarkupQuadPoints(pdfium2, annot) : void 0;
    const markup = markupQuadPoints !== void 0 ? { quadPoints: markupQuadPoints } : void 0;
    const inkStrokes = subtype === FpdfAnnotSubtype.INK ? readInkStrokes(pdfium2, annot) : void 0;
    const ink = inkStrokes !== void 0 ? { strokes: inkStrokes } : void 0;
    const line = subtype === FpdfAnnotSubtype.LINE ? readLineEndpoints(pdfium2, annot) : void 0;
    const freeText = subtype === FpdfAnnotSubtype.FREETEXT ? { text: contents ?? "", fontSize: 0 } : void 0;
    const tag = readAnnotStringValue(pdfium2, annot, "Subj") || void 0;
    const noteIcon = subtype === FpdfAnnotSubtype.TEXT ? readAnnotStringValue(pdfium2, annot, "Name") || void 0 : void 0;
    const opacityStr = readAnnotStringValue(pdfium2, annot, "CA");
    const opacity = opacityStr !== void 0 ? parseFloat(opacityStr) : void 0;
    const stamp = subtype === FpdfAnnotSubtype.STAMP ? { name: tag ?? "Draft" } : void 0;
    const redaction = subtype === FpdfAnnotSubtype.REDACT || tag === "redaction" ? {
      reason: readAnnotStringValue(pdfium2, annot, "IT") || void 0,
      overlayText: readAnnotStringValue(pdfium2, annot, "OverlayText") || contents || void 0
    } : void 0;
    let callout;
    if (subtype === FpdfAnnotSubtype.FREETEXT && tag === "callout") {
      const raw = readAnnotStringValue(pdfium2, annot, CALLOUT_DATA_KEY);
      if (raw) {
        try {
          const parsed = JSON.parse(raw);
          if (parsed && typeof parsed === "object" && parsed.endpoint && typeof parsed.endpoint.x === "number" && typeof parsed.endpoint.y === "number") {
            callout = {
              endpoint: { x: parsed.endpoint.x, y: parsed.endpoint.y },
              ...parsed.knee && typeof parsed.knee.x === "number" && typeof parsed.knee.y === "number" ? { knee: { x: parsed.knee.x, y: parsed.knee.y } } : {},
              ...parsed.lineEnding ? { lineEnding: parsed.lineEnding } : {}
            };
          }
        } catch {
        }
      }
    }
    let image;
    if (subtype === FpdfAnnotSubtype.STAMP && tag === "image") {
      const dataUri = readAnnotStringValue(pdfium2, annot, IMAGE_DATA_KEY);
      if (dataUri) {
        const sizeStr = readAnnotStringValue(pdfium2, annot, IMAGE_SIZE_KEY);
        const naturalStr = readAnnotStringValue(pdfium2, annot, IMAGE_NATURAL_KEY);
        const parseSize = (s) => {
          if (!s) return null;
          const m = s.match(/^(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)$/);
          if (!m) return null;
          return { w: parseFloat(m[1]), h: parseFloat(m[2]) };
        };
        const size = parseSize(sizeStr) ?? { w: rect.right - rect.left, h: rect.top - rect.bottom };
        const natural = parseSize(naturalStr);
        image = {
          imageRef: dataUri,
          width: size.w,
          height: size.h,
          ...natural ? { naturalWidth: natural.w, naturalHeight: natural.h } : {}
        };
      }
    }
    let widget;
    if (subtype === FpdfAnnotSubtype.WIDGET && formHandle && formHandle !== 0) {
      widget = readWidgetData(pdfium2, formHandle, annot, annotIndex ?? -1);
    }
    return {
      id: uuid(),
      pageIndex,
      subtype,
      rect: {
        left: rect.left,
        top: rect.top,
        right: rect.right,
        bottom: rect.bottom
      },
      flags,
      color,
      interiorColor,
      border,
      contents,
      author,
      modifiedDate,
      createdDate,
      tag,
      noteIcon,
      ...opacity !== void 0 ? { opacity } : {},
      ...stamp ? { stamp } : {},
      ...redaction ? { redaction } : {},
      ...callout ? { callout } : {},
      ...image ? { image } : {},
      ...widget ? { widget } : {},
      markup,
      ink,
      line,
      freeText
    };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function applyAnnotationProperties(pdfium2, annot, data) {
  if (data.rect !== void 0) {
    var _stack = [];
    try {
      const rectAlloc = __using(_stack, pdfium2.memory.alloc(FS_RECTF_SIZE));
      pdfium2.memory.writeRectF(rectAlloc.ptr, {
        left: data.rect.left,
        top: data.rect.top,
        right: data.rect.right,
        bottom: data.rect.bottom
      });
      pdfium2.fn._FPDFAnnot_SetRect(annot, rectAlloc.ptr);
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  }
  if (data.flags !== void 0) {
    pdfium2.fn._FPDFAnnot_SetFlags(annot, data.flags);
  }
  if (data.color !== void 0) {
    pdfium2.fn._FPDFAnnot_SetColor(
      annot,
      FpdfAnnotColorType.COLOR,
      data.color.r,
      data.color.g,
      data.color.b,
      data.color.a
    );
  }
  if (data.interiorColor !== void 0) {
    pdfium2.fn._FPDFAnnot_SetColor(
      annot,
      FpdfAnnotColorType.INTERIOR_COLOR,
      data.interiorColor.r,
      data.interiorColor.g,
      data.interiorColor.b,
      data.interiorColor.a
    );
  }
  if (data.border !== void 0) {
    pdfium2.fn._FPDFAnnot_SetBorder(
      annot,
      data.border.horizontalRadius,
      data.border.verticalRadius,
      data.border.width
    );
  }
  if (data.contents !== void 0) {
    var _stack2 = [];
    try {
      const keyAlloc = __using(_stack2, pdfium2.memory.toByteString("Contents"));
      const valAlloc = __using(_stack2, pdfium2.memory.toWideString(data.contents));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
    } catch (_2) {
      var _error2 = _2, _hasError2 = true;
    } finally {
      __callDispose(_stack2, _error2, _hasError2);
    }
  }
  if (data.author !== void 0) {
    var _stack3 = [];
    try {
      const keyAlloc = __using(_stack3, pdfium2.memory.toByteString("T"));
      const valAlloc = __using(_stack3, pdfium2.memory.toWideString(data.author));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
    } catch (_3) {
      var _error3 = _3, _hasError3 = true;
    } finally {
      __callDispose(_stack3, _error3, _hasError3);
    }
  }
  if (data.tag !== void 0) {
    var _stack4 = [];
    try {
      const keyAlloc = __using(_stack4, pdfium2.memory.toByteString("Subj"));
      const valAlloc = __using(_stack4, pdfium2.memory.toWideString(data.tag));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
    } catch (_4) {
      var _error4 = _4, _hasError4 = true;
    } finally {
      __callDispose(_stack4, _error4, _hasError4);
    }
  }
  if (data.modifiedDate !== void 0) {
    var _stack5 = [];
    try {
      const keyAlloc = __using(_stack5, pdfium2.memory.toByteString("M"));
      const valAlloc = __using(_stack5, pdfium2.memory.toWideString(data.modifiedDate));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
    } catch (_5) {
      var _error5 = _5, _hasError5 = true;
    } finally {
      __callDispose(_stack5, _error5, _hasError5);
    }
  }
  if (data.createdDate !== void 0) {
    var _stack6 = [];
    try {
      const keyAlloc = __using(_stack6, pdfium2.memory.toByteString("CreationDate"));
      const valAlloc = __using(_stack6, pdfium2.memory.toWideString(data.createdDate));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
    } catch (_6) {
      var _error6 = _6, _hasError6 = true;
    } finally {
      __callDispose(_stack6, _error6, _hasError6);
    }
  }
  if (data.noteIcon !== void 0) {
    var _stack7 = [];
    try {
      const keyAlloc = __using(_stack7, pdfium2.memory.toByteString("Name"));
      const valAlloc = __using(_stack7, pdfium2.memory.toWideString(data.noteIcon));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
    } catch (_7) {
      var _error7 = _7, _hasError7 = true;
    } finally {
      __callDispose(_stack7, _error7, _hasError7);
    }
  }
  if (data.opacity !== void 0) {
    var _stack8 = [];
    try {
      const keyAlloc = __using(_stack8, pdfium2.memory.toByteString("CA"));
      const valAlloc = __using(_stack8, pdfium2.memory.toWideString(String(data.opacity)));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
    } catch (_8) {
      var _error8 = _8, _hasError8 = true;
    } finally {
      __callDispose(_stack8, _error8, _hasError8);
    }
  }
  if (data.markup?.quadPoints !== void 0) {
    var _stack9 = [];
    try {
      const qpAlloc = __using(_stack9, pdfium2.memory.alloc(FS_QUADPOINTSF_SIZE));
      for (let i = 0; i < data.markup.quadPoints.length; i++) {
        const qp = data.markup.quadPoints[i];
        pdfium2.memory.writeQuadPointsF(qpAlloc.ptr, qp);
        pdfium2.fn._FPDFAnnot_AppendAttachmentPoints(annot, qpAlloc.ptr);
      }
    } catch (_9) {
      var _error9 = _9, _hasError9 = true;
    } finally {
      __callDispose(_stack9, _error9, _hasError9);
    }
  }
  if (data.ink?.strokes !== void 0) {
    pdfium2.fn._FPDFAnnot_RemoveInkList(annot);
    for (const stroke of data.ink.strokes) {
      var _stack10 = [];
      try {
        if (stroke.length === 0) continue;
        const pointsAlloc = __using(_stack10, pdfium2.memory.alloc(stroke.length * FS_POINTF_SIZE));
        for (let p = 0; p < stroke.length; p++) {
          const pt = stroke[p];
          const offset = pointsAlloc.ptr + p * FS_POINTF_SIZE;
          pdfium2.memory.writePointF(offset, { x: pt.x, y: pt.y });
        }
        pdfium2.fn._FPDFAnnot_AddInkStroke(annot, pointsAlloc.ptr, stroke.length);
      } catch (_10) {
        var _error10 = _10, _hasError10 = true;
      } finally {
        __callDispose(_stack10, _error10, _hasError10);
      }
    }
  }
  if (data.callout !== void 0) {
    var _stack11 = [];
    try {
      const keyAlloc = __using(_stack11, pdfium2.memory.toByteString(CALLOUT_DATA_KEY));
      const valAlloc = __using(_stack11, pdfium2.memory.toWideString(JSON.stringify(data.callout)));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
      const itKeyAlloc = __using(_stack11, pdfium2.memory.toByteString("IT"));
      const itValAlloc = __using(_stack11, pdfium2.memory.toWideString("FreeTextCallout"));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, itKeyAlloc.ptr, itValAlloc.ptr);
    } catch (_11) {
      var _error11 = _11, _hasError11 = true;
    } finally {
      __callDispose(_stack11, _error11, _hasError11);
    }
  }
  if (data.image !== void 0) {
    var _stack13 = [];
    try {
      const keyAlloc = __using(_stack13, pdfium2.memory.toByteString(IMAGE_DATA_KEY));
      const valAlloc = __using(_stack13, pdfium2.memory.toWideString(data.image.imageRef));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, keyAlloc.ptr, valAlloc.ptr);
      const sizeKey = __using(_stack13, pdfium2.memory.toByteString(IMAGE_SIZE_KEY));
      const sizeVal = __using(_stack13, pdfium2.memory.toWideString(`${data.image.width}x${data.image.height}`));
      pdfium2.fn._FPDFAnnot_SetStringValue(annot, sizeKey.ptr, sizeVal.ptr);
      if (data.image.naturalWidth !== void 0 && data.image.naturalHeight !== void 0) {
        var _stack12 = [];
        try {
          const natKey = __using(_stack12, pdfium2.memory.toByteString(IMAGE_NATURAL_KEY));
          const natVal = __using(_stack12, pdfium2.memory.toWideString(`${data.image.naturalWidth}x${data.image.naturalHeight}`));
          pdfium2.fn._FPDFAnnot_SetStringValue(annot, natKey.ptr, natVal.ptr);
        } catch (_12) {
          var _error12 = _12, _hasError12 = true;
        } finally {
          __callDispose(_stack12, _error12, _hasError12);
        }
      }
    } catch (_13) {
      var _error13 = _13, _hasError13 = true;
    } finally {
      __callDispose(_stack13, _error13, _hasError13);
    }
  }
}
function readPageAnnotations(pdfium2, docHandle, pageIndex, formHandle) {
  let page = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    const count = pdfium2.fn._FPDFPage_GetAnnotCount(page);
    const annotations = [];
    for (let i = 0; i < count; i++) {
      let annot = 0;
      try {
        annot = pdfium2.fn._FPDFPage_GetAnnot(page, i);
        if (annot === 0) continue;
        annotations.push(readAnnotation(pdfium2, annot, pageIndex, formHandle, i));
      } finally {
        if (annot !== 0) {
          pdfium2.fn._FPDFPage_CloseAnnot(annot);
        }
      }
    }
    return annotations;
  } finally {
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}
function createAnnotation(pdfium2, docHandle, pageIndex, data) {
  if (data.subtype === void 0) {
    throw new Error("Annotation subtype is required for creation");
  }
  let page = 0;
  let annot = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    annot = pdfium2.fn._FPDFPage_CreateAnnot(page, data.subtype);
    if (annot === 0) {
      throw new Error(`Failed to create annotation with subtype ${data.subtype}`);
    }
    applyAnnotationProperties(pdfium2, annot, data);
    pdfium2.fn._FPDFPage_GenerateContent(page);
    const result = readAnnotation(pdfium2, annot, pageIndex);
    return result;
  } finally {
    if (annot !== 0) {
      pdfium2.fn._FPDFPage_CloseAnnot(annot);
    }
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}
function updateAnnotation(pdfium2, docHandle, pageIndex, annotIndex, patch) {
  let page = 0;
  let annot = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    annot = pdfium2.fn._FPDFPage_GetAnnot(page, annotIndex);
    if (annot === 0) {
      throw new Error(`Annotation at index ${annotIndex} not found on page ${pageIndex}`);
    }
    applyAnnotationProperties(pdfium2, annot, patch);
    pdfium2.fn._FPDFPage_GenerateContent(page);
    const result = readAnnotation(pdfium2, annot, pageIndex);
    return result;
  } finally {
    if (annot !== 0) {
      pdfium2.fn._FPDFPage_CloseAnnot(annot);
    }
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}
function deleteAnnotation(pdfium2, docHandle, pageIndex, annotIndex) {
  let page = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    const ok = pdfium2.fn._FPDFPage_RemoveAnnot(page, annotIndex);
    if (ok === 0) {
      throw new Error(`Failed to remove annotation at index ${annotIndex} on page ${pageIndex}`);
    }
  } finally {
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}

// src/worker/form-ops.ts
import { FpdfAnnotSubtype as FpdfAnnotSubtype2, FpdfFormFieldType as FpdfFormFieldType2 } from "@truespar/lector-pdfium-wasm";
function readFormWideString2(pdfium2, getter) {
  var _stack = [];
  try {
    const len = getter(0, 0);
    if (len <= 2) return "";
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    getter(buf.ptr, len);
    return pdfium2.memory.fromWideString(buf.ptr);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readFieldName(pdfium2, formHandle, annot) {
  return readFormWideString2(
    pdfium2,
    (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetFormFieldName(formHandle, annot, buffer, buflen)
  );
}
function readFieldValue(pdfium2, formHandle, annot) {
  return readFormWideString2(
    pdfium2,
    (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetFormFieldValue(formHandle, annot, buffer, buflen)
  );
}
function readExportValue(pdfium2, formHandle, annot) {
  return readFormWideString2(
    pdfium2,
    (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetFormFieldExportValue(formHandle, annot, buffer, buflen)
  );
}
function readOptions(pdfium2, formHandle, annot) {
  const count = pdfium2.fn._FPDFAnnot_GetOptionCount(formHandle, annot);
  if (count <= 0) return [];
  const options = [];
  for (let i = 0; i < count; i++) {
    const label = readFormWideString2(
      pdfium2,
      (buffer, buflen) => pdfium2.fn._FPDFAnnot_GetOptionLabel(formHandle, annot, i, buffer, buflen)
    );
    const selected = pdfium2.fn._FPDFAnnot_IsOptionSelected(formHandle, annot, i) !== 0;
    options.push({ label, selected, index: i });
  }
  return options;
}
function readWidget(pdfium2, formHandle, annot, annotIndex) {
  const fieldType = pdfium2.fn._FPDFAnnot_GetFormFieldType(formHandle, annot);
  const fieldName = readFieldName(pdfium2, formHandle, annot);
  const fieldValue = readFieldValue(pdfium2, formHandle, annot);
  const fieldFlags = pdfium2.fn._FPDFAnnot_GetFormFieldFlags(formHandle, annot);
  const isCheckable = fieldType === FpdfFormFieldType2.CHECKBOX || fieldType === FpdfFormFieldType2.RADIOBUTTON;
  const exportValue = isCheckable ? readExportValue(pdfium2, formHandle, annot) : void 0;
  const isChecked = isCheckable ? pdfium2.fn._FPDFAnnot_IsChecked(formHandle, annot) !== 0 : void 0;
  const hasOptions = fieldType === FpdfFormFieldType2.COMBOBOX || fieldType === FpdfFormFieldType2.LISTBOX;
  const options = hasOptions ? readOptions(pdfium2, formHandle, annot) : void 0;
  return {
    fieldType,
    fieldName,
    fieldValue,
    exportValue,
    isChecked,
    options,
    annotIndex,
    fieldFlags
  };
}
function readPageFormFields(pdfium2, docHandle, formHandle, pageIndex) {
  let page = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    const annotCount = pdfium2.fn._FPDFPage_GetAnnotCount(page);
    const fields = [];
    for (let i = 0; i < annotCount; i++) {
      let annot = 0;
      try {
        annot = pdfium2.fn._FPDFPage_GetAnnot(page, i);
        if (annot === 0) continue;
        const subtype = pdfium2.fn._FPDFAnnot_GetSubtype(annot);
        if (subtype !== FpdfAnnotSubtype2.WIDGET) continue;
        fields.push(readWidget(pdfium2, formHandle, annot, i));
      } finally {
        if (annot !== 0) {
          pdfium2.fn._FPDFPage_CloseAnnot(annot);
        }
      }
    }
    return fields;
  } finally {
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}
function setFormFieldValue(pdfium2, docHandle, formHandle, pageIndex, fieldName, value) {
  let page = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    pdfium2.fn._FORM_OnAfterLoadPage(page, formHandle);
    const annotCount = pdfium2.fn._FPDFPage_GetAnnotCount(page);
    for (let i = 0; i < annotCount; i++) {
      let annot = 0;
      try {
        var _stack = [];
        try {
          annot = pdfium2.fn._FPDFPage_GetAnnot(page, i);
          if (annot === 0) continue;
          const subtype = pdfium2.fn._FPDFAnnot_GetSubtype(annot);
          if (subtype !== FpdfAnnotSubtype2.WIDGET) continue;
          const name = readFieldName(pdfium2, formHandle, annot);
          if (name !== fieldName) continue;
          pdfium2.fn._FORM_SetFocusedAnnot(formHandle, annot);
          pdfium2.fn._FORM_SelectAllText(formHandle, page);
          const valueAlloc = __using(_stack, pdfium2.memory.toWideString(value));
          pdfium2.fn._FORM_ReplaceSelection(formHandle, page, valueAlloc.ptr);
          pdfium2.fn._FORM_ForceToKillFocus(formHandle);
          return;
        } catch (_) {
          var _error = _, _hasError = true;
        } finally {
          __callDispose(_stack, _error, _hasError);
        }
      } finally {
        if (annot !== 0) {
          pdfium2.fn._FPDFPage_CloseAnnot(annot);
        }
      }
    }
    throw new Error(`Form field '${fieldName}' not found on page ${pageIndex}`);
  } finally {
    if (page !== 0) {
      pdfium2.fn._FORM_OnBeforeClosePage(page, formHandle);
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}
function setComboBoxByIndex(pdfium2, docHandle, formHandle, pageIndex, annotIndex, optionIndex) {
  let page = 0;
  let annot = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    pdfium2.fn._FORM_OnAfterLoadPage(page, formHandle);
    annot = pdfium2.fn._FPDFPage_GetAnnot(page, annotIndex);
    if (annot !== 0) {
      pdfium2.fn._FORM_SetFocusedAnnot(formHandle, annot);
      pdfium2.fn._FORM_SetIndexSelected(formHandle, page, optionIndex, 1);
    }
    pdfium2.fn._FORM_ForceToKillFocus(formHandle);
    pdfium2.fn._FORM_OnBeforeClosePage(page, formHandle);
  } finally {
    if (annot !== 0) pdfium2.fn._FPDFPage_CloseAnnot(annot);
    if (page !== 0) pdfium2.fn._FPDF_ClosePage(page);
  }
}
function clickFormWidget(pdfium2, docHandle, formHandle, pageIndex, pageX, pageY) {
  let page = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    pdfium2.fn._FORM_OnAfterLoadPage(page, formHandle);
    pdfium2.fn._FORM_OnLButtonDown(formHandle, page, 0, pageX, pageY);
    pdfium2.fn._FORM_OnLButtonUp(formHandle, page, 0, pageX, pageY);
    pdfium2.fn._FORM_ForceToKillFocus(formHandle);
    pdfium2.fn._FORM_OnBeforeClosePage(page, formHandle);
  } finally {
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}

// src/worker/save-ops.ts
import { FpdfAnnotSubtype as FpdfAnnotSubtype3 } from "@truespar/lector-pdfium-wasm";
var FPDF_FILEWRITE_SIZE = 8;
function saveDocumentAsCopy(pdfium2, docHandle) {
  var _stack = [];
  try {
    const chunks = [];
    let totalSize = 0;
    const writeBlockCallback = (_pThis, pData, size) => {
      const chunk = pdfium2.memory.fromHeap(pData, size);
      chunks.push(chunk);
      totalSize += size;
      return 1;
    };
    const module = pdfium2.module;
    const fileWriteAlloc = __using(_stack, pdfium2.memory.alloc(FPDF_FILEWRITE_SIZE));
    const funcPtr = module.addFunction(writeBlockCallback, "iiii");
    try {
      pdfium2.module.setValue(fileWriteAlloc.ptr, 1, "i32");
      pdfium2.module.setValue(fileWriteAlloc.ptr + 4, funcPtr, "i32");
      const ok = pdfium2.fn._FPDF_SaveAsCopy(docHandle, fileWriteAlloc.ptr, 0);
      if (ok === 0) {
        throw new Error("FPDF_SaveAsCopy failed");
      }
    } finally {
      module.removeFunction(funcPtr);
    }
    const result = new Uint8Array(totalSize);
    let offset = 0;
    for (const chunk of chunks) {
      result.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return result.buffer;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
var SUBTYPE_TO_XFDF_ELEMENT = {
  [FpdfAnnotSubtype3.TEXT]: "text",
  [FpdfAnnotSubtype3.FREETEXT]: "freetext",
  [FpdfAnnotSubtype3.LINE]: "line",
  [FpdfAnnotSubtype3.SQUARE]: "square",
  [FpdfAnnotSubtype3.CIRCLE]: "circle",
  [FpdfAnnotSubtype3.POLYGON]: "polygon",
  [FpdfAnnotSubtype3.POLYLINE]: "polyline",
  [FpdfAnnotSubtype3.HIGHLIGHT]: "highlight",
  [FpdfAnnotSubtype3.UNDERLINE]: "underline",
  [FpdfAnnotSubtype3.SQUIGGLY]: "squiggly",
  [FpdfAnnotSubtype3.STRIKEOUT]: "strikeout",
  [FpdfAnnotSubtype3.STAMP]: "stamp",
  [FpdfAnnotSubtype3.CARET]: "caret",
  [FpdfAnnotSubtype3.INK]: "ink",
  [FpdfAnnotSubtype3.FILEATTACHMENT]: "fileattachment",
  [FpdfAnnotSubtype3.SOUND]: "sound",
  [FpdfAnnotSubtype3.REDACT]: "redact"
};
function escapeXml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&apos;");
}
function colorToHex(color) {
  const r = color.r.toString(16).padStart(2, "0");
  const g = color.g.toString(16).padStart(2, "0");
  const b = color.b.toString(16).padStart(2, "0");
  return `#${r}${g}${b}`;
}
function rectToXfdf(rect) {
  return `${rect.left.toFixed(6)},${rect.bottom.toFixed(6)},${rect.right.toFixed(6)},${rect.top.toFixed(6)}`;
}
function annotationToXfdfElement(annot, indent) {
  const elementName = SUBTYPE_TO_XFDF_ELEMENT[annot.subtype];
  if (elementName === void 0) return "";
  const attrs = [];
  attrs.push(`page="${annot.pageIndex}"`);
  attrs.push(`rect="${rectToXfdf(annot.rect)}"`);
  if (annot.color !== void 0) {
    attrs.push(`color="${colorToHex(annot.color)}"`);
  }
  if (annot.interiorColor !== void 0) {
    attrs.push(`interior-color="${colorToHex(annot.interiorColor)}"`);
  }
  if (annot.flags !== void 0 && annot.flags !== 0) {
    attrs.push(`flags="${annot.flags}"`);
  }
  if (annot.border?.width !== void 0 && annot.border.width > 0) {
    attrs.push(`width="${annot.border.width.toFixed(6)}"`);
  }
  if (annot.author !== void 0) {
    attrs.push(`title="${escapeXml(annot.author)}"`);
  }
  if (annot.tag !== void 0) {
    attrs.push(`subject="${escapeXml(annot.tag)}"`);
  }
  if (annot.modifiedDate !== void 0) {
    attrs.push(`date="${escapeXml(annot.modifiedDate)}"`);
  }
  if (annot.createdDate !== void 0) {
    attrs.push(`creationdate="${escapeXml(annot.createdDate)}"`);
  }
  const children = [];
  if (annot.contents !== void 0 && annot.contents.length > 0) {
    children.push(`${indent}  <contents>${escapeXml(annot.contents)}</contents>`);
  }
  if (annot.markup?.quadPoints !== void 0 && annot.markup.quadPoints.length > 0) {
    const qpStrings = annot.markup.quadPoints.map(
      (qp) => `${qp.x1.toFixed(6)},${qp.y1.toFixed(6)},${qp.x2.toFixed(6)},${qp.y2.toFixed(6)},${qp.x3.toFixed(6)},${qp.y3.toFixed(6)},${qp.x4.toFixed(6)},${qp.y4.toFixed(6)}`
    );
    attrs.push(`coords="${qpStrings.join(",")}"`);
  }
  if (annot.ink?.strokes !== void 0 && annot.ink.strokes.length > 0) {
    for (const stroke of annot.ink.strokes) {
      if (stroke.length === 0) continue;
      const pointStr = stroke.map((pt) => `${pt.x.toFixed(6)},${pt.y.toFixed(6)}`).join(";");
      children.push(`${indent}  <inklist><gesture>${pointStr}</gesture></inklist>`);
    }
  }
  if (annot.line !== void 0) {
    const start = `${annot.line.start.x.toFixed(6)},${annot.line.start.y.toFixed(6)}`;
    const end = `${annot.line.end.x.toFixed(6)},${annot.line.end.y.toFixed(6)}`;
    attrs.push(`start="${start}"`);
    attrs.push(`end="${end}"`);
  }
  if (annot.freeText !== void 0) {
    if (annot.freeText.fontSize > 0) {
      const fontColorStr = annot.freeText.fontColor !== void 0 ? colorToHex(annot.freeText.fontColor) : "#000000";
      children.push(
        `${indent}  <defaultappearance>/${annot.freeText.fontSize} Tf ${fontColorStr}</defaultappearance>`
      );
    }
  }
  const attrStr = attrs.length > 0 ? ` ${attrs.join(" ")}` : "";
  if (children.length === 0) {
    return `${indent}<${elementName}${attrStr} />`;
  }
  const lines = [];
  lines.push(`${indent}<${elementName}${attrStr}>`);
  lines.push(...children);
  lines.push(`${indent}</${elementName}>`);
  return lines.join("\n");
}
function exportXfdf(_pdfium, _docHandle, annotations) {
  const indent = "    ";
  const annotElements = [];
  for (const annot of annotations) {
    const element = annotationToXfdfElement(annot, indent);
    if (element.length > 0) {
      annotElements.push(element);
    }
  }
  const lines = [];
  lines.push('<?xml version="1.0" encoding="UTF-8"?>');
  lines.push('<xfdf xmlns="http://ns.adobe.com/xfdf/" xml:space="preserve">');
  if (annotElements.length > 0) {
    lines.push("  <annots>");
    lines.push(...annotElements);
    lines.push("  </annots>");
  } else {
    lines.push("  <annots />");
  }
  lines.push("</xfdf>");
  return lines.join("\n");
}

// src/worker/text-ops.ts
var DOUBLE_SIZE = 8;
function withTextPage(pdfium2, docHandle, pageIndex, callback) {
  let page = 0;
  let textPage = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    textPage = pdfium2.fn._FPDFText_LoadPage(page);
    if (textPage === 0) {
      throw new Error(`Failed to load text page ${pageIndex}`);
    }
    return callback(page, textPage);
  } finally {
    if (textPage !== 0) {
      pdfium2.fn._FPDFText_ClosePage(textPage);
    }
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}
function readRectsForRange(pdfium2, textPage, charIndex, count) {
  var _stack = [];
  try {
    const rectCount = pdfium2.fn._FPDFText_CountRects(textPage, charIndex, count);
    if (rectCount <= 0) return [];
    const rects = [];
    const leftAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
    const topAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
    const rightAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
    const bottomAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
    for (let i = 0; i < rectCount; i++) {
      const ok = pdfium2.fn._FPDFText_GetRect(
        textPage,
        i,
        leftAlloc.ptr,
        topAlloc.ptr,
        rightAlloc.ptr,
        bottomAlloc.ptr
      );
      if (ok === 0) continue;
      rects.push({
        left: pdfium2.module.getValue(leftAlloc.ptr, "double"),
        top: pdfium2.module.getValue(topAlloc.ptr, "double"),
        right: pdfium2.module.getValue(rightAlloc.ptr, "double"),
        bottom: pdfium2.module.getValue(bottomAlloc.ptr, "double")
      });
    }
    return rects;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function extractPageText(pdfium2, docHandle, pageIndex) {
  return withTextPage(pdfium2, docHandle, pageIndex, (_page, textPage) => {
    var _stack = [];
    try {
      const charCount = pdfium2.fn._FPDFText_CountChars(textPage);
      if (charCount <= 0) return "";
      const bufferSize = (charCount + 1) * 2;
      const buf = __using(_stack, pdfium2.memory.alloc(bufferSize));
      pdfium2.fn._FPDFText_GetText(textPage, 0, charCount, buf.ptr);
      return pdfium2.memory.fromWideString(buf.ptr);
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  });
}
function extractPageCharInfo(pdfium2, docHandle, pageIndex) {
  return withTextPage(pdfium2, docHandle, pageIndex, (_page, textPage) => {
    var _stack = [];
    try {
      const charCount = pdfium2.fn._FPDFText_CountChars(textPage);
      if (charCount <= 0) return [];
      const chars = [];
      const leftAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
      const rightAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
      const bottomAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
      const topAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE));
      for (let i = 0; i < charCount; i++) {
        const charCode = pdfium2.fn._FPDFText_GetUnicode(textPage, i);
        const boxOk = pdfium2.fn._FPDFText_GetCharBox(
          textPage,
          i,
          leftAlloc.ptr,
          rightAlloc.ptr,
          bottomAlloc.ptr,
          topAlloc.ptr
        );
        let left = 0;
        let right = 0;
        let top = 0;
        let bottom = 0;
        if (boxOk !== 0) {
          left = pdfium2.module.getValue(leftAlloc.ptr, "double");
          right = pdfium2.module.getValue(rightAlloc.ptr, "double");
          bottom = pdfium2.module.getValue(bottomAlloc.ptr, "double");
          top = pdfium2.module.getValue(topAlloc.ptr, "double");
        }
        const fontSize = pdfium2.fn._FPDFText_GetFontSize(textPage, i);
        chars.push({
          charCode,
          char: String.fromCodePoint(charCode),
          left,
          right,
          top,
          bottom,
          fontSize
        });
      }
      return chars;
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  });
}
function searchPageText(pdfium2, docHandle, pageIndex, query, flags) {
  return withTextPage(pdfium2, docHandle, pageIndex, (_page, textPage) => {
    var _stack = [];
    try {
      const queryAlloc = __using(_stack, pdfium2.memory.toWideString(query));
      let handle = 0;
      try {
        handle = pdfium2.fn._FPDFText_FindStart(textPage, queryAlloc.ptr, flags, 0);
        if (handle === 0) {
          throw new Error(`Failed to start text search on page ${pageIndex}`);
        }
        const matches = [];
        while (pdfium2.fn._FPDFText_FindNext(handle) !== 0) {
          const charIndex = pdfium2.fn._FPDFText_GetSchResultIndex(handle);
          const length = pdfium2.fn._FPDFText_GetSchCount(handle);
          const rects = readRectsForRange(pdfium2, textPage, charIndex, length);
          matches.push({
            pageIndex,
            charIndex,
            length,
            rects
          });
        }
        return matches;
      } finally {
        if (handle !== 0) {
          pdfium2.fn._FPDFText_FindClose(handle);
        }
      }
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  });
}
function getTextRects(pdfium2, docHandle, pageIndex, charIndex, count) {
  return withTextPage(pdfium2, docHandle, pageIndex, (_page, textPage) => {
    return readRectsForRange(pdfium2, textPage, charIndex, count);
  });
}
function getCharIndexAtPos(pdfium2, docHandle, pageIndex, x, y, tolerance) {
  return withTextPage(pdfium2, docHandle, pageIndex, (_page, textPage) => {
    return pdfium2.fn._FPDFText_GetCharIndexAtPos(textPage, x, y, tolerance, tolerance);
  });
}

// src/worker/navigation-ops.ts
import { FpdfActionType } from "@truespar/lector-pdfium-wasm";
var DOUBLE_SIZE2 = 8;
function readDestination(pdfium2, docHandle, dest) {
  var _stack = [];
  try {
    if (dest === 0) return null;
    const pageIndex = pdfium2.fn._FPDFDest_GetDestPageIndex(docHandle, dest);
    if (pageIndex < 0) return null;
    const hasXAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const hasYAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const hasZoomAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const xAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE2));
    const yAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE2));
    const zoomAlloc = __using(_stack, pdfium2.memory.alloc(DOUBLE_SIZE2));
    pdfium2.fn._FPDFDest_GetLocationInPage(
      dest,
      hasXAlloc.ptr,
      hasYAlloc.ptr,
      hasZoomAlloc.ptr,
      xAlloc.ptr,
      yAlloc.ptr,
      zoomAlloc.ptr
    );
    const hasX = pdfium2.module.getValue(hasXAlloc.ptr, "i32") !== 0;
    const hasY = pdfium2.module.getValue(hasYAlloc.ptr, "i32") !== 0;
    const hasZoom = pdfium2.module.getValue(hasZoomAlloc.ptr, "i32") !== 0;
    return {
      pageIndex,
      x: hasX ? pdfium2.module.getValue(xAlloc.ptr, "float") : null,
      y: hasY ? pdfium2.module.getValue(yAlloc.ptr, "float") : null,
      zoom: hasZoom ? pdfium2.module.getValue(zoomAlloc.ptr, "float") : null
    };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readPdfiumWideString(pdfium2, getter) {
  var _stack = [];
  try {
    const len = getter(0, 0);
    if (len <= 2) return "";
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    getter(buf.ptr, len);
    return pdfium2.memory.fromWideString(buf.ptr);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readPdfiumByteString(pdfium2, getter) {
  var _stack = [];
  try {
    const len = getter(0, 0);
    if (len <= 1) return "";
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    getter(buf.ptr, len);
    return pdfium2.memory.fromByteString(buf.ptr);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function resolveAction(pdfium2, docHandle, action) {
  if (action === 0) return { type: "unknown" };
  const actionType = pdfium2.fn._FPDFAction_GetType(action);
  switch (actionType) {
    case FpdfActionType.GOTO: {
      const dest = pdfium2.fn._FPDFAction_GetDest(docHandle, action);
      const destination = readDestination(pdfium2, docHandle, dest);
      if (destination === null) return { type: "unknown" };
      return { type: "goto", destination };
    }
    case FpdfActionType.URI: {
      const uri = readPdfiumByteString(
        pdfium2,
        (buffer, buflen) => pdfium2.fn._FPDFAction_GetURIPath(docHandle, action, buffer, buflen)
      );
      if (uri.length === 0) return { type: "unknown" };
      return { type: "uri", uri };
    }
    case FpdfActionType.REMOTEGOTO: {
      const filePath = readPdfiumByteString(
        pdfium2,
        (buffer, buflen) => pdfium2.fn._FPDFAction_GetFilePath(action, buffer, buflen)
      );
      const dest = pdfium2.fn._FPDFAction_GetDest(docHandle, action);
      const destination = readDestination(pdfium2, docHandle, dest);
      return { type: "remote-goto", filePath, destination };
    }
    case FpdfActionType.LAUNCH: {
      const filePath = readPdfiumByteString(
        pdfium2,
        (buffer, buflen) => pdfium2.fn._FPDFAction_GetFilePath(action, buffer, buflen)
      );
      return { type: "launch", filePath };
    }
    default:
      return { type: "unknown" };
  }
}
var FS_RECTF_SIZE2 = 16;
function readBookmarkChildren(pdfium2, docHandle, parent) {
  const nodes = [];
  let bookmark = pdfium2.fn._FPDFBookmark_GetFirstChild(docHandle, parent);
  while (bookmark !== 0) {
    const title = readPdfiumWideString(
      pdfium2,
      (buffer, buflen) => pdfium2.fn._FPDFBookmark_GetTitle(bookmark, buffer, buflen)
    );
    let pageIndex = null;
    const dest = pdfium2.fn._FPDFBookmark_GetDest(docHandle, bookmark);
    if (dest !== 0) {
      const idx = pdfium2.fn._FPDFDest_GetDestPageIndex(docHandle, dest);
      if (idx >= 0) pageIndex = idx;
    } else {
      const action = pdfium2.fn._FPDFBookmark_GetAction(bookmark);
      if (action !== 0) {
        const actionDest = pdfium2.fn._FPDFAction_GetDest(docHandle, action);
        if (actionDest !== 0) {
          const idx = pdfium2.fn._FPDFDest_GetDestPageIndex(docHandle, actionDest);
          if (idx >= 0) pageIndex = idx;
        }
      }
    }
    const children = readBookmarkChildren(pdfium2, docHandle, bookmark);
    nodes.push({ title, pageIndex, children });
    bookmark = pdfium2.fn._FPDFBookmark_GetNextSibling(docHandle, bookmark);
  }
  return nodes;
}
function readBookmarkTree(pdfium2, docHandle) {
  return readBookmarkChildren(pdfium2, docHandle, 0);
}
function readPageLinks(pdfium2, docHandle, pageIndex) {
  let page = 0;
  try {
    var _stack2 = [];
    try {
      page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
      if (page === 0) {
        throw new Error(`Failed to load page ${pageIndex}`);
      }
      const links = [];
      const startPosAlloc = __using(_stack2, pdfium2.memory.alloc(4));
      const linkAnnotAlloc = __using(_stack2, pdfium2.memory.alloc(4));
      pdfium2.module.setValue(startPosAlloc.ptr, 0, "i32");
      while (pdfium2.fn._FPDFLink_Enumerate(
        page,
        startPosAlloc.ptr,
        linkAnnotAlloc.ptr
      ) !== 0) {
        var _stack = [];
        try {
          const linkHandle = pdfium2.module.getValue(
            linkAnnotAlloc.ptr,
            "i32"
          );
          const rectAlloc = __using(_stack, pdfium2.memory.alloc(FS_RECTF_SIZE2));
          const rectOk = pdfium2.fn._FPDFLink_GetAnnotRect(linkHandle, rectAlloc.ptr);
          if (rectOk === 0) continue;
          const rect = pdfium2.memory.readRectF(rectAlloc.ptr);
          let target;
          const dest = pdfium2.fn._FPDFLink_GetDest(docHandle, linkHandle);
          if (dest !== 0) {
            const destination = readDestination(pdfium2, docHandle, dest);
            target = destination !== null ? { type: "goto", destination } : { type: "unknown" };
          } else {
            const action = pdfium2.fn._FPDFLink_GetAction(linkHandle);
            target = resolveAction(pdfium2, docHandle, action);
          }
          links.push({
            rect: { left: rect.left, top: rect.top, right: rect.right, bottom: rect.bottom },
            target
          });
        } catch (_) {
          var _error = _, _hasError = true;
        } finally {
          __callDispose(_stack, _error, _hasError);
        }
      }
      return links;
    } catch (_2) {
      var _error2 = _2, _hasError2 = true;
    } finally {
      __callDispose(_stack2, _error2, _hasError2);
    }
  } finally {
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}
function readPageWebLinks(pdfium2, docHandle, pageIndex) {
  let page = 0;
  let textPage = 0;
  let linkPage = 0;
  try {
    page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
    if (page === 0) {
      throw new Error(`Failed to load page ${pageIndex}`);
    }
    textPage = pdfium2.fn._FPDFText_LoadPage(page);
    if (textPage === 0) {
      throw new Error(`Failed to load text page ${pageIndex}`);
    }
    linkPage = pdfium2.fn._FPDFLink_LoadWebLinks(textPage);
    if (linkPage === 0) return [];
    const count = pdfium2.fn._FPDFLink_CountWebLinks(linkPage);
    const webLinks = [];
    for (let i = 0; i < count; i++) {
      var _stack2 = [];
      try {
        const urlLen = pdfium2.fn._FPDFLink_GetURL(linkPage, i, 0, 0);
        let url = "";
        if (urlLen > 0) {
          var _stack = [];
          try {
            const urlBuf = __using(_stack, pdfium2.memory.alloc(urlLen * 2));
            pdfium2.fn._FPDFLink_GetURL(linkPage, i, urlBuf.ptr, urlLen);
            url = pdfium2.memory.fromWideString(urlBuf.ptr);
          } catch (_) {
            var _error = _, _hasError = true;
          } finally {
            __callDispose(_stack, _error, _hasError);
          }
        }
        if (url.length === 0) continue;
        const rectCount = pdfium2.fn._FPDFLink_CountRects(linkPage, i);
        const rects = [];
        const leftAlloc = __using(_stack2, pdfium2.memory.alloc(DOUBLE_SIZE2));
        const topAlloc = __using(_stack2, pdfium2.memory.alloc(DOUBLE_SIZE2));
        const rightAlloc = __using(_stack2, pdfium2.memory.alloc(DOUBLE_SIZE2));
        const bottomAlloc = __using(_stack2, pdfium2.memory.alloc(DOUBLE_SIZE2));
        for (let r = 0; r < rectCount; r++) {
          const ok = pdfium2.fn._FPDFLink_GetRect(
            linkPage,
            i,
            r,
            leftAlloc.ptr,
            topAlloc.ptr,
            rightAlloc.ptr,
            bottomAlloc.ptr
          );
          if (ok === 0) continue;
          rects.push({
            left: pdfium2.module.getValue(leftAlloc.ptr, "double"),
            top: pdfium2.module.getValue(topAlloc.ptr, "double"),
            right: pdfium2.module.getValue(rightAlloc.ptr, "double"),
            bottom: pdfium2.module.getValue(bottomAlloc.ptr, "double")
          });
        }
        webLinks.push({ url, rects });
      } catch (_2) {
        var _error2 = _2, _hasError2 = true;
      } finally {
        __callDispose(_stack2, _error2, _hasError2);
      }
    }
    return webLinks;
  } finally {
    if (linkPage !== 0) {
      pdfium2.fn._FPDFLink_CloseWebLinks(linkPage);
    }
    if (textPage !== 0) {
      pdfium2.fn._FPDFText_ClosePage(textPage);
    }
    if (page !== 0) {
      pdfium2.fn._FPDF_ClosePage(page);
    }
  }
}

// src/worker/bookmark-ops.ts
function addBookmark(pdfium2, docHandle, title, pageIndex, insertIndex) {
  const utf16 = new Uint8Array(title.length * 2);
  for (let i = 0; i < title.length; i++) {
    const code = title.charCodeAt(i);
    utf16[i * 2] = code & 255;
    utf16[i * 2 + 1] = code >> 8 & 255;
  }
  const alloc = pdfium2.memory.alloc(utf16.byteLength);
  try {
    new Uint8Array(pdfium2.module.HEAPU8.buffer, alloc.ptr, utf16.byteLength).set(utf16);
    return pdfium2.fn._FPDFBookmark_Add(
      docHandle,
      alloc.ptr,
      utf16.byteLength,
      pageIndex,
      insertIndex
    ) !== 0;
  } finally {
    alloc[Symbol.dispose]();
  }
}
function deleteBookmark(pdfium2, docHandle, index) {
  return pdfium2.fn._FPDFBookmark_Delete(docHandle, index) !== 0;
}
function moveBookmark(pdfium2, docHandle, fromIndex, toIndex) {
  return pdfium2.fn._FPDFBookmark_Move(docHandle, fromIndex, toIndex) !== 0;
}
function setBookmarkTitle(pdfium2, docHandle, index, title) {
  const utf16 = new Uint8Array(title.length * 2);
  for (let i = 0; i < title.length; i++) {
    const code = title.charCodeAt(i);
    utf16[i * 2] = code & 255;
    utf16[i * 2 + 1] = code >> 8 & 255;
  }
  const alloc = pdfium2.memory.alloc(utf16.byteLength);
  try {
    new Uint8Array(pdfium2.module.HEAPU8.buffer, alloc.ptr, utf16.byteLength).set(utf16);
    return pdfium2.fn._FPDFBookmark_SetTitle(
      docHandle,
      index,
      alloc.ptr,
      utf16.byteLength
    ) !== 0;
  } finally {
    alloc[Symbol.dispose]();
  }
}
function setBookmarkDest(pdfium2, docHandle, index, pageIndex) {
  return pdfium2.fn._FPDFBookmark_SetDest(docHandle, index, pageIndex) !== 0;
}

// src/worker/page-ops.ts
function deletePage(pdfium2, docHandle, pageIndex) {
  pdfium2.fn._FPDFPage_Delete(docHandle, pageIndex);
}
function insertBlankPage(pdfium2, docHandle, pageIndex, width, height) {
  const page = pdfium2.fn._FPDFPage_New(docHandle, pageIndex, width, height);
  if (!page) throw new Error("FPDFPage_New failed");
  pdfium2.fn._FPDFPage_GenerateContent(page);
  pdfium2.fn._FPDF_ClosePage(page);
}
function rotatePage(pdfium2, docHandle, pageIndex, rotation) {
  const page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
  if (!page) throw new Error("FPDF_LoadPage failed");
  try {
    pdfium2.fn._FPDFPage_SetRotation(page, rotation);
  } finally {
    pdfium2.fn._FPDF_ClosePage(page);
  }
}
function getPageRotation(pdfium2, docHandle, pageIndex) {
  const page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
  if (!page) throw new Error("FPDF_LoadPage failed");
  try {
    return pdfium2.fn._FPDFPage_GetRotation(page);
  } finally {
    pdfium2.fn._FPDF_ClosePage(page);
  }
}
function movePage(pdfium2, docHandle, fromIndex, toIndex) {
  var _stack = [];
  try {
    const indicesAlloc = __using(_stack, pdfium2.memory.alloc(4));
    pdfium2.module.setValue(indicesAlloc.ptr, fromIndex, "i32");
    const result = pdfium2.fn._FPDF_MovePages(docHandle, indicesAlloc.ptr, 1, toIndex);
    if (!result) {
      throw new Error(`Failed to move page ${fromIndex} to ${toIndex}`);
    }
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function duplicatePage(pdfium2, docHandle, pageIndex) {
  var _stack = [];
  try {
    const pageRange = String(pageIndex + 1);
    const rangeAlloc = __using(_stack, pdfium2.memory.toByteString(pageRange));
    const result = pdfium2.fn._FPDF_ImportPages(
      docHandle,
      // dest
      docHandle,
      // source (same document)
      rangeAlloc.ptr,
      pageIndex + 1
      // insert after the source page
    );
    if (!result) {
      throw new Error(`Failed to duplicate page ${pageIndex}`);
    }
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function flattenPage(pdfium2, docHandle, pageIndex, flag = 0) {
  const page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
  if (!page) throw new Error("FPDF_LoadPage failed");
  try {
    return pdfium2.fn._FPDFPage_Flatten(page, flag);
  } finally {
    pdfium2.fn._FPDF_ClosePage(page);
  }
}
var FPDF_ANNOT_REDACT = 28;
var FPDF_ANNOT_SQUARE = 5;
var FPDF_PAGEOBJ_FORM = 5;
var FLOAT_SIZE = 4;
var FS_MATRIX_SIZE = 24;
var FPDFBITMAP_BGRA = 4;
var REDACTION_RASTER_DPI = 150;
var IDENTITY_MATRIX = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };
function normalizeRect(r) {
  return {
    left: Math.min(r.left, r.right),
    right: Math.max(r.left, r.right),
    bottom: Math.min(r.bottom, r.top),
    top: Math.max(r.bottom, r.top)
  };
}
function rectsOverlap(a, b) {
  return a.right > b.left && a.left < b.right && a.top > b.bottom && a.bottom < b.top;
}
function composeMatrix(outer, inner) {
  return {
    a: outer.a * inner.a + outer.c * inner.b,
    b: outer.b * inner.a + outer.d * inner.b,
    c: outer.a * inner.c + outer.c * inner.d,
    d: outer.b * inner.c + outer.d * inner.d,
    e: outer.a * inner.e + outer.c * inner.f + outer.e,
    f: outer.b * inner.e + outer.d * inner.f + outer.f
  };
}
function transformRect(m, r) {
  const corners = [
    [r.left, r.bottom],
    [r.left, r.top],
    [r.right, r.bottom],
    [r.right, r.top]
  ];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const [x, y] of corners) {
    const tx = m.a * x + m.c * y + m.e;
    const ty = m.b * x + m.d * y + m.f;
    if (tx < minX) minX = tx;
    if (tx > maxX) maxX = tx;
    if (ty < minY) minY = ty;
    if (ty > maxY) maxY = ty;
  }
  return { left: minX, right: maxX, bottom: minY, top: maxY };
}
function isRedactionAnnot(pdfium2, annot) {
  var _stack = [];
  try {
    const subtype = pdfium2.fn._FPDFAnnot_GetSubtype(annot);
    if (subtype === FPDF_ANNOT_REDACT) return true;
    if (subtype !== FPDF_ANNOT_SQUARE) return false;
    const tagKey = __using(_stack, pdfium2.memory.toByteString("Subj"));
    const tagLen = pdfium2.fn._FPDFAnnot_GetStringValue(annot, tagKey.ptr, 0, 0);
    if (tagLen <= 2) return false;
    const tagBuf = __using(_stack, pdfium2.memory.alloc(tagLen));
    pdfium2.fn._FPDFAnnot_GetStringValue(annot, tagKey.ptr, tagBuf.ptr, tagLen);
    return pdfium2.memory.fromWideString(tagBuf.ptr) === "redaction";
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function drawOverlayText(pdfium2, docHandle, page, box, spec) {
  var _stack2 = [];
  try {
    const text = spec.overlayText;
    if (!text) return;
    const fontName = __using(_stack2, pdfium2.memory.toByteString("Helvetica"));
    const font = pdfium2.fn._FPDFText_LoadStandardFont(docHandle, fontName.ptr);
    if (!font) return;
    try {
      var _stack = [];
      try {
        const boxHeight = box.top - box.bottom;
        const fontSize = spec.overlayFontSize && spec.overlayFontSize > 0 ? spec.overlayFontSize : Math.max(4, Math.min(12, boxHeight * 0.6));
        const textObj = pdfium2.fn._FPDFPageObj_CreateTextObj(docHandle, font, fontSize);
        if (!textObj) return;
        const textAlloc = __using(_stack, pdfium2.memory.toWideString(text));
        pdfium2.fn._FPDFText_SetText(textObj, textAlloc.ptr);
        const color = spec.overlayColor ?? { r: 255, g: 255, b: 255 };
        pdfium2.fn._FPDFPageObj_SetFillColor(textObj, color.r, color.g, color.b, 255);
        const inset = Math.min(2, boxHeight * 0.15);
        const x = box.left + inset;
        const y = box.bottom + (boxHeight - fontSize) / 2 + fontSize * 0.2;
        pdfium2.fn._FPDFPageObj_Transform(textObj, 1, 0, 0, 1, x, y);
        pdfium2.fn._FPDFPage_InsertObject(page, textObj);
      } catch (_) {
        var _error = _, _hasError = true;
      } finally {
        __callDispose(_stack, _error, _hasError);
      }
    } finally {
      pdfium2.fn._FPDFFont_Close(font);
    }
  } catch (_2) {
    var _error2 = _2, _hasError2 = true;
  } finally {
    __callDispose(_stack2, _error2, _hasError2);
  }
}
function colorToArgb(c) {
  return (255 << 24 | c.r << 16 | c.g << 8 | c.b) >>> 0;
}
function rasterizeRedactions(pdfium2, docHandle, page, rects, specs) {
  const widthPt = pdfium2.fn._FPDF_GetPageWidthF(page);
  const heightPt = pdfium2.fn._FPDF_GetPageHeightF(page);
  const scale = REDACTION_RASTER_DPI / 72;
  const widthPx = Math.max(1, Math.round(widthPt * scale));
  const heightPx = Math.max(1, Math.round(heightPt * scale));
  const bitmap = pdfium2.fn._FPDFBitmap_CreateEx(widthPx, heightPx, FPDFBITMAP_BGRA, 0, 0);
  if (!bitmap) throw new Error("FPDFBitmap_CreateEx failed");
  try {
    pdfium2.fn._FPDFBitmap_FillRect(bitmap, 0, 0, widthPx, heightPx, 4294967295);
    pdfium2.fn._FPDF_RenderPageBitmap(bitmap, page, 0, 0, widthPx, heightPx, 0, 0);
    for (let i = 0; i < rects.length; i++) {
      const r = rects[i];
      const fill = specs[i]?.fillColor ?? { r: 0, g: 0, b: 0 };
      const pxLeft = Math.floor(r.left * scale);
      const pxTop = Math.floor((heightPt - r.top) * scale);
      const pxW = Math.max(0, Math.ceil((r.right - r.left) * scale));
      const pxH = Math.max(0, Math.ceil((r.top - r.bottom) * scale));
      pdfium2.fn._FPDFBitmap_FillRect(bitmap, pxLeft, pxTop, pxW, pxH, colorToArgb(fill));
    }
    const objCount = pdfium2.fn._FPDFPage_CountObjects(page);
    for (let i = objCount - 1; i >= 0; i--) {
      const obj = pdfium2.fn._FPDFPage_GetObject(page, i);
      if (!obj) continue;
      pdfium2.fn._FPDFPage_RemoveObject(page, obj);
      pdfium2.fn._FPDFPageObj_Destroy(obj);
    }
    const image = pdfium2.fn._FPDFPageObj_NewImageObj(docHandle);
    if (!image) throw new Error("FPDFPageObj_NewImageObj failed");
    pdfium2.fn._FPDFImageObj_SetBitmap(0, 0, image, bitmap);
    pdfium2.fn._FPDFImageObj_SetMatrix(image, widthPt, 0, 0, heightPt, 0, 0);
    pdfium2.fn._FPDFPage_InsertObject(page, image);
  } finally {
    pdfium2.fn._FPDFBitmap_Destroy(bitmap);
  }
}
function applyRedactions(pdfium2, docHandle, pageIndex, specs, removeAnnots) {
  if (specs.length === 0) return;
  const page = pdfium2.fn._FPDF_LoadPage(docHandle, pageIndex);
  if (!page) throw new Error("FPDF_LoadPage failed");
  try {
    var _stack = [];
    try {
      const rects = specs.map((s) => normalizeRect(s.rect));
      const leftAlloc = __using(_stack, pdfium2.memory.alloc(FLOAT_SIZE));
      const bottomAlloc = __using(_stack, pdfium2.memory.alloc(FLOAT_SIZE));
      const rightAlloc = __using(_stack, pdfium2.memory.alloc(FLOAT_SIZE));
      const topAlloc = __using(_stack, pdfium2.memory.alloc(FLOAT_SIZE));
      const matrixAlloc = __using(_stack, pdfium2.memory.alloc(FS_MATRIX_SIZE));
      const toRemove = [];
      const collect = (formObj, ctm) => {
        const isPage = formObj === null;
        const count = isPage ? pdfium2.fn._FPDFPage_CountObjects(page) : pdfium2.fn._FPDFFormObj_CountObjects(formObj);
        for (let i = count - 1; i >= 0; i--) {
          const obj = isPage ? pdfium2.fn._FPDFPage_GetObject(page, i) : pdfium2.fn._FPDFFormObj_GetObject(formObj, i);
          if (!obj) continue;
          if (pdfium2.fn._FPDFPageObj_GetType(obj) === FPDF_PAGEOBJ_FORM) {
            pdfium2.fn._FPDFPageObj_GetMatrix(obj, matrixAlloc.ptr);
            const formMatrix = pdfium2.memory.readMatrix(matrixAlloc.ptr);
            collect(obj, composeMatrix(ctm, formMatrix));
            continue;
          }
          if (!pdfium2.fn._FPDFPageObj_GetBounds(obj, leftAlloc.ptr, bottomAlloc.ptr, rightAlloc.ptr, topAlloc.ptr)) {
            continue;
          }
          const localBounds = normalizeRect({
            left: pdfium2.module.getValue(leftAlloc.ptr, "float"),
            bottom: pdfium2.module.getValue(bottomAlloc.ptr, "float"),
            right: pdfium2.module.getValue(rightAlloc.ptr, "float"),
            top: pdfium2.module.getValue(topAlloc.ptr, "float")
          });
          const pageBounds = transformRect(ctm, localBounds);
          for (const r of rects) {
            if (rectsOverlap(pageBounds, r)) {
              toRemove.push({ obj, parentForm: isPage ? null : formObj });
              break;
            }
          }
        }
      };
      collect(null, IDENTITY_MATRIX);
      const needsRaster = toRemove.some((rm) => rm.parentForm !== null);
      if (needsRaster) {
        rasterizeRedactions(pdfium2, docHandle, page, rects, specs);
        for (let i = 0; i < specs.length; i++) {
          drawOverlayText(pdfium2, docHandle, page, rects[i], specs[i]);
        }
      } else {
        for (const { obj } of toRemove) {
          pdfium2.fn._FPDFPage_RemoveObject(page, obj);
          pdfium2.fn._FPDFPageObj_Destroy(obj);
        }
        for (let i = 0; i < specs.length; i++) {
          const spec = specs[i];
          const r = rects[i];
          const fill = spec.fillColor ?? { r: 0, g: 0, b: 0 };
          const rect = pdfium2.fn._FPDFPageObj_CreateNewRect(
            r.left,
            r.bottom,
            r.right - r.left,
            r.top - r.bottom
          );
          if (rect) {
            pdfium2.fn._FPDFPageObj_SetFillColor(rect, fill.r, fill.g, fill.b, 255);
            pdfium2.fn._FPDFPath_SetDrawMode(rect, 1, 0);
            pdfium2.fn._FPDFPage_InsertObject(page, rect);
          }
          drawOverlayText(pdfium2, docHandle, page, r, spec);
        }
      }
      if (removeAnnots) {
        const annotCount = pdfium2.fn._FPDFPage_GetAnnotCount(page);
        for (let i = annotCount - 1; i >= 0; i--) {
          const annot = pdfium2.fn._FPDFPage_GetAnnot(page, i);
          if (!annot) continue;
          let isRedact = false;
          try {
            isRedact = isRedactionAnnot(pdfium2, annot);
          } finally {
            pdfium2.fn._FPDFPage_CloseAnnot(annot);
          }
          if (isRedact) pdfium2.fn._FPDFPage_RemoveAnnot(page, i);
        }
      }
      pdfium2.fn._FPDFPage_GenerateContent(page);
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  } finally {
    pdfium2.fn._FPDF_ClosePage(page);
  }
}

// src/worker/signature-ops.ts
function readSigBuffer(pdfium2, sig, readFn) {
  var _stack = [];
  try {
    const len = readFn(sig, 0, 0);
    if (len <= 0) return new Uint8Array(0);
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    readFn(sig, buf.ptr, len);
    return pdfium2.memory.fromHeap(buf.ptr, len);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readSigString(pdfium2, sig, readFn) {
  var _stack = [];
  try {
    const len = readFn(sig, 0, 0);
    if (len <= 0) return void 0;
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    readFn(sig, buf.ptr, len);
    const bytes = pdfium2.memory.fromHeap(buf.ptr, len - 1);
    return new TextDecoder().decode(bytes) || void 0;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function getSignatureCount(pdfium2, docHandle) {
  return pdfium2.fn._FPDF_GetSignatureCount(docHandle);
}
function getSignatureInfo(pdfium2, docHandle, sigIndex) {
  const sig = pdfium2.fn._FPDF_GetSignatureObject(docHandle, sigIndex);
  const contents = readSigBuffer(
    pdfium2,
    sig,
    (s, buf, len) => pdfium2.fn._FPDFSignatureObj_GetContents(s, buf, len)
  );
  const brCount = pdfium2.fn._FPDFSignatureObj_GetByteRange(sig, 0, 0);
  const byteRange = [];
  if (brCount > 0) {
    var _stack = [];
    try {
      const brBuf = __using(_stack, pdfium2.memory.alloc(brCount * 4));
      pdfium2.fn._FPDFSignatureObj_GetByteRange(sig, brBuf.ptr, brCount);
      for (let i = 0; i < brCount; i++) {
        byteRange.push(pdfium2.module.getValue(brBuf.ptr + i * 4, "i32"));
      }
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  }
  const subFilter = readSigString(
    pdfium2,
    sig,
    (s, buf, len) => pdfium2.fn._FPDFSignatureObj_GetSubFilter(s, buf, len)
  ) ?? "";
  const reason = readSigString(
    pdfium2,
    sig,
    (s, buf, len) => pdfium2.fn._FPDFSignatureObj_GetReason(s, buf, len)
  );
  const time = readSigString(
    pdfium2,
    sig,
    (s, buf, len) => pdfium2.fn._FPDFSignatureObj_GetTime(s, buf, len)
  );
  const docMDPPermission = pdfium2.fn._FPDFSignatureObj_GetDocMDPPermission(sig);
  return {
    index: sigIndex,
    contents,
    byteRange,
    subFilter,
    reason,
    time,
    docMDPPermission
  };
}

// src/worker/attachment-ops.ts
function readAttachmentName(pdfium2, attachment) {
  var _stack = [];
  try {
    const len = pdfium2.fn._FPDFAttachment_GetName(
      attachment,
      0,
      0
    );
    if (len <= 2) return "";
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    pdfium2.fn._FPDFAttachment_GetName(attachment, buf.ptr, len);
    return pdfium2.memory.fromWideString(buf.ptr) || "";
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function readAttachmentStringValue(pdfium2, attachment, key) {
  var _stack = [];
  try {
    const keyAlloc = __using(_stack, pdfium2.memory.toByteString(key));
    const len = pdfium2.fn._FPDFAttachment_GetStringValue(
      attachment,
      keyAlloc.ptr,
      0,
      0
    );
    if (len <= 2) return void 0;
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    pdfium2.fn._FPDFAttachment_GetStringValue(
      attachment,
      keyAlloc.ptr,
      buf.ptr,
      len
    );
    return pdfium2.memory.fromWideString(buf.ptr) || void 0;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function getAttachmentCount(pdfium2, docHandle) {
  return pdfium2.fn._FPDFDoc_GetAttachmentCount(docHandle);
}
function getAttachmentInfo(pdfium2, docHandle, index) {
  var _stack = [];
  try {
    const attachment = pdfium2.fn._FPDFDoc_GetAttachment(docHandle, index);
    if (!attachment) throw new Error(`Attachment at index ${index} not found`);
    const name = readAttachmentName(pdfium2, attachment);
    const outBufLen = __using(_stack, pdfium2.memory.alloc(4));
    pdfium2.fn._FPDFAttachment_GetFile(attachment, 0, 0, outBufLen.ptr);
    const size = pdfium2.module.getValue(outBufLen.ptr, "i32");
    const creationDate = readAttachmentStringValue(pdfium2, attachment, "CreationDate");
    const modDate = readAttachmentStringValue(pdfium2, attachment, "ModDate");
    return { index, name, size: Math.max(0, size), creationDate, modDate };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function getAttachmentData(pdfium2, docHandle, index) {
  var _stack = [];
  try {
    const attachment = pdfium2.fn._FPDFDoc_GetAttachment(docHandle, index);
    if (!attachment) throw new Error(`Attachment at index ${index} not found`);
    const outBufLen = __using(_stack, pdfium2.memory.alloc(4));
    pdfium2.fn._FPDFAttachment_GetFile(attachment, 0, 0, outBufLen.ptr);
    const len = pdfium2.module.getValue(outBufLen.ptr, "i32");
    if (len <= 0) return new ArrayBuffer(0);
    const buf = __using(_stack, pdfium2.memory.alloc(len));
    const success = pdfium2.fn._FPDFAttachment_GetFile(attachment, buf.ptr, len, outBufLen.ptr);
    if (!success) throw new Error(`Failed to read attachment data at index ${index}`);
    const bytes = pdfium2.memory.fromHeap(buf.ptr, len);
    const result = new ArrayBuffer(bytes.byteLength);
    new Uint8Array(result).set(bytes);
    return result;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function addAttachment(pdfium2, docHandle, name, data) {
  var _stack = [];
  try {
    const nameAlloc = __using(_stack, pdfium2.memory.toWideString(name));
    const attachment = pdfium2.fn._FPDFDoc_AddAttachment(docHandle, nameAlloc.ptr);
    if (!attachment) throw new Error(`Failed to create attachment "${name}"`);
    const bytes = new Uint8Array(data);
    const dataAlloc = __using(_stack, pdfium2.memory.toHeap(bytes));
    const success = pdfium2.fn._FPDFAttachment_SetFile(
      attachment,
      docHandle,
      dataAlloc.ptr,
      bytes.byteLength
    );
    if (!success) throw new Error(`Failed to write attachment data for "${name}"`);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function deleteAttachment(pdfium2, docHandle, index) {
  const result = pdfium2.fn._FPDFDoc_DeleteAttachment(docHandle, index);
  if (!result) throw new Error(`Failed to delete attachment at index ${index}`);
}

// src/worker/layer-ops.ts
function getLayerCount(pdfium2, docHandle) {
  const count = pdfium2.fn._FPDFDoc_GetOCGCount(docHandle);
  return Math.max(0, count);
}
function getAllLayers(pdfium2, docHandle) {
  const count = getLayerCount(pdfium2, docHandle);
  const layers = [];
  for (let i = 0; i < count; i++) {
    layers.push(getLayerInfo(pdfium2, docHandle, i));
  }
  return layers;
}
function getLayerInfo(pdfium2, docHandle, index) {
  const nameLen = pdfium2.fn._FPDFDoc_GetOCGName(
    docHandle,
    index,
    0,
    0
  );
  let name = "";
  if (nameLen > 2) {
    var _stack = [];
    try {
      const nameBuf = __using(_stack, pdfium2.memory.alloc(nameLen));
      pdfium2.fn._FPDFDoc_GetOCGName(docHandle, index, nameBuf.ptr, nameLen);
      name = pdfium2.memory.fromWideString(nameBuf.ptr) || "";
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  }
  const intentLen = pdfium2.fn._FPDFDoc_GetOCGIntent(
    docHandle,
    index,
    0,
    0
  );
  let intent = "";
  if (intentLen > 2) {
    var _stack2 = [];
    try {
      const intentBuf = __using(_stack2, pdfium2.memory.alloc(intentLen));
      pdfium2.fn._FPDFDoc_GetOCGIntent(docHandle, index, intentBuf.ptr, intentLen);
      intent = pdfium2.memory.fromWideString(intentBuf.ptr) || "";
    } catch (_2) {
      var _error2 = _2, _hasError2 = true;
    } finally {
      __callDispose(_stack2, _error2, _hasError2);
    }
  }
  const visResult = pdfium2.fn._FPDFDoc_GetOCGVisible(docHandle, index);
  const visible = visResult !== 0;
  return { index, name, intent, visible };
}
function setLayerVisible(pdfium2, docHandle, index, visible) {
  const result = pdfium2.fn._FPDFDoc_SetOCGVisible(docHandle, index, visible ? 1 : 0);
  if (!result) {
    throw new Error(`Failed to set layer ${index} visibility`);
  }
}

// src/worker/security-ops.ts
function setDocumentPassword(pdfium2, docHandle, options) {
  var _stack = [];
  try {
    const permissions = pdfium2.fn._FPDFDoc_GetPermissionFlags(
      options.allowPrint ?? true ? 1 : 0,
      options.allowModify ?? false ? 1 : 0,
      options.allowExtract ?? true ? 1 : 0,
      options.allowAnnotate ?? true ? 1 : 0
    );
    const userPwAlloc = __using(_stack, pdfium2.memory.toByteString(options.userPassword));
    const ownerPw = options.ownerPassword ?? options.userPassword;
    const ownerPwAlloc = __using(_stack, pdfium2.memory.toByteString(ownerPw));
    const ok = pdfium2.fn._FPDFDoc_SetPassword(
      docHandle,
      userPwAlloc.ptr,
      ownerPwAlloc.ptr,
      permissions
    );
    if (!ok) {
      throw new Error("Failed to set document password protection");
    }
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}

// src/worker/tsa-helpers.ts
function derLength(n) {
  if (n < 128) return new Uint8Array([n]);
  if (n < 256) return new Uint8Array([129, n]);
  if (n < 65536) return new Uint8Array([130, n >> 8 & 255, n & 255]);
  return new Uint8Array([131, n >> 16 & 255, n >> 8 & 255, n & 255]);
}
function concat(...parts) {
  let total = 0;
  for (const p of parts) total += p.length;
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}
function derTLV(tag, content) {
  return concat(new Uint8Array([tag]), derLength(content.length), content);
}
function derSequence(items) {
  return derTLV(48, concat(...items));
}
function derInteger(n) {
  if (n === 0) return new Uint8Array([2, 1, 0]);
  const bytes = [];
  let v = n;
  while (v > 0) {
    bytes.unshift(v & 255);
    v >>>= 8;
  }
  if (bytes[0] & 128) bytes.unshift(0);
  return derTLV(2, new Uint8Array(bytes));
}
function derOctetString(bytes) {
  return derTLV(4, bytes);
}
function derNull() {
  return new Uint8Array([5, 0]);
}
function derBoolean(b) {
  return new Uint8Array([1, 1, b ? 255 : 0]);
}
function derOID(oid) {
  const parts = oid.split(".").map((s) => parseInt(s, 10));
  if (parts.length < 2) throw new Error("invalid OID");
  const bytes = [parts[0] * 40 + parts[1]];
  for (let i = 2; i < parts.length; i++) {
    const v = parts[i];
    if (v < 128) {
      bytes.push(v);
    } else {
      const sub = [];
      let x = v;
      while (x > 0) {
        sub.unshift(x & 127);
        x >>>= 7;
      }
      for (let j = 0; j < sub.length - 1; j++) sub[j] |= 128;
      bytes.push(...sub);
    }
  }
  return derTLV(6, new Uint8Array(bytes));
}
function readBer(bytes, offset) {
  let p = offset;
  if (p >= bytes.length) throw new Error("BER: unexpected end");
  const tag = bytes[p++];
  const lenByte = bytes[p++];
  let length;
  if (lenByte < 128) {
    length = lenByte;
  } else {
    const lenBytes = lenByte & 127;
    if (lenBytes === 0 || lenBytes > 4) throw new Error("BER: bad length");
    length = 0;
    for (let i = 0; i < lenBytes; i++) length = length << 8 | bytes[p++];
  }
  return { tag, contentStart: p, contentEnd: p + length, nextOffset: p + length };
}
function buildTimeStampReq(sigHash) {
  const hashAlgo = derSequence([
    derOID("2.16.840.1.101.3.4.2.1"),
    derNull()
  ]);
  const messageImprint = derSequence([hashAlgo, derOctetString(sigHash)]);
  return derSequence([
    derInteger(1),
    // version
    messageImprint,
    derBoolean(true)
    // certReq — request the TSA cert in the response
  ]);
}
function extractTimeStampToken(response) {
  const outer = readBer(response, 0);
  if (outer.tag !== 48) throw new Error("TSA response: not a SEQUENCE");
  const statusInfo = readBer(response, outer.contentStart);
  if (statusInfo.tag !== 48) throw new Error("TSA response: PKIStatusInfo not SEQUENCE");
  const status = readBer(response, statusInfo.contentStart);
  if (status.tag !== 2) throw new Error("TSA response: PKIStatus not INTEGER");
  const statusValue = response[status.contentStart];
  if (statusValue !== 0 && statusValue !== 1) {
    throw new Error(`TSA rejected timestamp request (status=${statusValue})`);
  }
  if (statusInfo.nextOffset >= outer.contentEnd) {
    throw new Error("TSA response has no timeStampToken");
  }
  const token = readBer(response, statusInfo.nextOffset);
  if (token.tag !== 48) throw new Error("TimeStampToken not SEQUENCE");
  const tokenLen = token.nextOffset - statusInfo.nextOffset;
  return response.subarray(statusInfo.nextOffset, statusInfo.nextOffset + tokenLen);
}
function findSignatureValueInCms(cms) {
  const contentInfo = readBer(cms, 0);
  if (contentInfo.tag !== 48) throw new Error("CMS: not ContentInfo SEQUENCE");
  const contentType = readBer(cms, contentInfo.contentStart);
  if (contentType.tag !== 6) throw new Error("CMS: contentType not OID");
  const ctx = readBer(cms, contentType.nextOffset);
  if (ctx.tag !== 160) throw new Error("CMS: expected [0] EXPLICIT");
  const signedData = readBer(cms, ctx.contentStart);
  if (signedData.tag !== 48) throw new Error("CMS: SignedData not SEQUENCE");
  let p = signedData.contentStart;
  p = readBer(cms, p).nextOffset;
  p = readBer(cms, p).nextOffset;
  p = readBer(cms, p).nextOffset;
  while (p < signedData.contentEnd) {
    const o = readBer(cms, p);
    if (o.tag === 160 || o.tag === 161) {
      p = o.nextOffset;
    } else {
      break;
    }
  }
  const signerInfos = readBer(cms, p);
  if (signerInfos.tag !== 49) throw new Error("CMS: signerInfos not SET");
  const signerInfo = readBer(cms, signerInfos.contentStart);
  if (signerInfo.tag !== 48) throw new Error("CMS: SignerInfo not SEQUENCE");
  let q = signerInfo.contentStart;
  q = readBer(cms, q).nextOffset;
  q = readBer(cms, q).nextOffset;
  q = readBer(cms, q).nextOffset;
  let next = readBer(cms, q);
  if (next.tag === 160) {
    q = next.nextOffset;
  }
  q = readBer(cms, q).nextOffset;
  const sig = readBer(cms, q);
  if (sig.tag !== 4) throw new Error("CMS: signature not OCTET STRING");
  return cms.subarray(sig.contentStart, sig.contentEnd);
}

// src/worker/signing-ops.ts
function parsePfx(pdfium2, pfxData, password) {
  var _stack = [];
  try {
    const pfxBytes = new Uint8Array(pfxData);
    const pfxAlloc = __using(_stack, pdfium2.memory.alloc(pfxBytes.byteLength));
    pdfium2.module.HEAPU8.set(pfxBytes, pfxAlloc.ptr);
    const pwAlloc = __using(_stack, pdfium2.memory.toByteString(password));
    const errAlloc = __using(_stack, pdfium2.memory.alloc(512));
    pdfium2.module.HEAPU8.fill(0, errAlloc.ptr, errAlloc.ptr + 512);
    const handle = pdfium2.fn._lector_pkcs12_parse(
      pfxAlloc.ptr,
      pfxBytes.byteLength,
      pwAlloc.ptr,
      errAlloc.ptr,
      512
    );
    if (!handle) {
      const errBytes = pdfium2.memory.fromHeap(errAlloc.ptr, 512);
      const nullIdx = errBytes.indexOf(0);
      const errMsg = new TextDecoder().decode(errBytes.subarray(0, nullIdx > 0 ? nullIdx : 512));
      throw new Error(errMsg || "Failed to parse PFX file");
    }
    return handle;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function getSignerSubject(pdfium2, pfxHandle) {
  var _stack = [];
  try {
    const certCount = pdfium2.fn._lector_pkcs12_get_cert_count(pfxHandle);
    if (certCount <= 0) return "Unknown";
    const buf = __using(_stack, pdfium2.memory.alloc(256));
    const len = pdfium2.fn._lector_pkcs12_get_key_algo(pfxHandle, buf.ptr, 256);
    if (len > 0) {
      const bytes = pdfium2.memory.fromHeap(buf.ptr, len);
      return new TextDecoder().decode(bytes);
    }
    return "Unknown";
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function findAndPatchByteRange(pdfBytes) {
  let text = "";
  for (let i = 0; i < pdfBytes.length; i++) {
    text += String.fromCharCode(pdfBytes[i]);
  }
  const placeholderProbe = "<" + "0".repeat(256);
  let placeholderStart = text.indexOf(placeholderProbe);
  if (placeholderStart === -1) {
    const sigIdx = text.indexOf("/Sig");
    const contentsIdx = text.indexOf("/Contents");
    throw new Error(
      `Could not find /Contents placeholder in saved PDF. Diagnostics: pdfLen=${pdfBytes.length}, /Sig at ${sigIdx}, /Contents at ${contentsIdx}` + (contentsIdx >= 0 ? `, surrounding=${JSON.stringify(text.substring(contentsIdx, contentsIdx + 80))}` : "")
    );
  }
  const hexEnd = text.indexOf(">", placeholderStart);
  if (hexEnd === -1) {
    throw new Error("Could not find end of /Contents placeholder");
  }
  const hexStart = placeholderStart + 1;
  const contentsFieldStart = placeholderStart;
  const contentsFieldEnd = hexEnd + 1;
  const byteRanges = [
    0,
    contentsFieldStart,
    contentsFieldEnd,
    pdfBytes.length - contentsFieldEnd
  ];
  const byteRangeMarker = "/ByteRange";
  let brIdx = text.lastIndexOf(byteRangeMarker, placeholderStart);
  if (brIdx === -1) {
    brIdx = text.indexOf(byteRangeMarker, placeholderStart);
  }
  if (brIdx === -1) {
    throw new Error("Could not find /ByteRange placeholder in saved PDF");
  }
  const brOpen = text.indexOf("[", brIdx);
  if (brOpen === -1) {
    throw new Error("Could not find [ after /ByteRange");
  }
  const brStart = brOpen + 1;
  const brEnd = text.indexOf("]", brStart);
  if (brEnd === -1) {
    throw new Error("Could not find end of /ByteRange placeholder");
  }
  const originalBrLen = brEnd - brStart;
  const brValue = `${byteRanges[0]} ${byteRanges[1]} ${byteRanges[2]} ${byteRanges[3]}`;
  if (brValue.length > originalBrLen) {
    throw new Error(`ByteRange value too long: ${brValue.length} > ${originalBrLen}`);
  }
  const paddedBrValue = brValue.padEnd(originalBrLen, " ");
  const patched = new Uint8Array(pdfBytes);
  const encoder = new TextEncoder();
  const brBytes = encoder.encode(paddedBrValue);
  patched.set(brBytes, brStart);
  return {
    patchedPdf: patched,
    byteRanges,
    contentsHexStart: hexStart,
    contentsHexLen: hexEnd - hexStart
  };
}
function embedSignatureCms(pdfBytes, cmsDer, contentsHexStart, contentsHexLen) {
  const hexChars = "0123456789abcdef";
  let hex = "";
  for (let i = 0; i < cmsDer.length; i++) {
    hex += hexChars[cmsDer[i] >> 4];
    hex += hexChars[cmsDer[i] & 15];
  }
  if (hex.length > contentsHexLen) {
    throw new Error(
      `CMS signature (${hex.length} hex chars) exceeds placeholder (${contentsHexLen} hex chars). Increase placeholderSize.`
    );
  }
  hex = hex.padEnd(contentsHexLen, "0");
  const encoder = new TextEncoder();
  const hexBytes = encoder.encode(hex);
  pdfBytes.set(hexBytes, contentsHexStart);
  return pdfBytes;
}
function formatPdfUtcTime(d) {
  const pad = (n) => n.toString().padStart(2, "0");
  return pad(d.getUTCFullYear() % 100) + pad(d.getUTCMonth() + 1) + pad(d.getUTCDate()) + pad(d.getUTCHours()) + pad(d.getUTCMinutes()) + pad(d.getUTCSeconds()) + "Z";
}
async function fetchTimestampToken(pdfium2, cmsDer, tsaUrl) {
  var _stack = [];
  try {
    const sigValue = findSignatureValueInCms(cmsDer);
    const sigAlloc = __using(_stack, pdfium2.memory.alloc(sigValue.byteLength));
    pdfium2.module.HEAPU8.set(sigValue, sigAlloc.ptr);
    const ranges = new Int32Array([0, sigValue.byteLength]);
    const rangeAlloc = __using(_stack, pdfium2.memory.alloc(ranges.byteLength));
    pdfium2.module.HEAP32.set(ranges, rangeAlloc.ptr >> 2);
    const hashAlgAlloc = __using(_stack, pdfium2.memory.toByteString("SHA-256"));
    const hashOutAlloc = __using(_stack, pdfium2.memory.alloc(32));
    const hashOk = pdfium2.fn._lector_crypto_hash_byte_range(
      sigAlloc.ptr,
      sigValue.byteLength,
      rangeAlloc.ptr,
      2,
      hashAlgAlloc.ptr,
      hashOutAlloc.ptr,
      32
    );
    if (!hashOk) throw new Error("Failed to hash CMS signature value for TSA");
    const sigHash = pdfium2.memory.fromHeap(hashOutAlloc.ptr, 32).slice();
    const tsaRequest = buildTimeStampReq(sigHash);
    const reqBuffer = new ArrayBuffer(tsaRequest.byteLength);
    new Uint8Array(reqBuffer).set(tsaRequest);
    const response = await fetch(tsaUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/timestamp-query",
        "Accept": "application/timestamp-reply"
      },
      body: reqBuffer
    });
    if (!response.ok) {
      throw new Error(`TSA HTTP ${response.status}: ${response.statusText}`);
    }
    const responseBytes = new Uint8Array(await response.arrayBuffer());
    return extractTimeStampToken(responseBytes);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function prepareSignaturePlaceholder(pdfium2, docHandle, options) {
  var _stack = [];
  try {
    const placeholderSize = options.placeholderSize ?? 16384;
    const reasonAlloc = __using(_stack, options.reason ? pdfium2.memory.toByteString(options.reason) : void 0);
    const nameAlloc = __using(_stack, options.signerName ? pdfium2.memory.toByteString(options.signerName) : void 0);
    let appearanceAlloc;
    let appearancePtr = 0;
    let appearanceLen = 0;
    let appearanceW = 0;
    let appearanceH = 0;
    if (options.appearanceJpeg && options.appearanceWidth && options.appearanceHeight) {
      const jpegBytes = new Uint8Array(options.appearanceJpeg);
      appearanceAlloc = pdfium2.memory.alloc(jpegBytes.byteLength);
      pdfium2.module.HEAPU8.set(jpegBytes, appearanceAlloc.ptr);
      appearancePtr = appearanceAlloc.ptr;
      appearanceLen = jpegBytes.byteLength;
      appearanceW = options.appearanceWidth;
      appearanceH = options.appearanceHeight;
    }
    try {
      const ok = pdfium2.fn._lector_sig_prepare_field(
        docHandle,
        options.pageIndex,
        options.rectLeft,
        options.rectBottom,
        options.rectRight,
        options.rectTop,
        reasonAlloc?.ptr ?? 0,
        nameAlloc?.ptr ?? 0,
        placeholderSize,
        options.mdpLevel ?? 0,
        appearancePtr,
        appearanceLen,
        appearanceW,
        appearanceH
      );
      if (!ok) {
        throw new Error("Failed to prepare signature field");
      }
    } finally {
      appearanceAlloc?.[Symbol.dispose]();
    }
    const savedBuffer = saveDocumentAsCopy(pdfium2, docHandle);
    const initialBytes = new Uint8Array(savedBuffer);
    const brInfo = findAndPatchByteRange(initialBytes);
    const patchedPdf = brInfo.patchedPdf;
    const byteRangeArray = new Int32Array([
      brInfo.byteRanges[0],
      brInfo.byteRanges[1],
      brInfo.byteRanges[2],
      brInfo.byteRanges[3]
    ]);
    const pdfAlloc = __using(_stack, pdfium2.memory.alloc(patchedPdf.byteLength));
    pdfium2.module.HEAPU8.set(patchedPdf, pdfAlloc.ptr);
    const brAlloc = __using(_stack, pdfium2.memory.alloc(byteRangeArray.byteLength));
    pdfium2.module.HEAP32.set(byteRangeArray, brAlloc.ptr >> 2);
    const hashAlgAlloc = __using(_stack, pdfium2.memory.toByteString("SHA-256"));
    const hashOutAlloc = __using(_stack, pdfium2.memory.alloc(32));
    const hashOk = pdfium2.fn._lector_crypto_hash_byte_range(
      pdfAlloc.ptr,
      patchedPdf.byteLength,
      brAlloc.ptr,
      4,
      // 4 = number of int32 entries (2 pairs of offset+length)
      hashAlgAlloc.ptr,
      hashOutAlloc.ptr,
      32
    );
    if (!hashOk) {
      throw new Error("Failed to hash byte ranges");
    }
    const hash = pdfium2.memory.fromHeap(hashOutAlloc.ptr, 32).slice();
    return {
      patchedPdf,
      contentsHexStart: brInfo.contentsHexStart,
      contentsHexLen: brInfo.contentsHexLen,
      byteRange: brInfo.byteRanges,
      hash,
      hashAlgorithm: "SHA-256",
      signingTime: /* @__PURE__ */ new Date()
    };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
async function produceCmsWithPfx(pdfium2, pfxHandle, hash, signingTime, tsaUrl) {
  var _stack = [];
  try {
    const docHashAlloc = __using(_stack, pdfium2.memory.alloc(hash.byteLength));
    pdfium2.module.HEAPU8.set(hash, docHashAlloc.ptr);
    const signingTimeStr = formatPdfUtcTime(signingTime);
    const signingTimeAlloc = __using(_stack, pdfium2.memory.toByteString(signingTimeStr));
    const outLenAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const cmsErrAlloc = __using(_stack, pdfium2.memory.alloc(512));
    pdfium2.module.HEAPU8.fill(0, cmsErrAlloc.ptr, cmsErrAlloc.ptr + 512);
    const cmsPtr = pdfium2.fn._lector_cms_sign(
      pfxHandle,
      docHashAlloc.ptr,
      hash.byteLength,
      signingTimeAlloc.ptr,
      0,
      0,
      // no TSA token on first call
      outLenAlloc.ptr,
      cmsErrAlloc.ptr,
      512
    );
    if (!cmsPtr) {
      const errBytes = pdfium2.memory.fromHeap(cmsErrAlloc.ptr, 512);
      const nullIdx = errBytes.indexOf(0);
      const errMsg = new TextDecoder().decode(errBytes.subarray(0, nullIdx > 0 ? nullIdx : 512));
      throw new Error(errMsg || "Failed to create CMS signature");
    }
    const cmsLen = pdfium2.module.getValue(outLenAlloc.ptr, "i32");
    let cmsDer = pdfium2.memory.fromHeap(cmsPtr, cmsLen).slice();
    pdfium2.fn._lector_cms_free(cmsPtr);
    if (!tsaUrl) {
      return cmsDer;
    }
    const tsaToken = await fetchTimestampToken(pdfium2, cmsDer, tsaUrl);
    const tsaTokenAlloc = __using(_stack, pdfium2.memory.alloc(tsaToken.byteLength));
    pdfium2.module.HEAPU8.set(tsaToken, tsaTokenAlloc.ptr);
    const outLenAlloc2 = __using(_stack, pdfium2.memory.alloc(4));
    const cmsErrAlloc2 = __using(_stack, pdfium2.memory.alloc(512));
    pdfium2.module.HEAPU8.fill(0, cmsErrAlloc2.ptr, cmsErrAlloc2.ptr + 512);
    const cmsPtr2 = pdfium2.fn._lector_cms_sign(
      pfxHandle,
      docHashAlloc.ptr,
      hash.byteLength,
      signingTimeAlloc.ptr,
      tsaTokenAlloc.ptr,
      tsaToken.byteLength,
      outLenAlloc2.ptr,
      cmsErrAlloc2.ptr,
      512
    );
    if (!cmsPtr2) {
      const errBytes = pdfium2.memory.fromHeap(cmsErrAlloc2.ptr, 512);
      const nullIdx = errBytes.indexOf(0);
      const errMsg = new TextDecoder().decode(errBytes.subarray(0, nullIdx > 0 ? nullIdx : 512));
      throw new Error(errMsg || "Failed to create CMS signature with TSA");
    }
    const cmsLen2 = pdfium2.module.getValue(outLenAlloc2.ptr, "i32");
    cmsDer = pdfium2.memory.fromHeap(cmsPtr2, cmsLen2).slice();
    pdfium2.fn._lector_cms_free(cmsPtr2);
    return cmsDer;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
async function signDocument(pdfium2, docHandle, options) {
  const pfxHandle = parsePfx(pdfium2, options.pfxData, options.pfxPassword);
  try {
    const signerSubject = getSignerSubject(pdfium2, pfxHandle);
    const prepared = prepareSignaturePlaceholder(pdfium2, docHandle, options);
    const cmsDer = await produceCmsWithPfx(
      pdfium2,
      pfxHandle,
      prepared.hash,
      prepared.signingTime,
      options.tsaUrl
    );
    const finalPdf = embedSignatureCms(
      prepared.patchedPdf,
      cmsDer,
      prepared.contentsHexStart,
      prepared.contentsHexLen
    );
    return {
      signedPdf: finalPdf.buffer,
      signerSubject
    };
  } finally {
    pdfium2.fn._lector_pkcs12_free(pfxHandle);
  }
}

// src/worker/capture-ops.ts
var DEFAULT_DPI = 300;
var DEFAULT_BG = 4294967295;
var DEFAULT_FLAGS = 3;
async function captureRegion(pdfium2, docHandle, options) {
  const { fn } = pdfium2;
  const dpi = options.dpi ?? DEFAULT_DPI;
  const rotation = options.rotation ?? 0;
  const bg = options.backgroundColor ?? DEFAULT_BG;
  const scale = dpi / 72;
  if (options.rect.width <= 0 || options.rect.height <= 0) {
    throw new Error(
      `captureRegion: invalid rect (${options.rect.width}\xD7${options.rect.height})`
    );
  }
  const bitmapW = Math.max(1, Math.round(options.rect.width * scale));
  const bitmapH = Math.max(1, Math.round(options.rect.height * scale));
  let page = 0;
  let bitmap = 0;
  try {
    page = fn._FPDF_LoadPage(docHandle, options.pageIndex);
    if (page === 0) {
      const errCode = fn._FPDF_GetLastError();
      throw new Error(
        `Failed to load page ${options.pageIndex}: pdfium error ${errCode}`
      );
    }
    const preW = fn._FPDF_GetPageWidthF(page);
    const preH = fn._FPDF_GetPageHeightF(page);
    const rotatedW = rotation === 1 || rotation === 3 ? preH : preW;
    const rotatedH = rotation === 1 || rotation === 3 ? preW : preH;
    const fullPageW = Math.round(rotatedW * scale);
    const fullPageH = Math.round(rotatedH * scale);
    const startX = -Math.round(options.rect.x * scale);
    const startY = -Math.round(options.rect.y * scale);
    bitmap = fn._FPDFBitmap_CreateEx(bitmapW, bitmapH, 4, 0, 0);
    if (bitmap === 0) {
      throw new Error(
        `Failed to create capture bitmap (${bitmapW}\xD7${bitmapH}): out of memory`
      );
    }
    fn._FPDFBitmap_FillRect(bitmap, 0, 0, bitmapW, bitmapH, bg);
    const flags = DEFAULT_FLAGS & ~16;
    fn._FPDF_RenderPageBitmap(
      bitmap,
      page,
      startX,
      startY,
      fullPageW,
      fullPageH,
      rotation,
      flags
    );
    fn._lector_render_form_widgets(
      docHandle,
      bitmap,
      page,
      startX,
      startY,
      fullPageW,
      fullPageH,
      rotation,
      flags
    );
    const bufferPtr = fn._FPDFBitmap_GetBuffer(bitmap);
    const stride = fn._FPDFBitmap_GetStride(bitmap);
    const totalBytes = stride * bitmapH;
    const pixelsCopy = new Uint8Array(totalBytes);
    pixelsCopy.set(pdfium2.memory.heapView(bufferPtr, totalBytes));
    const pixels = new Uint32Array(pixelsCopy.buffer);
    for (let i = 0; i < pixels.length; i++) {
      const v = pixels[i];
      pixels[i] = v & 4278255360 | (v & 255) << 16 | (v & 16711680) >>> 16;
    }
    const rgba = new Uint8ClampedArray(pixelsCopy.buffer);
    const imageData = new ImageData(rgba, bitmapW, bitmapH);
    return await createImageBitmap(imageData);
  } finally {
    if (bitmap !== 0) {
      fn._FPDFBitmap_Destroy(bitmap);
    }
    if (page !== 0) {
      fn._FPDF_ClosePage(page);
    }
  }
}

// src/worker/comparison-ops.ts
function normaliseText(text) {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}
var TEXT_PAGE_MIN_CHARS = 20;
var TEXT_PAGE_FULL_REPLACE_THRESHOLD = 0.15;
var TEXT_PAGE_FULL_REPLACE_MIN_TOKENS = 30;
var FULL_REPLACE_SNIPPET_LIMIT = 600;
function tokenise(text) {
  const tokens = [];
  let i = 0;
  while (i < text.length) {
    while (i < text.length && /\s/.test(text[i])) i++;
    if (i >= text.length) break;
    const start = i;
    while (i < text.length && !/\s/.test(text[i])) i++;
    tokens.push({
      text: text.slice(start, i),
      charStart: start,
      charEnd: i
    });
  }
  return tokens;
}
function lcsDiff(a, b, eq = (x, y) => x === y) {
  const n = a.length;
  const m = b.length;
  const dp = new Int32Array((n + 1) * (m + 1));
  const w = m + 1;
  for (let i2 = 1; i2 <= n; i2++) {
    for (let j2 = 1; j2 <= m; j2++) {
      if (eq(a[i2 - 1], b[j2 - 1])) {
        dp[i2 * w + j2] = dp[(i2 - 1) * w + (j2 - 1)] + 1;
      } else {
        dp[i2 * w + j2] = Math.max(dp[(i2 - 1) * w + j2], dp[i2 * w + (j2 - 1)]);
      }
    }
  }
  const ops = [];
  let i = n;
  let j = m;
  while (i > 0 && j > 0) {
    if (eq(a[i - 1], b[j - 1])) {
      ops.push({ kind: "eq", a: a[i - 1], b: b[j - 1], aIdx: i - 1, bIdx: j - 1 });
      i--;
      j--;
    } else if (dp[(i - 1) * w + j] >= dp[i * w + (j - 1)]) {
      ops.push({ kind: "del", a: a[i - 1], aIdx: i - 1 });
      i--;
    } else {
      ops.push({ kind: "ins", b: b[j - 1], bIdx: j - 1 });
      j--;
    }
  }
  while (i > 0) {
    ops.push({ kind: "del", a: a[i - 1], aIdx: i - 1 });
    i--;
  }
  while (j > 0) {
    ops.push({ kind: "ins", b: b[j - 1], bIdx: j - 1 });
    j--;
  }
  return ops.reverse();
}
function unionCharRects(chars, start, end) {
  let left = Infinity;
  let right = -Infinity;
  let top = -Infinity;
  let bottom = Infinity;
  let any = false;
  for (let k = start; k < end && k < chars.length; k++) {
    const c = chars[k];
    if (c.left === c.right || c.top === c.bottom) continue;
    if (c.left < left) left = c.left;
    if (c.right > right) right = c.right;
    if (c.top > top) top = c.top;
    if (c.bottom < bottom) bottom = c.bottom;
    any = true;
  }
  if (!any) return null;
  return { left, right, top, bottom };
}
var REGION_DPI = 96;
var REGION_TILE_PX = 16;
var REGION_TILE_CHANGE_THRESHOLD = 0.05;
var REGION_PIXEL_TOLERANCE = 24;
function diffPixelsByTile(rgbaA, rgbaB, width, height) {
  const tilesX = Math.ceil(width / REGION_TILE_PX);
  const tilesY = Math.ceil(height / REGION_TILE_PX);
  const out = [];
  for (let ty = 0; ty < tilesY; ty++) {
    const y0 = ty * REGION_TILE_PX;
    const y1 = Math.min(y0 + REGION_TILE_PX, height);
    for (let tx = 0; tx < tilesX; tx++) {
      const x0 = tx * REGION_TILE_PX;
      const x1 = Math.min(x0 + REGION_TILE_PX, width);
      let changedPixels = 0;
      let totalPixels = 0;
      for (let y = y0; y < y1; y++) {
        const rowOffset = y * width * 4;
        for (let x = x0; x < x1; x++) {
          const i = rowOffset + x * 4;
          const dr = Math.abs(rgbaA[i] - rgbaB[i]);
          const dg = Math.abs(rgbaA[i + 1] - rgbaB[i + 1]);
          const db = Math.abs(rgbaA[i + 2] - rgbaB[i + 2]);
          if (dr > REGION_PIXEL_TOLERANCE || dg > REGION_PIXEL_TOLERANCE || db > REGION_PIXEL_TOLERANCE) {
            changedPixels++;
          }
          totalPixels++;
        }
      }
      const delta = totalPixels > 0 ? changedPixels / totalPixels : 0;
      if (delta > REGION_TILE_CHANGE_THRESHOLD) {
        out.push({ tileX: tx, tileY: ty, delta });
      }
    }
  }
  return out;
}
function groupTilesIntoRects(tiles, tilesX, tilesY) {
  if (tiles.length === 0) return [];
  const grid = new Uint8Array(tilesX * tilesY);
  const deltaGrid = new Float32Array(tilesX * tilesY);
  for (const t of tiles) {
    grid[t.tileY * tilesX + t.tileX] = 1;
    deltaGrid[t.tileY * tilesX + t.tileX] = t.delta;
  }
  const visited = new Uint8Array(tilesX * tilesY);
  const rects = [];
  for (const start of tiles) {
    const startKey = start.tileY * tilesX + start.tileX;
    if (visited[startKey]) continue;
    const stack = [startKey];
    visited[startKey] = 1;
    let minX = start.tileX, maxX = start.tileX;
    let minY = start.tileY, maxY = start.tileY;
    let sumDelta = 0;
    let count = 0;
    while (stack.length > 0) {
      const key = stack.pop();
      const x = key % tilesX;
      const y = Math.floor(key / tilesX);
      sumDelta += deltaGrid[key];
      count++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || nx >= tilesX || ny < 0 || ny >= tilesY) continue;
          const nk = ny * tilesX + nx;
          if (visited[nk] || !grid[nk]) continue;
          visited[nk] = 1;
          stack.push(nk);
        }
      }
    }
    rects.push({
      x: minX * REGION_TILE_PX,
      y: minY * REGION_TILE_PX,
      w: (maxX - minX + 1) * REGION_TILE_PX,
      h: (maxY - minY + 1) * REGION_TILE_PX,
      delta: count > 0 ? sumDelta / count : 0
    });
  }
  return rects;
}
function compareRegionPages(pdfium2, docA, pageA, widthAPts, heightAPts, docB, pageB, widthBPts, heightBPts) {
  const wPts = Math.max(widthAPts, widthBPts);
  const hPts = Math.max(heightAPts, heightBPts);
  const widthPx = Math.max(1, Math.round(wPts * REGION_DPI / 72));
  const heightPx = Math.max(1, Math.round(hPts * REGION_DPI / 72));
  const a = renderPageToRgba(pdfium2, docA, pageA, widthPx, heightPx);
  const b = renderPageToRgba(pdfium2, docB, pageB, widthPx, heightPx);
  const tilesX = Math.ceil(widthPx / REGION_TILE_PX);
  const tilesY = Math.ceil(heightPx / REGION_TILE_PX);
  const tiles = diffPixelsByTile(a.rgba, b.rgba, widthPx, heightPx);
  const rects = groupTilesIntoRects(tiles, tilesX, tilesY);
  const ptPerPx = 72 / REGION_DPI;
  return rects.map((r) => ({
    type: "region",
    pageA,
    pageB,
    rectA: {
      left: r.x * ptPerPx,
      top: r.y * ptPerPx,
      right: (r.x + r.w) * ptPerPx,
      bottom: (r.y + r.h) * ptPerPx
    },
    rectB: {
      left: r.x * ptPerPx,
      top: r.y * ptPerPx,
      right: (r.x + r.w) * ptPerPx,
      bottom: (r.y + r.h) * ptPerPx
    },
    pixelDelta: r.delta
  }));
}
function compareTextPages(pageA, textA, charsA, pageB, textB, charsB) {
  const tokensA = tokenise(textA);
  const tokensB = tokenise(textB);
  const ops = lcsDiff(tokensA, tokensB, (x, y) => x.text.toLowerCase() === y.text.toLowerCase());
  const eqCount = ops.reduce((n, op) => n + (op.kind === "eq" ? 1 : 0), 0);
  const maxTokens = Math.max(tokensA.length, tokensB.length);
  const similarity = maxTokens > 0 ? eqCount / maxTokens : 1;
  if (maxTokens >= TEXT_PAGE_FULL_REPLACE_MIN_TOKENS && similarity < TEXT_PAGE_FULL_REPLACE_THRESHOLD) {
    const rectA = unionCharRects(charsA, 0, charsA.length);
    const rectB = unionCharRects(charsB, 0, charsB.length);
    const snipBefore = textA.length > FULL_REPLACE_SNIPPET_LIMIT ? textA.slice(0, FULL_REPLACE_SNIPPET_LIMIT) + "\u2026" : textA;
    const snipAfter = textB.length > FULL_REPLACE_SNIPPET_LIMIT ? textB.slice(0, FULL_REPLACE_SNIPPET_LIMIT) + "\u2026" : textB;
    return [{
      type: "replace",
      pageA,
      pageB,
      ...rectA ? { rectA } : {},
      ...rectB ? { rectB } : {},
      textBefore: snipBefore,
      textAfter: snipAfter
    }];
  }
  const blocks = [];
  for (const op of ops) {
    if (op.kind === "eq") {
      blocks.push({ kind: "eq" });
    } else if (op.kind === "del") {
      const last = blocks[blocks.length - 1];
      if (last && last.kind === "del") {
        last.dels.push({ tok: op.a, idx: op.aIdx });
      } else if (last && last.kind === "ins") {
        blocks[blocks.length - 1] = {
          kind: "rep",
          dels: [{ tok: op.a, idx: op.aIdx }],
          inss: last.inss
        };
      } else if (last && last.kind === "rep") {
        last.dels.push({ tok: op.a, idx: op.aIdx });
      } else {
        blocks.push({ kind: "del", dels: [{ tok: op.a, idx: op.aIdx }] });
      }
    } else {
      const last = blocks[blocks.length - 1];
      if (last && last.kind === "ins") {
        last.inss.push({ tok: op.b, idx: op.bIdx });
      } else if (last && last.kind === "del") {
        blocks[blocks.length - 1] = {
          kind: "rep",
          dels: last.dels,
          inss: [{ tok: op.b, idx: op.bIdx }]
        };
      } else if (last && last.kind === "rep") {
        last.inss.push({ tok: op.b, idx: op.bIdx });
      } else {
        blocks.push({ kind: "ins", inss: [{ tok: op.b, idx: op.bIdx }] });
      }
    }
  }
  const changes = [];
  for (const block of blocks) {
    if (block.kind === "eq") continue;
    if (block.kind === "del") {
      const first = block.dels[0].tok;
      const last = block.dels[block.dels.length - 1].tok;
      const rect = unionCharRects(charsA, first.charStart, last.charEnd);
      const text = block.dels.map((d) => d.tok.text).join(" ");
      changes.push({
        type: "delete",
        pageA,
        pageB,
        ...rect ? { rectA: rect } : {},
        textBefore: text
      });
    } else if (block.kind === "ins") {
      const first = block.inss[0].tok;
      const last = block.inss[block.inss.length - 1].tok;
      const rect = unionCharRects(charsB, first.charStart, last.charEnd);
      const text = block.inss.map((d) => d.tok.text).join(" ");
      changes.push({
        type: "insert",
        pageA,
        pageB,
        ...rect ? { rectB: rect } : {},
        textAfter: text
      });
    } else {
      const dFirst = block.dels[0].tok;
      const dLast = block.dels[block.dels.length - 1].tok;
      const iFirst = block.inss[0].tok;
      const iLast = block.inss[block.inss.length - 1].tok;
      const rectA = unionCharRects(charsA, dFirst.charStart, dLast.charEnd);
      const rectB = unionCharRects(charsB, iFirst.charStart, iLast.charEnd);
      changes.push({
        type: "replace",
        pageA,
        pageB,
        ...rectA ? { rectA } : {},
        ...rectB ? { rectB } : {},
        textBefore: block.dels.map((d) => d.tok.text).join(" "),
        textAfter: block.inss.map((d) => d.tok.text).join(" ")
      });
    }
  }
  return changes;
}
function compareDocuments(pdfium2, docHandleA, pageCountA, pageSizesA, docHandleB, pageCountB, pageSizesB) {
  const fingerprintsA = [];
  const textsA = [];
  for (let i2 = 0; i2 < pageCountA; i2++) {
    const text = extractPageText(pdfium2, docHandleA, i2);
    textsA.push(text);
    fingerprintsA.push(normaliseText(text));
  }
  const fingerprintsB = [];
  const textsB = [];
  for (let i2 = 0; i2 < pageCountB; i2++) {
    const text = extractPageText(pdfium2, docHandleB, i2);
    textsB.push(text);
    fingerprintsB.push(normaliseText(text));
  }
  const pageOps = lcsDiff(fingerprintsA, fingerprintsB);
  const pageDiffs = [];
  let totalChanges = 0;
  for (const op of pageOps) {
    if (op.kind === "eq") {
      const aIdx = op.aIdx;
      const bIdx = op.bIdx;
      const hasTextA = textsA[aIdx].length >= TEXT_PAGE_MIN_CHARS;
      const hasTextB = textsB[bIdx].length >= TEXT_PAGE_MIN_CHARS;
      if (hasTextA && hasTextB) {
        const charsA = extractPageCharInfo(pdfium2, docHandleA, aIdx);
        const charsB = extractPageCharInfo(pdfium2, docHandleB, bIdx);
        const changes = compareTextPages(
          aIdx,
          textsA[aIdx],
          charsA,
          bIdx,
          textsB[bIdx],
          charsB
        );
        totalChanges += changes.length;
        pageDiffs.push({
          pageA: aIdx,
          pageB: bIdx,
          mode: changes.length === 0 ? "identical" : "text",
          changes
        });
      } else {
        pageDiffs.push({
          pageA: aIdx,
          pageB: bIdx,
          mode: "identical",
          changes: []
        });
      }
      continue;
    }
    if (op.kind === "del") {
      pageDiffs.push({
        pageA: op.aIdx,
        pageB: null,
        mode: "deleted",
        changes: []
      });
      totalChanges++;
      continue;
    }
    if (op.kind === "ins") {
      pageDiffs.push({
        pageA: null,
        pageB: op.bIdx,
        mode: "inserted",
        changes: []
      });
      totalChanges++;
      continue;
    }
  }
  function comparePagePair(aIdx, bIdx) {
    const textA = textsA[aIdx];
    const textB = textsB[bIdx];
    const hasTextA = textA.length >= TEXT_PAGE_MIN_CHARS;
    const hasTextB = textB.length >= TEXT_PAGE_MIN_CHARS;
    if (hasTextA && hasTextB) {
      const charsA = extractPageCharInfo(pdfium2, docHandleA, aIdx);
      const charsB = extractPageCharInfo(pdfium2, docHandleB, bIdx);
      const changes = compareTextPages(
        aIdx,
        textA,
        charsA,
        bIdx,
        textB,
        charsB
      );
      totalChanges += changes.length;
      return {
        pageA: aIdx,
        pageB: bIdx,
        mode: changes.length === 0 ? "identical" : "text",
        changes
      };
    }
    if (!hasTextA && !hasTextB) {
      const sizeA = pageSizesA[aIdx];
      const sizeB = pageSizesB[bIdx];
      const changes = compareRegionPages(
        pdfium2,
        docHandleA,
        aIdx,
        sizeA.width,
        sizeA.height,
        docHandleB,
        bIdx,
        sizeB.width,
        sizeB.height
      );
      totalChanges += changes.length;
      return {
        pageA: aIdx,
        pageB: bIdx,
        mode: changes.length === 0 ? "identical" : "region",
        changes
      };
    }
    totalChanges += 1;
    return {
      pageA: aIdx,
      pageB: bIdx,
      mode: "mismatched",
      changes: [{
        type: "replace",
        pageA: aIdx,
        pageB: bIdx,
        textBefore: hasTextA ? textA : "(scanned page)",
        textAfter: hasTextB ? textB : "(scanned page)"
      }]
    };
  }
  const coalesced = [];
  let i = 0;
  while (i < pageDiffs.length) {
    const cur = pageDiffs[i];
    if (cur.mode !== "inserted" && cur.mode !== "deleted") {
      coalesced.push(cur);
      i++;
      continue;
    }
    const runStart = i;
    while (i < pageDiffs.length && (pageDiffs[i].mode === "inserted" || pageDiffs[i].mode === "deleted")) {
      i++;
    }
    const run = pageDiffs.slice(runStart, i);
    const dels = [];
    const inss = [];
    for (const d of run) {
      if (d.mode === "deleted") dels.push(d);
      else inss.push(d);
    }
    totalChanges -= run.length;
    const pairCount = Math.min(dels.length, inss.length);
    for (let k = 0; k < pairCount; k++) {
      const aIdx = dels[k].pageA;
      const bIdx = inss[k].pageB;
      coalesced.push(comparePagePair(aIdx, bIdx));
    }
    for (let k = pairCount; k < dels.length; k++) {
      coalesced.push(dels[k]);
      totalChanges += 1;
    }
    for (let k = pairCount; k < inss.length; k++) {
      coalesced.push(inss[k]);
      totalChanges += 1;
    }
  }
  return {
    pageCountA,
    pageCountB,
    pageDiffs: coalesced,
    totalChanges
  };
}

// src/worker/merge-split-ops.ts
function getPageCount(pdfium2, doc) {
  return pdfium2.fn._FPDF_GetPageCount(doc);
}
function createNewDocument(pdfium2) {
  const doc = pdfium2.fn._FPDF_CreateNewDocument();
  if (doc === 0) {
    throw new Error("FPDF_CreateNewDocument failed");
  }
  return doc;
}
function closeDocument(pdfium2, doc) {
  pdfium2.fn._FPDF_CloseDocument(doc);
}
function mergeDocuments(pdfium2, sourceHandles) {
  if (sourceHandles.length === 0) {
    throw new Error("No source documents provided for merge");
  }
  const dest = createNewDocument(pdfium2);
  try {
    let insertIndex = 0;
    for (const source of sourceHandles) {
      const result = pdfium2.fn._FPDF_ImportPages(
        dest,
        source,
        0,
        // null = all pages
        insertIndex
      );
      if (!result) {
        throw new Error("FPDF_ImportPages failed during merge");
      }
      insertIndex += getPageCount(pdfium2, source);
    }
    return saveDocumentAsCopy(pdfium2, dest);
  } finally {
    closeDocument(pdfium2, dest);
  }
}
function splitDocument(pdfium2, sourceDoc, ranges) {
  if (ranges.length === 0) {
    throw new Error("No ranges provided for split");
  }
  const results = [];
  for (const range of ranges) {
    const pageRange = `${range.start + 1}-${range.end + 1}`;
    const dest = createNewDocument(pdfium2);
    try {
      var _stack = [];
      try {
        const rangeAlloc = __using(_stack, pdfium2.memory.toByteString(pageRange));
        const ok = pdfium2.fn._FPDF_ImportPages(dest, sourceDoc, rangeAlloc.ptr, 0);
        if (!ok) {
          throw new Error(`FPDF_ImportPages failed for range ${pageRange}`);
        }
        results.push(saveDocumentAsCopy(pdfium2, dest));
      } catch (_) {
        var _error = _, _hasError = true;
      } finally {
        __callDispose(_stack, _error, _hasError);
      }
    } finally {
      closeDocument(pdfium2, dest);
    }
  }
  return results;
}
function extractPages(pdfium2, sourceDoc, pageIndices) {
  if (pageIndices.length === 0) {
    throw new Error("No page indices provided for extraction");
  }
  const dest = createNewDocument(pdfium2);
  try {
    var _stack = [];
    try {
      const byteLen = pageIndices.length * 4;
      const indicesAlloc = __using(_stack, pdfium2.memory.alloc(byteLen));
      for (let i = 0; i < pageIndices.length; i++) {
        pdfium2.module.setValue(indicesAlloc.ptr + i * 4, pageIndices[i], "i32");
      }
      const ok = pdfium2.fn._FPDF_ImportPagesByIndex(
        dest,
        sourceDoc,
        indicesAlloc.ptr,
        pageIndices.length,
        0
        // insert at beginning
      );
      if (!ok) {
        throw new Error("FPDF_ImportPagesByIndex failed");
      }
      return saveDocumentAsCopy(pdfium2, dest);
    } catch (_) {
      var _error = _, _hasError = true;
    } finally {
      __callDispose(_stack, _error, _hasError);
    }
  } finally {
    closeDocument(pdfium2, dest);
  }
}

// src/worker/crypto-ops.ts
var OFF_STATUS = 0;
var OFF_INTEGRITY_VALID = 4;
var OFF_SIGNATURE_VALID = 8;
var OFF_CERTIFICATE_VALID = 12;
var OFF_IS_TIMESTAMPED = 16;
var OFF_IS_EXPIRED = 20;
var OFF_IS_SELF_SIGNED = 24;
var OFF_HASH_ALG_LEN = 36;
var OFF_HASH_ALG = 40;
var OFF_SUBJECT_LEN = 40 + 64;
var OFF_SUBJECT = 40 + 68;
var OFF_ISSUER_LEN = 40 + 68 + 256;
var OFF_ISSUER = 40 + 68 + 260;
var OFF_SERIAL_LEN = 40 + 68 + 260 + 256;
var OFF_SERIAL = 40 + 68 + 264 + 256;
var OFF_ERROR_LEN = 40 + 68 + 264 + 256 + 128;
var OFF_ERROR = 40 + 68 + 268 + 256 + 128;
var STATUS_VALID = 0;
var STATUS_INVALID = 1;
var STATUS_UNKNOWN = 2;
function readI32(pdfium2, ptr, offset) {
  return pdfium2.module.getValue(ptr + offset, "i32");
}
function readString(pdfium2, ptr, lenOffset, strOffset) {
  const len = readI32(pdfium2, ptr, lenOffset);
  if (len <= 0) return "";
  const bytes = pdfium2.memory.fromHeap(ptr + strOffset, len);
  return new TextDecoder().decode(bytes);
}
function validateSignatureOnHeap(pdfium2, pkcs7Der, pdfHeapPtr, pdfHeapLen, byteRange) {
  var _stack = [];
  try {
    const rangeAlloc = __using(_stack, pdfium2.memory.alloc(byteRange.length * 4));
    for (let i = 0; i < byteRange.length; i++) {
      pdfium2.module.setValue(rangeAlloc.ptr + i * 4, byteRange[i], "i32");
    }
    const hashAlgAlloc = __using(_stack, pdfium2.memory.toByteString("SHA-256"));
    const hashOutAlloc = __using(_stack, pdfium2.memory.alloc(64));
    const hashLen = pdfium2.fn._lector_crypto_hash_byte_range(
      pdfHeapPtr,
      pdfHeapLen,
      rangeAlloc.ptr,
      byteRange.length,
      hashAlgAlloc.ptr,
      hashOutAlloc.ptr,
      64
    );
    if (hashLen <= 0) {
      return {
        status: "error",
        signerCertificate: null,
        hashAlgorithm: "",
        isTimestamped: false,
        integrityValid: false,
        signatureValid: false,
        certificateValid: false,
        errorMessage: "Failed to hash PDF byte range"
      };
    }
    const pkcs7Alloc = __using(_stack, pdfium2.memory.toHeap(pkcs7Der.buffer));
    const resultBuf = pdfium2.fn._lector_crypto_alloc_result();
    try {
      pdfium2.fn._lector_crypto_validate_pkcs7(
        pkcs7Alloc.ptr,
        pkcs7Der.length,
        hashOutAlloc.ptr,
        hashLen,
        hashAlgAlloc.ptr,
        resultBuf
      );
      const statusCode = readI32(pdfium2, resultBuf, OFF_STATUS);
      const status = statusCode === STATUS_VALID ? "valid" : statusCode === STATUS_INVALID ? "invalid" : statusCode === STATUS_UNKNOWN ? "unknown" : "error";
      const subject = readString(pdfium2, resultBuf, OFF_SUBJECT_LEN, OFF_SUBJECT);
      const issuer = readString(pdfium2, resultBuf, OFF_ISSUER_LEN, OFF_ISSUER);
      const serial = readString(pdfium2, resultBuf, OFF_SERIAL_LEN, OFF_SERIAL);
      const hashAlg = readString(pdfium2, resultBuf, OFF_HASH_ALG_LEN, OFF_HASH_ALG);
      const errorMsg = readString(pdfium2, resultBuf, OFF_ERROR_LEN, OFF_ERROR);
      const signerCertificate = subject ? {
        subject,
        issuer,
        serialNumber: serial,
        isExpired: readI32(pdfium2, resultBuf, OFF_IS_EXPIRED) !== 0,
        isSelfSigned: readI32(pdfium2, resultBuf, OFF_IS_SELF_SIGNED) !== 0
      } : null;
      return {
        status,
        signerCertificate,
        hashAlgorithm: hashAlg || "SHA-256",
        isTimestamped: readI32(pdfium2, resultBuf, OFF_IS_TIMESTAMPED) !== 0,
        integrityValid: readI32(pdfium2, resultBuf, OFF_INTEGRITY_VALID) !== 0,
        signatureValid: readI32(pdfium2, resultBuf, OFF_SIGNATURE_VALID) !== 0,
        certificateValid: readI32(pdfium2, resultBuf, OFF_CERTIFICATE_VALID) !== 0,
        errorMessage: errorMsg || void 0
      };
    } finally {
      pdfium2.fn._lector_crypto_free(resultBuf);
    }
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}

// src/worker/acroform-js-ops.ts
function getJavaScriptActionCount(pdfium2, docHandle) {
  return pdfium2.fn._FPDFDoc_GetJavaScriptActionCount(docHandle);
}
function getJavaScriptAction(pdfium2, docHandle, index) {
  const action = pdfium2.fn._FPDFDoc_GetJavaScriptAction(docHandle, index);
  if (action === 0) {
    throw new Error(`Failed to get JavaScript action at index ${index}`);
  }
  try {
    const nameLen = pdfium2.fn._FPDFJavaScriptAction_GetName(action, 0, 0);
    let name = "";
    if (nameLen > 2) {
      var _stack = [];
      try {
        const nameBuf = __using(_stack, pdfium2.memory.alloc(nameLen));
        pdfium2.fn._FPDFJavaScriptAction_GetName(action, nameBuf.ptr, nameLen);
        name = pdfium2.memory.fromWideString(nameBuf.ptr) || "";
      } catch (_) {
        var _error = _, _hasError = true;
      } finally {
        __callDispose(_stack, _error, _hasError);
      }
    }
    const scriptLen = pdfium2.fn._FPDFJavaScriptAction_GetScript(action, 0, 0);
    let script = "";
    if (scriptLen > 2) {
      var _stack2 = [];
      try {
        const scriptBuf = __using(_stack2, pdfium2.memory.alloc(scriptLen));
        pdfium2.fn._FPDFJavaScriptAction_GetScript(action, scriptBuf.ptr, scriptLen);
        script = pdfium2.memory.fromWideString(scriptBuf.ptr) || "";
      } catch (_2) {
        var _error2 = _2, _hasError2 = true;
      } finally {
        __callDispose(_stack2, _error2, _hasError2);
      }
    }
    return { name, script };
  } finally {
    pdfium2.fn._FPDFDoc_CloseJavaScriptAction(action);
  }
}
function getAllJavaScriptActions(pdfium2, docHandle) {
  const count = getJavaScriptActionCount(pdfium2, docHandle);
  const actions = [];
  for (let i = 0; i < count; i++) {
    try {
      actions.push(getJavaScriptAction(pdfium2, docHandle, i));
    } catch {
    }
  }
  return actions;
}
function createJSRuntime(pdfium2) {
  const handle = pdfium2.fn._lector_js_create_runtime();
  if (handle === 0) {
    throw new Error("Failed to create QuickJS runtime");
  }
  return handle;
}
function destroyJSRuntime(pdfium2, handle) {
  pdfium2.fn._lector_js_destroy_runtime(handle);
}
function evalScript(pdfium2, handle, script) {
  var _stack = [];
  try {
    const scriptAlloc = __using(_stack, pdfium2.memory.toByteString(script));
    const result = pdfium2.fn._lector_js_eval(handle, scriptAlloc.ptr, scriptAlloc.size - 1);
    return result === 0;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function getGlobalString(pdfium2, handle, varName) {
  var _stack = [];
  try {
    const nameAlloc = __using(_stack, pdfium2.memory.toByteString(varName));
    const resultPtr = pdfium2.fn._lector_js_get_result(handle, nameAlloc.ptr);
    if (resultPtr === 0) return null;
    let str = "";
    let offset = 0;
    while (true) {
      const byte = pdfium2.module.HEAPU8[resultPtr + offset];
      if (byte === 0 || byte === void 0) break;
      str += String.fromCharCode(byte);
      offset++;
    }
    pdfium2.fn._lector_js_free_result(resultPtr);
    return str;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}

// src/worker/linearization-ops.ts
function createLinearContext(pdfium2, fileSize, initialData) {
  var _stack = [];
  try {
    const dataAlloc = __using(_stack, pdfium2.memory.toHeap(initialData.buffer));
    const handle = pdfium2.fn._lector_linear_create(fileSize, dataAlloc.ptr, initialData.length);
    if (handle === 0) {
      throw new Error("Failed to create linearization context");
    }
    return handle;
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function feedLinearData(pdfium2, handle, offset, data) {
  var _stack = [];
  try {
    const dataAlloc = __using(_stack, pdfium2.memory.toHeap(data.buffer));
    pdfium2.fn._lector_linear_feed(handle, offset, dataAlloc.ptr, data.length);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function isLinearized(pdfium2, handle) {
  return pdfium2.fn._lector_linear_is_linearized(handle);
}
function isDocAvail(pdfium2, handle) {
  var _stack = [];
  try {
    const maxHints = 128;
    const hintsAlloc = __using(_stack, pdfium2.memory.alloc(maxHints * 4));
    const countAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const available = pdfium2.fn._lector_linear_is_doc_avail(
      handle,
      hintsAlloc.ptr,
      maxHints,
      countAlloc.ptr
    ) === 1;
    const hintCount = pdfium2.module.getValue(countAlloc.ptr, "i32");
    const hints = [];
    for (let i = 0; i < hintCount; i += 2) {
      hints.push({
        offset: pdfium2.module.getValue(hintsAlloc.ptr + i * 4, "i32"),
        length: pdfium2.module.getValue(hintsAlloc.ptr + (i + 1) * 4, "i32")
      });
    }
    return { available, hints };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function isPageAvail(pdfium2, handle, pageIndex) {
  var _stack = [];
  try {
    const maxHints = 128;
    const hintsAlloc = __using(_stack, pdfium2.memory.alloc(maxHints * 4));
    const countAlloc = __using(_stack, pdfium2.memory.alloc(4));
    const available = pdfium2.fn._lector_linear_is_page_avail(
      handle,
      pageIndex,
      hintsAlloc.ptr,
      maxHints,
      countAlloc.ptr
    ) === 1;
    const hintCount = pdfium2.module.getValue(countAlloc.ptr, "i32");
    const hints = [];
    for (let i = 0; i < hintCount; i += 2) {
      hints.push({
        offset: pdfium2.module.getValue(hintsAlloc.ptr + i * 4, "i32"),
        length: pdfium2.module.getValue(hintsAlloc.ptr + (i + 1) * 4, "i32")
      });
    }
    return { available, hints };
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
function getLinearDocument(pdfium2, handle, password) {
  let pwPtr = 0;
  let pwAlloc = null;
  if (password) {
    pwAlloc = pdfium2.memory.toByteString(password);
    pwPtr = pwAlloc.ptr;
  }
  try {
    const doc = pdfium2.fn._lector_linear_get_document(handle, pwPtr);
    if (doc === 0) {
      throw new Error("Failed to get document from linearization context");
    }
    return doc;
  } finally {
    pwAlloc?.[Symbol.dispose]();
  }
}
function getFirstPageNum(pdfium2, handle) {
  return pdfium2.fn._lector_linear_get_first_page(handle);
}
function destroyLinearContext(pdfium2, handle) {
  pdfium2.fn._lector_linear_destroy(handle);
}

// src/worker/pdfium-worker.ts
var pdfium = null;
var documentStore = null;
var nextHandleId = 1;
var jsRuntimes = /* @__PURE__ */ new Map();
var linearContexts = /* @__PURE__ */ new Map();
function assertReady() {
  if (pdfium === null || documentStore === null) {
    throw new Error("Worker not initialized \u2014 call init() first");
  }
  return { pdfium, store: documentStore };
}
function refreshPageSizes(p, store, docId) {
  var _stack = [];
  try {
    const state = store.resolve(docId);
    const pageCount = p.fn._FPDF_GetPageCount(state.docHandle);
    const pageSizes = [];
    const sizeAlloc = __using(_stack, p.memory.alloc(FS_SIZEF_SIZE));
    for (let i = 0; i < pageCount; i++) {
      p.fn._FPDF_GetPageSizeByIndexF(state.docHandle, i, sizeAlloc.ptr);
      const size = p.memory.readSizeF(sizeAlloc.ptr);
      pageSizes.push({ width: size.width, height: size.height });
    }
    store.updatePageInfo(docId, pageSizes);
  } catch (_) {
    var _error = _, _hasError = true;
  } finally {
    __callDispose(_stack, _error, _hasError);
  }
}
var workerApi = {
  async init(wasmUrl, wasmJsUrl) {
    try {
      const maxThreads = Math.min(navigator.hardwareConcurrency ?? 4, 4);
      Object.defineProperty(navigator, "hardwareConcurrency", {
        value: maxThreads,
        configurable: true
      });
      const loaderModule = await import(
        /* @vite-ignore */
        wasmJsUrl
      );
      const createModule = loaderModule.default;
      const { createInstantiateWasmHook } = await import("./wasm-cache-IXSAKVDT.js");
      pdfium = await createPdfiumInstance(createModule, {
        locateFile: () => wasmUrl,
        mainScriptUrlOrBlob: wasmJsUrl,
        instantiateWasm: createInstantiateWasmHook(wasmUrl)
      });
      documentStore = new DocumentStore(pdfium);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async openDocument(data, password) {
    try {
      const { pdfium: p, store } = assertReady();
      const hashBuf = crypto.subtle ? await crypto.subtle.digest("SHA-256", data) : sha256(data);
      const hashBytes = new Uint8Array(hashBuf);
      let sha2562 = "";
      for (let i = 0; i < hashBytes.length; i++) {
        sha2562 += hashBytes[i].toString(16).padStart(2, "0");
      }
      const alloc = p.memory.toHeap(data);
      let passwordPtr = 0;
      let passwordAlloc = null;
      if (password !== void 0 && password.length > 0) {
        const pwdAlloc = p.memory.toByteString(password);
        passwordPtr = pwdAlloc.ptr;
        passwordAlloc = pwdAlloc;
      }
      let formInfoAlloc = null;
      let registered = false;
      try {
        var _stack = [];
        try {
          const docHandle = p.fn._FPDF_LoadMemDocument(
            alloc.ptr,
            alloc.size,
            passwordPtr
          );
          checkHandle(docHandle, () => p.fn._FPDF_GetLastError(), "FPDF_LoadMemDocument");
          const pageCount = p.fn._FPDF_GetPageCount(docHandle);
          const pageSizes = [];
          const sizeAlloc = __using(_stack, p.memory.alloc(FS_SIZEF_SIZE));
          for (let i = 0; i < pageCount; i++) {
            p.fn._FPDF_GetPageSizeByIndexF(docHandle, i, sizeAlloc.ptr);
            const size = p.memory.readSizeF(sizeAlloc.ptr);
            pageSizes.push({ width: size.width, height: size.height });
          }
          formInfoAlloc = p.memory.alloc(512);
          const ffiView = p.memory.heapView(formInfoAlloc.ptr, 512);
          ffiView.fill(0);
          ffiView[0] = 2;
          const formHandle = p.fn._FPDFDOC_InitFormFillEnvironment(
            docHandle,
            formInfoAlloc.ptr
          );
          const docId = store.register({
            docHandle,
            formHandle,
            formInfoAlloc,
            pageCount,
            pageSizes,
            pdfAlloc: alloc,
            sha256: sha2562
          });
          registered = true;
          return docId;
        } catch (_) {
          var _error = _, _hasError = true;
        } finally {
          __callDispose(_stack, _error, _hasError);
        }
      } finally {
        passwordAlloc?.[Symbol.dispose]();
        if (!registered) {
          alloc[Symbol.dispose]();
          formInfoAlloc?.[Symbol.dispose]();
        }
      }
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getDocumentHash(docId) {
    try {
      const { store } = assertReady();
      return store.resolve(docId).sha256;
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async closeDocument(docId) {
    try {
      const { store } = assertReady();
      store.release(docId);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getPageCount(docId) {
    try {
      const { store } = assertReady();
      return store.resolve(docId).pageCount;
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getPageSize(docId, pageIndex) {
    try {
      const { store } = assertReady();
      const state = store.resolve(docId);
      const size = state.pageSizes[pageIndex];
      if (size === void 0) {
        throw new Error(`Page index ${pageIndex} out of range (0..${state.pageCount - 1})`);
      }
      return size;
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getAllPageSizes(docId) {
    try {
      const { store } = assertReady();
      return store.resolve(docId).pageSizes;
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async renderPage(docId, pageIndex, width, height, options) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const resolved = { ...DEFAULT_RENDER_OPTIONS, ...options };
      const bitmap = await renderPageToImageBitmap(
        p,
        state.docHandle,
        pageIndex,
        width,
        height,
        resolved,
        state.formHandle
      );
      return transfer(bitmap, [bitmap]);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async renderPageTile(docId, pageIndex, tileX, tileY, tileW, tileH, fullW, fullH, options) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const resolved = { ...DEFAULT_RENDER_OPTIONS, ...options };
      const bitmap = await renderPageTileToImageBitmap(
        p,
        state.docHandle,
        pageIndex,
        tileX,
        tileY,
        tileW,
        tileH,
        fullW,
        fullH,
        resolved,
        state.formHandle
      );
      return transfer(bitmap, [bitmap]);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async cancelTask(_taskId) {
    return false;
  },
  // ── Annotation CRUD ──
  async getAnnotations(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return readPageAnnotations(p, state.docHandle, pageIndex, state.formHandle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async createAnnotation(docId, pageIndex, data) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return createAnnotation(p, state.docHandle, pageIndex, data);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async updateAnnotation(docId, pageIndex, annotIndex, patch) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return updateAnnotation(p, state.docHandle, pageIndex, annotIndex, patch);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async deleteAnnotation(docId, pageIndex, annotIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      deleteAnnotation(p, state.docHandle, pageIndex, annotIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Form fields ──
  async getFormFields(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return readPageFormFields(p, state.docHandle, state.formHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async setFormFieldValue(docId, pageIndex, fieldName, value) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      setFormFieldValue(p, state.docHandle, state.formHandle, pageIndex, fieldName, value);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async setComboBoxByIndex(docId, pageIndex, annotIndex, optionIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      setComboBoxByIndex(p, state.docHandle, state.formHandle, pageIndex, annotIndex, optionIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async clickFormWidget(docId, pageIndex, pageX, pageY) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      clickFormWidget(p, state.docHandle, state.formHandle, pageIndex, pageX, pageY);
      return readPageFormFields(p, state.docHandle, state.formHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Text extraction ──
  async getPageText(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return extractPageText(p, state.docHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getPageCharInfo(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return extractPageCharInfo(p, state.docHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async searchPage(docId, pageIndex, query, flags) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return searchPageText(p, state.docHandle, pageIndex, query, flags);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getTextRects(docId, pageIndex, charIndex, count) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getTextRects(p, state.docHandle, pageIndex, charIndex, count);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getCharIndexAtPos(docId, pageIndex, x, y, tolerance) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getCharIndexAtPos(p, state.docHandle, pageIndex, x, y, tolerance);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Navigation ──
  async getBookmarks(docId) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return readBookmarkTree(p, state.docHandle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async addBookmark(docId, title, pageIndex, insertIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return addBookmark(p, state.docHandle, title, pageIndex, insertIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async deleteBookmark(docId, index) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return deleteBookmark(p, state.docHandle, index);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async moveBookmark(docId, fromIndex, toIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return moveBookmark(p, state.docHandle, fromIndex, toIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async setBookmarkTitle(docId, index, title) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return setBookmarkTitle(p, state.docHandle, index, title);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async setBookmarkDest(docId, index, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return setBookmarkDest(p, state.docHandle, index, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getPageLinks(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return readPageLinks(p, state.docHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getPageWebLinks(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return readPageWebLinks(p, state.docHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Save / Export ──
  async saveAsCopy(docId) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      if (state.formHandle !== 0) {
        const pageCount = p.fn._FPDF_GetPageCount(state.docHandle);
        for (let pi = 0; pi < pageCount; pi++) {
          const fields = readPageFormFields(p, state.docHandle, state.formHandle, pi);
          for (const f of fields) {
            if (f.fieldType === 6) {
              const value = f.fieldValue || "";
              setFormFieldValue(p, state.docHandle, state.formHandle, pi, f.fieldName, "");
              setFormFieldValue(p, state.docHandle, state.formHandle, pi, f.fieldName, value);
            }
          }
        }
      }
      const buf = saveDocumentAsCopy(p, state.docHandle);
      return transfer(buf, [buf]);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async setDocumentPassword(docId, options) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      setDocumentPassword(p, state.docHandle, options);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async signDocument(docId, options) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return await signDocument(p, state.docHandle, options);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async captureRegion(docId, options) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const bitmap = await captureRegion(p, state.docHandle, options);
      return transfer(bitmap, [bitmap]);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async compareDocuments(docIdA, docIdB) {
    try {
      const { pdfium: p, store } = assertReady();
      const stateA = store.resolve(docIdA);
      const stateB = store.resolve(docIdB);
      return compareDocuments(
        p,
        stateA.docHandle,
        stateA.pageCount,
        stateA.pageSizes,
        stateB.docHandle,
        stateB.pageCount,
        stateB.pageSizes
      );
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async exportXfdf(docId) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const allAnnotations = [];
      const pageCount = p.fn._FPDF_GetPageCount(state.docHandle);
      for (let i = 0; i < pageCount; i++) {
        const pageAnnots = readPageAnnotations(p, state.docHandle, i, state.formHandle);
        allAnnotations.push(...pageAnnots);
      }
      return exportXfdf(p, state.docHandle, allAnnotations);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Merge / Split ──
  async mergeDocuments(docIds) {
    try {
      const { pdfium: p, store } = assertReady();
      const handles = docIds.map((id) => store.resolve(id).docHandle);
      const buf = mergeDocuments(p, handles);
      return transfer(buf, [buf]);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async splitDocument(docId, ranges) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const bufs = splitDocument(p, state.docHandle, ranges);
      return bufs.map((buf) => transfer(buf, [buf]));
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async extractPages(docId, pageIndices) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const buf = extractPages(p, state.docHandle, pageIndices);
      return transfer(buf, [buf]);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Page operations ──
  async deletePage(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      deletePage(p, state.docHandle, pageIndex);
      refreshPageSizes(p, store, docId);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async insertBlankPage(docId, pageIndex, width, height) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      insertBlankPage(p, state.docHandle, pageIndex, width, height);
      refreshPageSizes(p, store, docId);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async rotatePage(docId, pageIndex, rotation) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      rotatePage(p, state.docHandle, pageIndex, rotation);
      refreshPageSizes(p, store, docId);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getPageRotation(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getPageRotation(p, state.docHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async movePage(docId, fromIndex, toIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      movePage(p, state.docHandle, fromIndex, toIndex);
      refreshPageSizes(p, store, docId);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async duplicatePage(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      duplicatePage(p, state.docHandle, pageIndex);
      refreshPageSizes(p, store, docId);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async flattenPage(docId, pageIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return flattenPage(p, state.docHandle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async applyRedactions(docId, pageIndex, specs, removeAnnots) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      applyRedactions(p, state.docHandle, pageIndex, specs, removeAnnots);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Signatures ──
  async getSignatureCount(docId) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getSignatureCount(p, state.docHandle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getSignatureInfo(docId, sigIndex) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getSignatureInfo(p, state.docHandle, sigIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Attachments ──
  async getAttachmentCount(docId) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getAttachmentCount(p, state.docHandle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getAttachmentInfo(docId, index) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getAttachmentInfo(p, state.docHandle, index);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getAttachmentData(docId, index) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const buf = getAttachmentData(p, state.docHandle, index);
      return transfer(buf, [buf]);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async addAttachment(docId, name, data) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      addAttachment(p, state.docHandle, name, data);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async deleteAttachment(docId, index) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      deleteAttachment(p, state.docHandle, index);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Layers (OCG) ──
  async getLayers(docId) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getAllLayers(p, state.docHandle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async setLayerVisible(docId, layerIndex, visible) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      setLayerVisible(p, state.docHandle, layerIndex, visible);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Crypto signature validation ──
  async validateSignature(docId, sigIndex, _pdfBytes) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const alloc = state.pdfAlloc;
      if (!alloc) {
        throw new Error(
          "Signature validation is unavailable for progressively-loaded (linearized) documents \u2014 the full document bytes are not resident on the heap."
        );
      }
      const sigInfo = getSignatureInfo(p, state.docHandle, sigIndex);
      return validateSignatureOnHeap(p, sigInfo.contents, alloc.ptr, alloc.size, sigInfo.byteRange);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  /**
   * Validate every signature in a document in a single round-trip.
   *
   * Uses the PDF bytes that pdfium already has resident on the WASM
   * heap (state.pdfAlloc), so no main-thread copy and no per-signature
   * malloc/memcpy is needed. The pdfBytes argument is unused — kept on
   * the API for backwards compatibility — and the caller should still
   * transfer it via Comlink so the main side does not retain a copy.
   */
  async validateAllSignatures(docId, _pdfBytes) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      const count = getSignatureCount(p, state.docHandle);
      if (count === 0) return [];
      const alloc = state.pdfAlloc;
      if (!alloc) {
        throw new Error(
          "Signature validation is unavailable for progressively-loaded (linearized) documents \u2014 the full document bytes are not resident on the heap."
        );
      }
      const heapPtr = alloc.ptr;
      const heapLen = alloc.size;
      const results = [];
      for (let i = 0; i < count; i++) {
        const sigInfo = getSignatureInfo(p, state.docHandle, i);
        results.push(validateSignatureOnHeap(p, sigInfo.contents, heapPtr, heapLen, sigInfo.byteRange));
      }
      return results;
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── AcroForm JavaScript ──
  async getJavaScriptActions(docId) {
    try {
      const { pdfium: p, store } = assertReady();
      const state = store.resolve(docId);
      return getAllJavaScriptActions(p, state.docHandle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async createJSRuntime() {
    try {
      const { pdfium: p } = assertReady();
      const handle = createJSRuntime(p);
      const id = nextHandleId++;
      jsRuntimes.set(id, handle);
      return id;
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async destroyJSRuntime(handleId) {
    try {
      const { pdfium: p } = assertReady();
      const handle = jsRuntimes.get(handleId);
      if (handle !== void 0) {
        destroyJSRuntime(p, handle);
        jsRuntimes.delete(handleId);
      }
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async evalScript(handleId, script) {
    try {
      const { pdfium: p } = assertReady();
      const handle = jsRuntimes.get(handleId);
      if (handle === void 0) throw new Error(`JS runtime ${handleId} not found`);
      return evalScript(p, handle, script);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getJSGlobal(handleId, varName) {
    try {
      const { pdfium: p } = assertReady();
      const handle = jsRuntimes.get(handleId);
      if (handle === void 0) throw new Error(`JS runtime ${handleId} not found`);
      return getGlobalString(p, handle, varName);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  // ── Linearized loading ──
  async createLinearContext(fileSize, initialData) {
    try {
      const { pdfium: p } = assertReady();
      const handle = createLinearContext(p, fileSize, new Uint8Array(initialData));
      const id = nextHandleId++;
      linearContexts.set(id, handle);
      return id;
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async feedLinearData(contextId, offset, data) {
    try {
      const { pdfium: p } = assertReady();
      const handle = linearContexts.get(contextId);
      if (handle === void 0) throw new Error(`Linear context ${contextId} not found`);
      feedLinearData(p, handle, offset, new Uint8Array(data));
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async isLinearized(contextId) {
    try {
      const { pdfium: p } = assertReady();
      const handle = linearContexts.get(contextId);
      if (handle === void 0) throw new Error(`Linear context ${contextId} not found`);
      return isLinearized(p, handle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async isDocAvail(contextId) {
    try {
      const { pdfium: p } = assertReady();
      const handle = linearContexts.get(contextId);
      if (handle === void 0) throw new Error(`Linear context ${contextId} not found`);
      return isDocAvail(p, handle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async isPageAvail(contextId, pageIndex) {
    try {
      const { pdfium: p } = assertReady();
      const handle = linearContexts.get(contextId);
      if (handle === void 0) throw new Error(`Linear context ${contextId} not found`);
      return isPageAvail(p, handle, pageIndex);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getLinearDocument(contextId, password) {
    try {
      const { pdfium: p, store } = assertReady();
      const handle = linearContexts.get(contextId);
      if (handle === void 0) throw new Error(`Linear context ${contextId} not found`);
      const docHandle = getLinearDocument(p, handle, password);
      const pageCount = p.fn._FPDF_GetPageCount(docHandle);
      const pageSizes = [];
      for (let i = 0; i < pageCount; i++) {
        var _stack = [];
        try {
          const sizeAlloc = __using(_stack, p.memory.alloc(FS_SIZEF_SIZE));
          const ok = p.fn._FPDF_GetPageSizeByIndexF(docHandle, i, sizeAlloc.ptr);
          if (ok) {
            pageSizes.push({
              width: p.module.getValue(sizeAlloc.ptr, "float"),
              height: p.module.getValue(sizeAlloc.ptr + 4, "float")
            });
          } else {
            pageSizes.push({ width: 612, height: 792 });
          }
        } catch (_) {
          var _error = _, _hasError = true;
        } finally {
          __callDispose(_stack, _error, _hasError);
        }
      }
      const formInfoAlloc = p.memory.alloc(512);
      const ffiView = p.memory.heapView(formInfoAlloc.ptr, 512);
      ffiView.fill(0);
      ffiView[0] = 2;
      const formHandle = p.fn._FPDFDOC_InitFormFillEnvironment(docHandle, formInfoAlloc.ptr);
      let registered = false;
      try {
        const id = store.register({ docHandle, formHandle, formInfoAlloc, pageCount, pageSizes, pdfAlloc: null, sha256: "" });
        registered = true;
        return id;
      } finally {
        if (!registered) {
          if (formHandle !== 0) p.fn._FPDFDOC_ExitFormFillEnvironment(formHandle);
          formInfoAlloc[Symbol.dispose]();
        }
      }
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async getLinearFirstPage(contextId) {
    try {
      const { pdfium: p } = assertReady();
      const handle = linearContexts.get(contextId);
      if (handle === void 0) throw new Error(`Linear context ${contextId} not found`);
      return getFirstPageNum(p, handle);
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async destroyLinearContext(contextId) {
    try {
      const { pdfium: p } = assertReady();
      const handle = linearContexts.get(contextId);
      if (handle !== void 0) {
        destroyLinearContext(p, handle);
        linearContexts.delete(contextId);
      }
    } catch (err) {
      throw serializePdfiumError(err);
    }
  },
  async destroy() {
    try {
      if (documentStore !== null) {
        documentStore[Symbol.dispose]();
        documentStore = null;
      }
      if (pdfium !== null) {
        pdfium[Symbol.dispose]();
        pdfium = null;
      }
      close();
    } catch (err) {
      throw serializePdfiumError(err);
    }
  }
};
expose(workerApi);
/*! Bundled license information:

comlink/dist/esm/comlink.mjs:
  (**
   * @license
   * Copyright 2019 Google LLC
   * SPDX-License-Identifier: Apache-2.0
   *)
*/
