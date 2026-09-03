import type { WasmPointer } from '../types/handles.js';
import type { PdfiumModule } from '../types/module.js';
import { WasmAlloc } from './allocator.js';
/**
 * Copy a JS ArrayBuffer or Uint8Array into the WASM heap.
 * Returns a Disposable allocation with `.ptr` and `.length`.
 *
 * ```ts
 * using buf = toHeap(module, pdfBytes);
 * const doc = module._FPDF_LoadMemDocument(buf.ptr, buf.length, 0 as WasmPointer);
 * // auto-freed when scope exits
 * ```
 */
export declare function toHeap(module: PdfiumModule, data: ArrayBuffer | Uint8Array): WasmAlloc;
/** Copy bytes from the WASM heap into a new Uint8Array. */
export declare function fromHeap(module: PdfiumModule, ptr: WasmPointer, length: number): Uint8Array;
/**
 * Get a zero-copy Uint8Array view into WASM heap memory.
 * WARNING: Invalidated if WASM memory grows (e.g. on malloc).
 */
export declare function heapView(module: PdfiumModule, ptr: WasmPointer, length: number): Uint8Array;
//# sourceMappingURL=buffer.d.ts.map