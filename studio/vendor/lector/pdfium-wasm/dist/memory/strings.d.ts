import type { WasmPointer } from '../types/handles.js';
import type { PdfiumModule } from '../types/module.js';
import { WasmAlloc } from './allocator.js';
/**
 * Write a JS string as a null-terminated UTF-8 C string on the WASM heap.
 * Returns a Disposable allocation — use `using` or call `[Symbol.dispose]()`.
 *
 * ```ts
 * using str = toByteString(module, 'hello');
 * module._SomeFunction(str.ptr);
 * // auto-freed when scope exits
 * ```
 */
export declare function toByteString(module: PdfiumModule, str: string): WasmAlloc;
/**
 * Write a JS string as a null-terminated UTF-16LE wide string on the WASM heap.
 * Returns a Disposable allocation — use `using` or call `[Symbol.dispose]()`.
 */
export declare function toWideString(module: PdfiumModule, str: string): WasmAlloc;
/** Read a null-terminated UTF-8 C string from the WASM heap. */
export declare function fromByteString(module: PdfiumModule, ptr: WasmPointer): string;
/** Read a null-terminated UTF-16LE wide string from the WASM heap. */
export declare function fromWideString(module: PdfiumModule, ptr: WasmPointer): string;
//# sourceMappingURL=strings.d.ts.map