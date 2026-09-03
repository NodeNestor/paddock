import type { WasmPointer } from '../types/handles.js';
import type { PdfiumModule } from '../types/module.js';
/** A WASM heap allocation that auto-frees via `using`. */
export declare class WasmAlloc implements Disposable {
    #private;
    readonly ptr: WasmPointer;
    readonly size: number;
    constructor(module: PdfiumModule, size: number);
    [Symbol.dispose](): void;
}
/** Allocate `size` bytes on the WASM heap. Returns a Disposable allocation. */
export declare function wasmAlloc(module: PdfiumModule, size: number): WasmAlloc;
/** Allocate `size` bytes. Throws on failure. Non-disposable — caller must free. */
export declare function wasmMalloc(module: PdfiumModule, size: number): WasmPointer;
/** Free a WASM heap pointer. No-op if ptr is 0. */
export declare function wasmFree(module: PdfiumModule, ptr: WasmPointer): void;
//# sourceMappingURL=allocator.d.ts.map