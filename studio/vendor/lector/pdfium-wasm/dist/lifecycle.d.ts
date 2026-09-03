import type { PdfiumExports } from './generated/bindings.js';
import { WasmAlloc } from './memory/allocator.js';
import type { WasmPointer } from './types/handles.js';
import type { PdfiumModule, PdfiumModuleConfig } from './types/module.js';
import type { FsMatrix, FsPointF, FsQuadPointsF, FsRectF, FsSizeF } from './types/structs.js';
/** Memory management helpers bound to a specific module instance. */
export interface PdfiumMemory {
    /** Allocate WASM heap memory. Returns a Disposable — use `using`. */
    alloc(size: number): WasmAlloc;
    /** Write a UTF-8 string to WASM heap. Returns Disposable. */
    toByteString(str: string): WasmAlloc;
    /** Write a UTF-16LE string to WASM heap. Returns Disposable. */
    toWideString(str: string): WasmAlloc;
    /** Read a UTF-8 C string from WASM heap. */
    fromByteString(ptr: WasmPointer): string;
    /** Read a UTF-16LE string from WASM heap. */
    fromWideString(ptr: WasmPointer): string;
    /** Copy JS buffer to WASM heap. Returns Disposable. */
    toHeap(data: ArrayBuffer | Uint8Array): WasmAlloc;
    /** Copy from WASM heap to a new Uint8Array. */
    fromHeap(ptr: WasmPointer, length: number): Uint8Array;
    /** Zero-copy view into WASM heap (invalidated on memory growth). */
    heapView(ptr: WasmPointer, length: number): Uint8Array;
    /** Read an FS_RECTF struct from WASM heap. */
    readRectF(ptr: WasmPointer): FsRectF;
    /** Write an FS_RECTF struct to WASM heap. */
    writeRectF(ptr: WasmPointer, rect: FsRectF): void;
    /** Read an FS_MATRIX struct from WASM heap. */
    readMatrix(ptr: WasmPointer): FsMatrix;
    /** Write an FS_MATRIX struct to WASM heap. */
    writeMatrix(ptr: WasmPointer, matrix: FsMatrix): void;
    /** Read an FS_SIZEF struct from WASM heap. */
    readSizeF(ptr: WasmPointer): FsSizeF;
    /** Write an FS_SIZEF struct to WASM heap. */
    writeSizeF(ptr: WasmPointer, size: FsSizeF): void;
    /** Read an FS_POINTF struct from WASM heap. */
    readPointF(ptr: WasmPointer): FsPointF;
    /** Write an FS_POINTF struct to WASM heap. */
    writePointF(ptr: WasmPointer, point: FsPointF): void;
    /** Read an FS_QUADPOINTSF struct from WASM heap. */
    readQuadPointsF(ptr: WasmPointer): FsQuadPointsF;
    /** Write an FS_QUADPOINTSF struct to WASM heap. */
    writeQuadPointsF(ptr: WasmPointer, qp: FsQuadPointsF): void;
}
/**
 * A fully initialized pdfium instance.
 * Implements Disposable — use `using` for automatic cleanup.
 */
export interface PdfiumInstance extends Disposable {
    /** The raw Emscripten module. */
    readonly module: PdfiumModule;
    /**
     * All 452 pdfium C API functions as direct WASM exports.
     * Zero overhead — no cwrap, no marshaling.
     *
     * ```ts
     * const doc = pdfium.fn._FPDF_LoadMemDocument(ptr, size, pwd);
     * const count = pdfium.fn._FPDF_GetPageCount(doc);
     * ```
     */
    readonly fn: PdfiumExports;
    /** WASM heap memory management with Disposable support. */
    readonly memory: PdfiumMemory;
}
/**
 * Load the pdfium WASM module and initialize the library.
 *
 * Returns a Disposable PdfiumInstance — use `using` for automatic cleanup:
 * ```ts
 * using pdfium = await createPdfiumInstance(createModule);
 * using buf = pdfium.memory.toHeap(pdfBytes);
 * const doc = pdfium.fn._FPDF_LoadMemDocument(buf.ptr, buf.size, 0 as WasmPointer);
 * ```
 *
 * @param createModule The Emscripten factory (default export from pdfium.js)
 * @param config Optional module configuration (locateFile, etc.)
 */
export declare function createPdfiumInstance(createModule: (config?: PdfiumModuleConfig) => Promise<PdfiumModule>, config?: PdfiumModuleConfig): Promise<PdfiumInstance>;
//# sourceMappingURL=lifecycle.d.ts.map