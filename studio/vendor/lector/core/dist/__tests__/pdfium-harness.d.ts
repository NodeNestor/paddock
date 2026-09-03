import type { FpdfDocument, PdfiumInstance, WasmPointer } from '@truespar/lector-pdfium-wasm';
import type { PageSize } from '../types/render.js';
/**
 * Initialize pdfium WASM from the browser-served files.
 * Returns a singleton — safe to call multiple times.
 */
export declare function getPdfium(): Promise<PdfiumInstance>;
/**
 * Destroy the shared pdfium instance.
 * Call in afterAll() of the test suite root.
 */
export declare function destroyPdfium(): void;
/** Data for an open test document. */
export interface TestDocument {
    readonly docHandle: FpdfDocument;
    readonly pageCount: number;
    readonly pageSizes: PageSize[];
    /** The raw PDF bytes (keep alive — pdfium doesn't copy). */
    readonly pdfAlloc: {
        ptr: WasmPointer;
        size: number;
        [Symbol.dispose](): void;
    };
}
/**
 * Load a PDF from an ArrayBuffer into pdfium.
 * The caller must close the document when done.
 */
export declare function loadDocument(pdfium: PdfiumInstance, data: ArrayBuffer, password?: string): TestDocument;
/**
 * Close a test document and free its resources.
 */
export declare function closeDocument(pdfium: PdfiumInstance, doc: TestDocument): void;
/**
 * Fetch a test PDF from the file system.
 * In browser mode, we fetch from Vite's server.
 */
export declare function fetchTestPdf(path: string): Promise<ArrayBuffer>;
//# sourceMappingURL=pdfium-harness.d.ts.map