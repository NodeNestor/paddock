/**
 * Linearized PDF progressive loading operations.
 *
 * Wraps the custom WASM linearization functions that manage
 * FPDF_FILEACCESS / FX_FILEAVAIL / FX_DOWNLOADHINTS callback
 * structs for pdfium's progressive loading API.
 */
import type { FpdfDocument, PdfiumInstance, WasmPointer } from '@truespar/lector-pdfium-wasm';
export interface DownloadHint {
    readonly offset: number;
    readonly length: number;
}
/**
 * Create a linearization context for progressive loading.
 *
 * @param fileSize      Total file size from HTTP Content-Length.
 * @param initialData   First chunk of the PDF (typically 64KB-1MB).
 * @returns Opaque context handle.
 */
export declare function createLinearContext(pdfium: PdfiumInstance, fileSize: number, initialData: Uint8Array): WasmPointer;
/**
 * Feed a new data chunk into the linearization context.
 */
export declare function feedLinearData(pdfium: PdfiumInstance, handle: WasmPointer, offset: number, data: Uint8Array): void;
/**
 * Check if the PDF is linearized.
 * @returns 1 = linearized, 0 = not linearized, -1 = need more data.
 */
export declare function isLinearized(pdfium: PdfiumInstance, handle: WasmPointer): number;
/**
 * Check if enough document structure is available to open.
 * Returns availability status and any hints for needed byte ranges.
 */
export declare function isDocAvail(pdfium: PdfiumInstance, handle: WasmPointer): {
    available: boolean;
    hints: DownloadHint[];
};
/**
 * Check if a specific page is available for rendering.
 */
export declare function isPageAvail(pdfium: PdfiumInstance, handle: WasmPointer, pageIndex: number): {
    available: boolean;
    hints: DownloadHint[];
};
/**
 * Get the document handle from the linearization context.
 * Call only after isDocAvail returns available: true.
 */
export declare function getLinearDocument(pdfium: PdfiumInstance, handle: WasmPointer, password?: string): FpdfDocument;
/**
 * Get the first available page index.
 */
export declare function getFirstPageNum(pdfium: PdfiumInstance, handle: WasmPointer): number;
/**
 * Destroy the linearization context. Does NOT close the document.
 */
export declare function destroyLinearContext(pdfium: PdfiumInstance, handle: WasmPointer): void;
//# sourceMappingURL=linearization-ops.d.ts.map