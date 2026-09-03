/**
 * AcroForm JavaScript execution operations.
 *
 * Wraps the QuickJS-based WASM engine for executing PDF form field
 * scripts (calculations, formatting, validation) in a sandboxed
 * environment. All operations run in the worker thread.
 */
import type { FpdfDocument, PdfiumInstance, WasmPointer } from '@truespar/lector-pdfium-wasm';
export interface PdfJavaScriptAction {
    readonly name: string;
    readonly script: string;
}
/**
 * Get the count of document-level JavaScript actions.
 */
export declare function getJavaScriptActionCount(pdfium: PdfiumInstance, docHandle: FpdfDocument): number;
/**
 * Read a single document-level JavaScript action.
 */
export declare function getJavaScriptAction(pdfium: PdfiumInstance, docHandle: FpdfDocument, index: number): PdfJavaScriptAction;
/**
 * Read all document-level JavaScript actions.
 */
export declare function getAllJavaScriptActions(pdfium: PdfiumInstance, docHandle: FpdfDocument): PdfJavaScriptAction[];
/**
 * Create a new QuickJS sandbox runtime in WASM.
 * Returns an opaque handle pointer.
 */
export declare function createJSRuntime(pdfium: PdfiumInstance): WasmPointer;
/**
 * Destroy a QuickJS sandbox runtime.
 */
export declare function destroyJSRuntime(pdfium: PdfiumInstance, handle: WasmPointer): void;
/**
 * Evaluate a JavaScript string in the sandbox.
 * Returns true on success, false on error.
 */
export declare function evalScript(pdfium: PdfiumInstance, handle: WasmPointer, script: string): boolean;
/**
 * Read a global variable from the sandbox as a string.
 */
export declare function getGlobalString(pdfium: PdfiumInstance, handle: WasmPointer, varName: string): string | null;
//# sourceMappingURL=acroform-js-ops.d.ts.map