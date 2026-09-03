import { FpdfError } from './types/enums.js';
/** Typed error for pdfium API failures. */
export declare class PdfiumError extends Error {
    readonly code: FpdfError;
    constructor(code: FpdfError, context?: string);
}
/**
 * Assert an FPDF_BOOL result is truthy (non-zero).
 * Throws PdfiumError with the last error code if the result is 0.
 */
export declare function checkBool(result: number, getLastErrorFn: () => number, context?: string): asserts result is 1;
/**
 * Assert a handle is non-null (non-zero pointer).
 * Throws PdfiumError with the last error code if the handle is 0.
 */
export declare function checkHandle<T extends number>(handle: T, getLastErrorFn: () => number, context?: string): asserts handle is T;
//# sourceMappingURL=error.d.ts.map