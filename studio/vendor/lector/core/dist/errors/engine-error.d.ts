/** Error codes for the Lector rendering engine. */
export declare const EngineErrorCode: {
    readonly UNKNOWN: "UNKNOWN";
    readonly FILE: "FILE";
    readonly FORMAT: "FORMAT";
    readonly PASSWORD_REQUIRED: "PASSWORD_REQUIRED";
    readonly SECURITY: "SECURITY";
    readonly PAGE: "PAGE";
    readonly NOT_INITIALIZED: "NOT_INITIALIZED";
    readonly ALREADY_DESTROYED: "ALREADY_DESTROYED";
    readonly DOCUMENT_NOT_FOUND: "DOCUMENT_NOT_FOUND";
    readonly WORKER_TERMINATED: "WORKER_TERMINATED";
    readonly RENDER_ABORTED: "RENDER_ABORTED";
};
export type EngineErrorCode = (typeof EngineErrorCode)[keyof typeof EngineErrorCode];
/** Structured error for all Lector engine failures. */
export declare class EngineError extends Error {
    readonly name = "EngineError";
    readonly code: EngineErrorCode;
    readonly pdfiumCode?: number;
    constructor(code: EngineErrorCode, message: string, pdfiumCode?: number);
}
/**
 * Deserialize an error received from the worker into a typed EngineError.
 *
 * Handles:
 * - SerializedPdfiumError envelopes (from worker-side pdfium failures)
 * - EngineError instances (passthrough)
 * - Unknown errors (wrapped as UNKNOWN)
 */
export declare function fromSerializedError(err: unknown): EngineError;
//# sourceMappingURL=engine-error.d.ts.map