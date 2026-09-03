/** Serializable error envelope for cross-worker error propagation. */
export interface SerializedPdfiumError {
    readonly name: 'PdfiumError';
    readonly message: string;
    readonly code: number;
    readonly context?: string;
}
/** Serialize an unknown error into a cross-boundary envelope. */
export declare function serializePdfiumError(error: unknown): SerializedPdfiumError;
/** Type guard: is the value a SerializedPdfiumError envelope? */
export declare function isSerializedPdfiumError(value: unknown): value is SerializedPdfiumError;
//# sourceMappingURL=errors.d.ts.map