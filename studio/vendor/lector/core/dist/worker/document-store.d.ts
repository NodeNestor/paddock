import type { FpdfDocument, FpdfFormHandle, PdfiumInstance, WasmAlloc } from '@truespar/lector-pdfium-wasm';
import type { DocumentId } from '../types/handle-id.js';
import type { PageSize } from '../types/render.js';
/** Internal state for a single open PDF document. */
export interface DocumentState {
    /** Pdfium document handle (opaque WASM pointer). */
    readonly docHandle: FpdfDocument;
    /** Form-fill environment handle for form field access and rendering. */
    readonly formHandle: FpdfFormHandle;
    /** WASM heap allocation for the FPDF_FORMFILLINFO struct (must outlive formHandle). */
    readonly formInfoAlloc: WasmAlloc;
    /** Cached page count. */
    readonly pageCount: number;
    /** Cached dimensions for every page (indexed by page number). */
    readonly pageSizes: ReadonlyArray<PageSize>;
    /**
     * PDF bytes allocation on the WASM heap.
     * Must remain alive for the entire lifetime of the document because
     * pdfium's FPDF_LoadMemDocument does NOT copy the data.
     *
     * `null` for progressively-loaded (linearized) documents, which stream
     * via byte-range requests rather than holding one contiguous buffer.
     * Features that need the full resident bytes (e.g. signature validation)
     * must guard against null.
     */
    readonly pdfAlloc: WasmAlloc | null;
    /**
     * Hex-encoded SHA-256 of the original PDF bytes, computed once at
     * open time. Used by features like document comparison to short-
     * circuit on byte-identical inputs without doing any further work.
     */
    readonly sha256: string;
}
/**
 * Manages open PDF documents in the worker.
 *
 * Wraps a HandleRegistry to provide type-safe document lifecycle management.
 * When a document is released or the store is disposed, the pdfium document
 * is closed and the WASM heap allocation for the PDF bytes is freed.
 */
export declare class DocumentStore implements Disposable {
    #private;
    constructor(pdfium: PdfiumInstance);
    /** Register a new document and return its opaque ID. */
    register(state: DocumentState): DocumentId;
    /** Resolve a document ID to its internal state. Throws if not found. */
    resolve(docId: DocumentId): DocumentState;
    /** Close a document by ID, freeing all associated resources. */
    release(docId: DocumentId): void;
    /** Check whether a document ID is valid. */
    has(docId: DocumentId): boolean;
    /**
     * Update cached page info after a page operation (insert, delete, move, rotate).
     * The caller must pass freshly-read page sizes from pdfium.
     */
    updatePageInfo(docId: DocumentId, pageSizes: ReadonlyArray<PageSize>): void;
    /** Number of currently open documents. */
    get size(): number;
    /** Close all open documents and free all WASM resources. */
    [Symbol.dispose](): void;
}
//# sourceMappingURL=document-store.d.ts.map