import type { DocumentId, TaskId } from './handle-id.js';
import type { PageSize, RenderOptions } from './render.js';
import type { AnnotationData, WidgetData } from '../data/types.js';
import type { TextCharInfo, TextRect, TextSearchMatch } from '../worker/text-ops.js';
import type { BookmarkNode, PageLink, WebLink } from '../worker/navigation-ops.js';
import type { SignatureInfo } from '../worker/signature-ops.js';
import type { AttachmentInfo } from '../worker/attachment-ops.js';
import type { LayerInfo } from '../worker/layer-ops.js';
/**
 * A single redaction to apply on a page. Carries the geometry plus the
 * fill/overlay styling so the destructive operation is fully described by
 * value and can be replayed identically on any worker.
 */
export interface RedactionSpec {
    /** Redaction rect in PDF page coordinates (origin bottom-left). */
    readonly rect: {
        left: number;
        bottom: number;
        right: number;
        top: number;
    };
    /** Box fill color. Defaults to opaque black. */
    readonly fillColor?: {
        r: number;
        g: number;
        b: number;
    };
    /** Optional text drawn inside the box after the content is removed. */
    readonly overlayText?: string;
    /** Overlay text color. Defaults to white. */
    readonly overlayColor?: {
        r: number;
        g: number;
        b: number;
    };
    /** Overlay font size in points. Auto-fit to the box height when omitted. */
    readonly overlayFontSize?: number;
}
/**
 * Comlink-compatible contract for the pdfium Web Worker.
 *
 * Every method is async because all calls cross the worker boundary.
 * Transferable values (ArrayBuffer, ImageBitmap) are handled by Comlink's
 * transfer proxy automatically.
 */
export interface PdfiumWorkerApi {
    /** Initialize the pdfium WASM module inside the worker. */
    init(wasmUrl: string, wasmJsUrl: string): Promise<void>;
    /** Load a PDF from raw bytes. Returns an opaque document handle. */
    openDocument(data: ArrayBuffer, password?: string): Promise<DocumentId>;
    /** Close a document and free all associated WASM resources. */
    closeDocument(docId: DocumentId): Promise<void>;
    /** Get the total number of pages in a document. */
    getPageCount(docId: DocumentId): Promise<number>;
    /** Get the dimensions (in PDF points) of a single page. */
    getPageSize(docId: DocumentId, pageIndex: number): Promise<PageSize>;
    /** Get the dimensions of every page in the document. */
    getAllPageSizes(docId: DocumentId): Promise<ReadonlyArray<PageSize>>;
    /**
     * Render a page to an ImageBitmap at the specified pixel dimensions.
     *
     * The returned ImageBitmap is transferred (zero-copy) from the worker
     * to the main thread. The caller owns it and must call `.close()` when done.
     */
    renderPage(docId: DocumentId, pageIndex: number, width: number, height: number, options?: RenderOptions): Promise<ImageBitmap>;
    /**
     * Render a rectangular tile of a page. Same quality as `renderPage`
     * but only allocates memory for the tile (e.g. 512×512) instead of
     * the full page. Used by tile-based rendering for large pages at
     * high zoom.
     */
    renderPageTile(docId: DocumentId, pageIndex: number, tileX: number, tileY: number, tileW: number, tileH: number, fullW: number, fullH: number, options?: RenderOptions): Promise<ImageBitmap>;
    /** Cancel a pending task. Returns true if the task was found and cancelled. */
    cancelTask(taskId: TaskId): Promise<boolean>;
    /** Read all annotations from a page. */
    getAnnotations(docId: DocumentId, pageIndex: number): Promise<AnnotationData[]>;
    /** Create a new annotation on a page. Returns the created annotation data. */
    createAnnotation(docId: DocumentId, pageIndex: number, data: Partial<AnnotationData>): Promise<AnnotationData>;
    /** Update an existing annotation. Returns the updated data. */
    updateAnnotation(docId: DocumentId, pageIndex: number, annotIndex: number, patch: Partial<AnnotationData>): Promise<AnnotationData>;
    /** Delete an annotation from a page. */
    deleteAnnotation(docId: DocumentId, pageIndex: number, annotIndex: number): Promise<void>;
    /** Read all form fields (widget annotations) from a page. */
    getFormFields(docId: DocumentId, pageIndex: number): Promise<WidgetData[]>;
    /** Set a form field value by field name (text fields, comboboxes, listboxes). */
    setFormFieldValue(docId: DocumentId, pageIndex: number, fieldName: string, value: string): Promise<void>;
    /** Set a combobox/listbox selection by annotation index and option index. */
    setComboBoxByIndex(docId: DocumentId, pageIndex: number, annotIndex: number, optionIndex: number): Promise<void>;
    /**
     * Simulate a mouse click on a form widget at the given page coordinates.
     * Uses the shared form handle for both interaction and rendering.
     * Returns the updated form field data for the page.
     */
    clickFormWidget(docId: DocumentId, pageIndex: number, pageX: number, pageY: number): Promise<WidgetData[]>;
    /** Save the document as a new PDF. Returns the PDF bytes. */
    saveAsCopy(docId: DocumentId): Promise<ArrayBuffer>;
    /** Set AES-256 password protection on a document. Call before saveAsCopy(). */
    setDocumentPassword(docId: DocumentId, options: import('../worker/security-ops.js').PasswordProtectionOptions): Promise<void>;
    /** Export annotations as XFDF XML string. */
    exportXfdf(docId: DocumentId): Promise<string>;
    /** Merge multiple open documents into a single new PDF. */
    mergeDocuments(docIds: DocumentId[]): Promise<ArrayBuffer>;
    /** Split a document into multiple PDFs by page ranges (0-based inclusive). */
    splitDocument(docId: DocumentId, ranges: Array<{
        start: number;
        end: number;
    }>): Promise<ArrayBuffer[]>;
    /** Extract specific pages from a document into a new PDF. */
    extractPages(docId: DocumentId, pageIndices: number[]): Promise<ArrayBuffer>;
    /**
     * Compare two open documents and return a structured diff. Both
     * documents must already be open via `openDocument()`. Pages are
     * aligned via LCS on text fingerprints, then compared word-by-word
     * for text pages or pixel-region for image-only pages.
     */
    compareDocuments(docIdA: DocumentId, docIdB: DocumentId): Promise<import('../worker/comparison-ops.js').ComparisonResult>;
    /**
     * Get the hex-encoded SHA-256 of an open document's original bytes.
     * Returns the empty string for documents loaded via the linearised
     * path (no contiguous buffer to hash).
     */
    getDocumentHash(docId: DocumentId): Promise<string>;
    /** Extract all text from a page as a single string. */
    getPageText(docId: DocumentId, pageIndex: number): Promise<string>;
    /** Extract per-character position, bounding box, and font info for a page. */
    getPageCharInfo(docId: DocumentId, pageIndex: number): Promise<TextCharInfo[]>;
    /** Search for text on a page. Returns all matches with bounding rects. */
    searchPage(docId: DocumentId, pageIndex: number, query: string, flags: number): Promise<TextSearchMatch[]>;
    /** Get bounding rectangles for a range of characters on a page. */
    getTextRects(docId: DocumentId, pageIndex: number, charIndex: number, count: number): Promise<TextRect[]>;
    /** Get the character index at a given position on a page. */
    getCharIndexAtPos(docId: DocumentId, pageIndex: number, x: number, y: number, tolerance: number): Promise<number>;
    /** Read the bookmark (outline / table of contents) tree. */
    getBookmarks(docId: DocumentId): Promise<BookmarkNode[]>;
    /** Add a top-level bookmark. */
    addBookmark(docId: DocumentId, title: string, pageIndex: number, insertIndex: number): Promise<boolean>;
    /** Delete a top-level bookmark by index. */
    deleteBookmark(docId: DocumentId, index: number): Promise<boolean>;
    /** Move a top-level bookmark from one index to another. */
    moveBookmark(docId: DocumentId, fromIndex: number, toIndex: number): Promise<boolean>;
    /** Set the title of a top-level bookmark. */
    setBookmarkTitle(docId: DocumentId, index: number, title: string): Promise<boolean>;
    /** Set the destination page of a top-level bookmark. */
    setBookmarkDest(docId: DocumentId, index: number, pageIndex: number): Promise<boolean>;
    /** Read all link annotations from a page. */
    getPageLinks(docId: DocumentId, pageIndex: number): Promise<PageLink[]>;
    /** Read auto-detected web links from a page's text content. */
    getPageWebLinks(docId: DocumentId, pageIndex: number): Promise<WebLink[]>;
    /** Delete a page from the document. */
    deletePage(docId: DocumentId, pageIndex: number): Promise<void>;
    /** Insert a new blank page at the given index. */
    insertBlankPage(docId: DocumentId, pageIndex: number, width: number, height: number): Promise<void>;
    /** Set page rotation (0=0°, 1=90°, 2=180°, 3=270°). */
    rotatePage(docId: DocumentId, pageIndex: number, rotation: number): Promise<void>;
    /** Get page rotation (0=0°, 1=90°, 2=180°, 3=270°). */
    getPageRotation(docId: DocumentId, pageIndex: number): Promise<number>;
    /** Move a page from one index to another. */
    movePage(docId: DocumentId, fromIndex: number, toIndex: number): Promise<void>;
    /** Duplicate a page (inserts copy after the source page). */
    duplicatePage(docId: DocumentId, pageIndex: number): Promise<void>;
    /** Flatten annotations on a page into the page content. Returns 0=ok, 1=nothing, 2=fail. */
    flattenPage(docId: DocumentId, pageIndex: number): Promise<number>;
    /**
     * Apply redactions on a page. For every spec, destructively removes all
     * page objects that overlap its rect — including objects nested inside
     * Form XObjects — then draws the fill box and optional overlay text and
     * regenerates the content stream.
     *
     * Specs are passed explicitly (rather than read from the page's
     * annotations) so the primary worker and every render-pool worker apply
     * the byte-identical mutation. Pool workers never received the redaction
     * annotations, so they could not derive the rects themselves.
     *
     * @param removeAnnots When true, also deletes the redaction annotations
     *   from the page (primary worker only — pool copies have none).
     */
    applyRedactions(docId: DocumentId, pageIndex: number, specs: readonly RedactionSpec[], removeAnnots: boolean): Promise<void>;
    /** Get the number of digital signatures in a document. */
    getSignatureCount(docId: DocumentId): Promise<number>;
    /** Read detailed information about a specific signature. */
    getSignatureInfo(docId: DocumentId, sigIndex: number): Promise<SignatureInfo>;
    /** Get the number of embedded file attachments. */
    getAttachmentCount(docId: DocumentId): Promise<number>;
    /** Read metadata for a specific attachment. */
    getAttachmentInfo(docId: DocumentId, index: number): Promise<AttachmentInfo>;
    /** Read the file content of an attachment. */
    getAttachmentData(docId: DocumentId, index: number): Promise<ArrayBuffer>;
    /** Add a new file attachment to the document. */
    addAttachment(docId: DocumentId, name: string, data: ArrayBuffer): Promise<void>;
    /** Delete an attachment by index. */
    deleteAttachment(docId: DocumentId, index: number): Promise<void>;
    /** Get all layers (Optional Content Groups) in the document. */
    getLayers(docId: DocumentId): Promise<LayerInfo[]>;
    /** Set the visibility of a layer by index. Triggers re-render. */
    setLayerVisible(docId: DocumentId, layerIndex: number, visible: boolean): Promise<void>;
    /** Validate a PKCS#7/CMS signature cryptographically. */
    validateSignature(docId: DocumentId, sigIndex: number, pdfBytes: ArrayBuffer): Promise<import('../worker/crypto-ops.js').SignatureValidationResult>;
    /**
     * Validate every signature in a document in a single round-trip.
     * The PDF bytes should be marked transferable via Comlink.transfer
     * by the caller for zero-copy delivery.
     */
    validateAllSignatures(docId: DocumentId, pdfBytes: ArrayBuffer): Promise<import('../worker/crypto-ops.js').SignatureValidationResult[]>;
    /** Digitally sign the document with a PFX/P12 certificate. Returns the signed PDF. */
    signDocument(docId: DocumentId, options: import('../worker/signing-ops.js').SigningOptions): Promise<import('../worker/signing-ops.js').SigningResult>;
    /**
     * Render a sub-region of a page to an `ImageBitmap` at the requested DPI.
     * Used by the marquee-capture tool to grab a screenshot-quality copy of
     * a page region for clipboard copy or PNG download.
     */
    captureRegion(docId: DocumentId, options: import('../worker/capture-ops.js').CaptureRegionOptions): Promise<ImageBitmap>;
    /** Get all document-level JavaScript actions. */
    getJavaScriptActions(docId: DocumentId): Promise<import('../worker/acroform-js-ops.js').PdfJavaScriptAction[]>;
    /** Create a sandboxed JS runtime for form scripts. Returns an opaque handle ID. */
    createJSRuntime(): Promise<number>;
    /** Destroy a JS runtime. */
    destroyJSRuntime(handleId: number): Promise<void>;
    /** Execute a script in the sandbox. Returns true on success. */
    evalScript(handleId: number, script: string): Promise<boolean>;
    /** Read a global variable from the sandbox as a string. */
    getJSGlobal(handleId: number, varName: string): Promise<string | null>;
    /** Create a linearization context for progressive PDF loading. */
    createLinearContext(fileSize: number, initialData: ArrayBuffer): Promise<number>;
    /** Feed a data chunk into the linearization context. */
    feedLinearData(contextId: number, offset: number, data: ArrayBuffer): Promise<void>;
    /** Check if the PDF is linearized (1=yes, 0=no, -1=need more data). */
    isLinearized(contextId: number): Promise<number>;
    /** Check if document structure is available. Returns hints for needed ranges. */
    isDocAvail(contextId: number): Promise<{
        available: boolean;
        hints: Array<{
            offset: number;
            length: number;
        }>;
    }>;
    /** Check if a page is available. Returns hints for needed ranges. */
    isPageAvail(contextId: number, pageIndex: number): Promise<{
        available: boolean;
        hints: Array<{
            offset: number;
            length: number;
        }>;
    }>;
    /** Get the document handle from a linearization context. Returns DocumentId. */
    getLinearDocument(contextId: number, password?: string): Promise<DocumentId>;
    /** Get the first available page index. */
    getLinearFirstPage(contextId: number): Promise<number>;
    /** Destroy a linearization context (does not close the document). */
    destroyLinearContext(contextId: number): Promise<void>;
    /** Tear down the pdfium instance and release all resources. */
    destroy(): Promise<void>;
}
//# sourceMappingURL=worker-api.d.ts.map