import type { PageSize } from './render.js';
/** Metadata and structure information for an open PDF document. */
export interface DocumentInfo {
    /** Total number of pages in the document. */
    readonly pageCount: number;
    /** Dimensions of every page in PDF points, indexed by page number. */
    readonly pageSizes: ReadonlyArray<PageSize>;
    /** PDF specification version (e.g. 17 for PDF 1.7). 0 if unavailable. */
    readonly fileVersion: number;
    /** Document permission flags from the PDF security handler. */
    readonly permissions: number;
    /** Form type: 0 = none, 1 = AcroForm, 2 = XFA full, 3 = XFA foreground. */
    readonly formType: number;
}
//# sourceMappingURL=document.d.ts.map