import type { FpdfDocument, PdfiumInstance } from '@truespar/lector-pdfium-wasm';
/** A node in the PDF bookmark (outline) tree. */
export interface BookmarkNode {
    readonly title: string;
    readonly pageIndex: number | null;
    readonly children: BookmarkNode[];
}
/** Destination within the document (page + optional position/zoom). */
export interface DestinationInfo {
    readonly pageIndex: number;
    readonly x: number | null;
    readonly y: number | null;
    readonly zoom: number | null;
}
/** Types of link targets. */
export type LinkTarget = {
    readonly type: 'goto';
    readonly destination: DestinationInfo;
} | {
    readonly type: 'uri';
    readonly uri: string;
} | {
    readonly type: 'remote-goto';
    readonly filePath: string;
    readonly destination: DestinationInfo | null;
} | {
    readonly type: 'launch';
    readonly filePath: string;
} | {
    readonly type: 'unknown';
};
/** A link annotation on a page. */
export interface PageLink {
    readonly rect: {
        readonly left: number;
        readonly top: number;
        readonly right: number;
        readonly bottom: number;
    };
    readonly target: LinkTarget;
}
/** A web link detected from page text (auto-detected URL). */
export interface WebLink {
    readonly url: string;
    readonly rects: Array<{
        readonly left: number;
        readonly top: number;
        readonly right: number;
        readonly bottom: number;
    }>;
}
/** Read the full bookmark (outline / table of contents) tree. */
export declare function readBookmarkTree(pdfium: PdfiumInstance, docHandle: FpdfDocument): BookmarkNode[];
/** Read all link annotations from a page. */
export declare function readPageLinks(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number): PageLink[];
/** Read auto-detected web links (URLs) from a page's text content. */
export declare function readPageWebLinks(pdfium: PdfiumInstance, docHandle: FpdfDocument, pageIndex: number): WebLink[];
/**
 * Get the total number of pages in the document.
 * Convenience wrapper used by navigation operations.
 */
export declare function getDocumentPageCount(pdfium: PdfiumInstance, docHandle: FpdfDocument): number;
//# sourceMappingURL=navigation-ops.d.ts.map