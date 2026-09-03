import type { FpdfAction, FpdfBookmark, FpdfDest, FpdfDocument, WasmPointer } from '../types/handles.js';
export declare const bookmarkDescriptor: {
    readonly FPDFBookmark_GetFirstChild: readonly ["number", readonly ["number", "number"]];
    readonly FPDFBookmark_GetNextSibling: readonly ["number", readonly ["number", "number"]];
    readonly FPDFBookmark_GetTitle: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFBookmark_GetCount: readonly ["number", readonly ["number"]];
    readonly FPDFBookmark_Find: readonly ["number", readonly ["number", "number"]];
    readonly FPDFBookmark_GetDest: readonly ["number", readonly ["number", "number"]];
    readonly FPDFBookmark_GetAction: readonly ["number", readonly ["number"]];
};
export interface BookmarkBindings {
    FPDFBookmark_GetFirstChild(document: FpdfDocument, bookmark: FpdfBookmark): FpdfBookmark;
    FPDFBookmark_GetNextSibling(document: FpdfDocument, bookmark: FpdfBookmark): FpdfBookmark;
    FPDFBookmark_GetTitle(bookmark: FpdfBookmark, buffer: WasmPointer, buflen: number): number;
    FPDFBookmark_GetCount(bookmark: FpdfBookmark): number;
    FPDFBookmark_Find(document: FpdfDocument, title: WasmPointer): FpdfBookmark;
    FPDFBookmark_GetDest(document: FpdfDocument, bookmark: FpdfBookmark): FpdfDest;
    FPDFBookmark_GetAction(bookmark: FpdfBookmark): FpdfAction;
}
//# sourceMappingURL=bookmark.d.ts.map