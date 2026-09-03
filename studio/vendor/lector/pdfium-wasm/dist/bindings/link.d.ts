import type { FpdfAction, FpdfAnnotation, FpdfDest, FpdfDocument, FpdfLink, FpdfPage, FpdfPageLink, FpdfTextPage, WasmPointer } from '../types/handles.js';
export declare const linkDescriptor: {
    readonly FPDFLink_GetLinkAtPoint: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFLink_GetLinkZOrderAtPoint: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFLink_GetDest: readonly ["number", readonly ["number", "number"]];
    readonly FPDFLink_GetAction: readonly ["number", readonly ["number"]];
    readonly FPDFLink_Enumerate: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFLink_GetAnnot: readonly ["number", readonly ["number", "number"]];
    readonly FPDFLink_GetAnnotRect: readonly ["number", readonly ["number", "number"]];
    readonly FPDFLink_CountQuadPoints: readonly ["number", readonly ["number"]];
    readonly FPDFLink_GetQuadPoints: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFLink_LoadWebLinks: readonly ["number", readonly ["number"]];
    readonly FPDFLink_CountWebLinks: readonly ["number", readonly ["number"]];
    readonly FPDFLink_GetURL: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFLink_CountRects: readonly ["number", readonly ["number", "number"]];
    readonly FPDFLink_GetRect: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDFLink_GetTextRange: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFLink_CloseWebLinks: readonly [null, readonly ["number"]];
};
export interface LinkBindings {
    FPDFLink_GetLinkAtPoint(page: FpdfPage, x: number, y: number): FpdfLink;
    FPDFLink_GetLinkZOrderAtPoint(page: FpdfPage, x: number, y: number): number;
    FPDFLink_GetDest(document: FpdfDocument, link: FpdfLink): FpdfDest;
    FPDFLink_GetAction(link: FpdfLink): FpdfAction;
    FPDFLink_Enumerate(page: FpdfPage, startPos: WasmPointer, linkAnnot: WasmPointer): number;
    FPDFLink_GetAnnot(page: FpdfPage, linkAnnot: FpdfLink): FpdfAnnotation;
    FPDFLink_GetAnnotRect(linkAnnot: FpdfLink, rect: WasmPointer): number;
    FPDFLink_CountQuadPoints(linkAnnot: FpdfLink): number;
    FPDFLink_GetQuadPoints(linkAnnot: FpdfLink, quadIndex: number, quadPoints: WasmPointer): number;
    FPDFLink_LoadWebLinks(textPage: FpdfTextPage): FpdfPageLink;
    FPDFLink_CountWebLinks(linkPage: FpdfPageLink): number;
    FPDFLink_GetURL(linkPage: FpdfPageLink, linkIndex: number, buffer: WasmPointer, buflen: number): number;
    FPDFLink_CountRects(linkPage: FpdfPageLink, linkIndex: number): number;
    FPDFLink_GetRect(linkPage: FpdfPageLink, linkIndex: number, rectIndex: number, left: WasmPointer, top: WasmPointer, right: WasmPointer, bottom: WasmPointer): number;
    FPDFLink_GetTextRange(linkPage: FpdfPageLink, linkIndex: number, startCharIndex: WasmPointer, charCount: WasmPointer): number;
    FPDFLink_CloseWebLinks(linkPage: FpdfPageLink): void;
}
//# sourceMappingURL=link.d.ts.map