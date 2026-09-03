import type { FpdfDocument, FpdfPageRange, WasmPointer } from '../types/handles.js';
export declare const viewerRefDescriptor: {
    readonly FPDF_VIEWERREF_GetPrintScaling: readonly ["number", readonly ["number"]];
    readonly FPDF_VIEWERREF_GetNumCopies: readonly ["number", readonly ["number"]];
    readonly FPDF_VIEWERREF_GetPrintPageRange: readonly ["number", readonly ["number"]];
    readonly FPDF_VIEWERREF_GetPrintPageRangeCount: readonly ["number", readonly ["number"]];
    readonly FPDF_VIEWERREF_GetPrintPageRangeElement: readonly ["number", readonly ["number", "number"]];
    readonly FPDF_VIEWERREF_GetDuplex: readonly ["number", readonly ["number"]];
    readonly FPDF_VIEWERREF_GetName: readonly ["number", readonly ["number", "number", "number", "number"]];
};
export interface ViewerRefBindings {
    FPDF_VIEWERREF_GetPrintScaling(document: FpdfDocument): number;
    FPDF_VIEWERREF_GetNumCopies(document: FpdfDocument): number;
    FPDF_VIEWERREF_GetPrintPageRange(document: FpdfDocument): FpdfPageRange;
    FPDF_VIEWERREF_GetPrintPageRangeCount(pagerange: FpdfPageRange): number;
    FPDF_VIEWERREF_GetPrintPageRangeElement(pagerange: FpdfPageRange, index: number): number;
    FPDF_VIEWERREF_GetDuplex(document: FpdfDocument): number;
    FPDF_VIEWERREF_GetName(document: FpdfDocument, key: WasmPointer, buffer: WasmPointer, length: number): number;
}
//# sourceMappingURL=viewer-ref.d.ts.map