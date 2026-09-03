import type { FpdfAvail, FpdfDocument, WasmPointer } from '../types/handles.js';
export declare const availDescriptor: {
    readonly FPDFAvail_Create: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAvail_Destroy: readonly [null, readonly ["number"]];
    readonly FPDFAvail_IsDocAvail: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAvail_GetDocument: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAvail_GetFirstPageNum: readonly ["number", readonly ["number"]];
    readonly FPDFAvail_IsPageAvail: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFAvail_IsFormAvail: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAvail_IsLinearized: readonly ["number", readonly ["number"]];
};
export interface AvailBindings {
    FPDFAvail_Create(fileAvail: WasmPointer, file: WasmPointer): FpdfAvail;
    FPDFAvail_Destroy(avail: FpdfAvail): void;
    FPDFAvail_IsDocAvail(avail: FpdfAvail, hints: WasmPointer): number;
    FPDFAvail_GetDocument(avail: FpdfAvail, password: WasmPointer): FpdfDocument;
    FPDFAvail_GetFirstPageNum(doc: FpdfDocument): number;
    FPDFAvail_IsPageAvail(avail: FpdfAvail, pageIndex: number, hints: WasmPointer): number;
    FPDFAvail_IsFormAvail(avail: FpdfAvail, hints: WasmPointer): number;
    FPDFAvail_IsLinearized(avail: FpdfAvail): number;
}
//# sourceMappingURL=avail.d.ts.map