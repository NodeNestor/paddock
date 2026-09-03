import type { FpdfAction, FpdfDest, FpdfDocument, WasmPointer } from '../types/handles.js';
export declare const destDescriptor: {
    readonly FPDFDest_GetDestPageIndex: readonly ["number", readonly ["number", "number"]];
    readonly FPDFDest_GetView: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFDest_GetLocationInPage: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDFAction_GetType: readonly ["number", readonly ["number"]];
    readonly FPDFAction_GetDest: readonly ["number", readonly ["number", "number"]];
    readonly FPDFAction_GetFilePath: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFAction_GetURIPath: readonly ["number", readonly ["number", "number", "number", "number"]];
};
export interface DestBindings {
    FPDFDest_GetDestPageIndex(document: FpdfDocument, dest: FpdfDest): number;
    FPDFDest_GetView(dest: FpdfDest, pNumParams: WasmPointer, pParams: WasmPointer): number;
    FPDFDest_GetLocationInPage(dest: FpdfDest, hasXVal: WasmPointer, hasYVal: WasmPointer, hasZoomVal: WasmPointer, x: WasmPointer, y: WasmPointer, zoom: WasmPointer): number;
    FPDFAction_GetType(action: FpdfAction): number;
    FPDFAction_GetDest(document: FpdfDocument, action: FpdfAction): FpdfDest;
    FPDFAction_GetFilePath(action: FpdfAction, buffer: WasmPointer, buflen: number): number;
    FPDFAction_GetURIPath(document: FpdfDocument, action: FpdfAction, buffer: WasmPointer, buflen: number): number;
}
//# sourceMappingURL=dest.d.ts.map