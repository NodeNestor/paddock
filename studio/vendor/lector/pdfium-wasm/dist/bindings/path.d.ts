import type { FpdfPageObject, FpdfPathSegment, WasmPointer } from '../types/handles.js';
export declare const pathDescriptor: {
    readonly FPDFPath_CountSegments: readonly ["number", readonly ["number"]];
    readonly FPDFPath_GetPathSegment: readonly ["number", readonly ["number", "number"]];
    readonly FPDFPath_MoveTo: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFPath_LineTo: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFPath_BezierTo: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDFPath_Close: readonly ["number", readonly ["number"]];
    readonly FPDFPath_SetDrawMode: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFPath_GetDrawMode: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFPathSegment_GetPoint: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFPathSegment_GetType: readonly ["number", readonly ["number"]];
    readonly FPDFPathSegment_GetClose: readonly ["number", readonly ["number"]];
};
export interface PathBindings {
    FPDFPath_CountSegments(path: FpdfPageObject): number;
    FPDFPath_GetPathSegment(path: FpdfPageObject, index: number): FpdfPathSegment;
    FPDFPath_MoveTo(path: FpdfPageObject, x: number, y: number): number;
    FPDFPath_LineTo(path: FpdfPageObject, x: number, y: number): number;
    FPDFPath_BezierTo(path: FpdfPageObject, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number): number;
    FPDFPath_Close(path: FpdfPageObject): number;
    FPDFPath_SetDrawMode(path: FpdfPageObject, fillmode: number, stroke: number): number;
    FPDFPath_GetDrawMode(path: FpdfPageObject, fillmode: WasmPointer, stroke: WasmPointer): number;
    FPDFPathSegment_GetPoint(segment: FpdfPathSegment, x: WasmPointer, y: WasmPointer): number;
    FPDFPathSegment_GetType(segment: FpdfPathSegment): number;
    FPDFPathSegment_GetClose(segment: FpdfPathSegment): number;
}
//# sourceMappingURL=path.d.ts.map