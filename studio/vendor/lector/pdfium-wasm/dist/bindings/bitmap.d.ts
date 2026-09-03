import type { FpdfBitmap, WasmPointer } from '../types/handles.js';
export declare const bitmapDescriptor: {
    readonly FPDFBitmap_Create: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFBitmap_CreateEx: readonly ["number", readonly ["number", "number", "number", "number", "number"]];
    readonly FPDFBitmap_FillRect: readonly ["number", readonly ["number", "number", "number", "number", "number", "number"]];
    readonly FPDFBitmap_GetBuffer: readonly ["number", readonly ["number"]];
    readonly FPDFBitmap_GetWidth: readonly ["number", readonly ["number"]];
    readonly FPDFBitmap_GetHeight: readonly ["number", readonly ["number"]];
    readonly FPDFBitmap_GetStride: readonly ["number", readonly ["number"]];
    readonly FPDFBitmap_GetFormat: readonly ["number", readonly ["number"]];
    readonly FPDFBitmap_Destroy: readonly [null, readonly ["number"]];
};
export interface BitmapBindings {
    FPDFBitmap_Create(width: number, height: number, alpha: number): FpdfBitmap;
    FPDFBitmap_CreateEx(width: number, height: number, format: number, firstScan: WasmPointer, stride: number): FpdfBitmap;
    FPDFBitmap_FillRect(bitmap: FpdfBitmap, left: number, top: number, width: number, height: number, color: number): number;
    FPDFBitmap_GetBuffer(bitmap: FpdfBitmap): WasmPointer;
    FPDFBitmap_GetWidth(bitmap: FpdfBitmap): number;
    FPDFBitmap_GetHeight(bitmap: FpdfBitmap): number;
    FPDFBitmap_GetStride(bitmap: FpdfBitmap): number;
    FPDFBitmap_GetFormat(bitmap: FpdfBitmap): number;
    FPDFBitmap_Destroy(bitmap: FpdfBitmap): void;
}
//# sourceMappingURL=bitmap.d.ts.map