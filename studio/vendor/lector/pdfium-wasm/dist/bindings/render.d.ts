import type { FpdfBitmap, FpdfPage, WasmPointer } from '../types/handles.js';
export declare const renderDescriptor: {
    readonly FPDF_RenderPageBitmap: readonly [null, readonly ["number", "number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDF_RenderPageBitmapWithMatrix: readonly [null, readonly ["number", "number", "number", "number", "number"]];
    readonly FPDF_RenderPageBitmapWithColorScheme_Start: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDF_RenderPageBitmap_Start: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDF_RenderPage_Continue: readonly ["number", readonly ["number", "number"]];
    readonly FPDF_RenderPage_Close: readonly [null, readonly ["number"]];
};
export interface RenderBindings {
    FPDF_RenderPageBitmap(bitmap: FpdfBitmap, page: FpdfPage, startX: number, startY: number, sizeX: number, sizeY: number, rotate: number, flags: number): void;
    FPDF_RenderPageBitmapWithMatrix(bitmap: FpdfBitmap, page: FpdfPage, matrix: WasmPointer, clipping: WasmPointer, flags: number): void;
    FPDF_RenderPageBitmapWithColorScheme_Start(bitmap: FpdfBitmap, page: FpdfPage, startX: number, startY: number, sizeX: number, sizeY: number, rotate: number, flags: number, colorScheme: WasmPointer, pause: WasmPointer): number;
    FPDF_RenderPageBitmap_Start(bitmap: FpdfBitmap, page: FpdfPage, startX: number, startY: number, sizeX: number, sizeY: number, rotate: number, flags: number, pause: WasmPointer): number;
    FPDF_RenderPage_Continue(page: FpdfPage, pause: WasmPointer): number;
    FPDF_RenderPage_Close(page: FpdfPage): void;
}
//# sourceMappingURL=render.d.ts.map