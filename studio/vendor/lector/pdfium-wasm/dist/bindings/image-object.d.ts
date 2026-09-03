import type { FpdfBitmap, FpdfDocument, FpdfPage, FpdfPageObject, WasmPointer } from '../types/handles.js';
export declare const imageObjectDescriptor: {
    readonly FPDFImageObj_LoadJpegFile: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFImageObj_LoadJpegFileInline: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFImageObj_SetMatrix: readonly ["number", readonly ["number", "number", "number", "number", "number", "number", "number"]];
    readonly FPDFImageObj_SetBitmap: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFImageObj_GetBitmap: readonly ["number", readonly ["number"]];
    readonly FPDFImageObj_GetRenderedBitmap: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFImageObj_GetImageDataDecoded: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFImageObj_GetImageDataRaw: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFImageObj_GetImageFilterCount: readonly ["number", readonly ["number"]];
    readonly FPDFImageObj_GetImageFilter: readonly ["number", readonly ["number", "number", "number", "number"]];
    readonly FPDFImageObj_GetImageMetadata: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFImageObj_GetImagePixelSize: readonly ["number", readonly ["number", "number", "number"]];
    readonly FPDFImageObj_GetIccProfileDataDecoded: readonly ["number", readonly ["number", "number", "number", "number", "number"]];
};
export interface ImageObjectBindings {
    FPDFImageObj_LoadJpegFile(pages: WasmPointer, count: number, imageObject: FpdfPageObject, fileAccess: WasmPointer): number;
    FPDFImageObj_LoadJpegFileInline(pages: WasmPointer, count: number, imageObject: FpdfPageObject, fileAccess: WasmPointer): number;
    FPDFImageObj_SetMatrix(imageObject: FpdfPageObject, a: number, b: number, c: number, d: number, e: number, f: number): number;
    FPDFImageObj_SetBitmap(pages: WasmPointer, count: number, imageObject: FpdfPageObject, bitmap: FpdfBitmap): number;
    FPDFImageObj_GetBitmap(imageObject: FpdfPageObject): FpdfBitmap;
    FPDFImageObj_GetRenderedBitmap(document: FpdfDocument, page: FpdfPage, imageObject: FpdfPageObject): FpdfBitmap;
    FPDFImageObj_GetImageDataDecoded(imageObject: FpdfPageObject, buffer: WasmPointer, buflen: number): number;
    FPDFImageObj_GetImageDataRaw(imageObject: FpdfPageObject, buffer: WasmPointer, buflen: number): number;
    FPDFImageObj_GetImageFilterCount(imageObject: FpdfPageObject): number;
    FPDFImageObj_GetImageFilter(imageObject: FpdfPageObject, index: number, buffer: WasmPointer, buflen: number): number;
    FPDFImageObj_GetImageMetadata(imageObject: FpdfPageObject, page: FpdfPage, metadata: WasmPointer): number;
    FPDFImageObj_GetImagePixelSize(imageObject: FpdfPageObject, width: WasmPointer, height: WasmPointer): number;
    FPDFImageObj_GetIccProfileDataDecoded(imageObject: FpdfPageObject, page: FpdfPage, buffer: WasmPointer, buflen: number, outBuflen: WasmPointer): number;
}
//# sourceMappingURL=image-object.d.ts.map